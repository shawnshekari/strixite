#include "serve/engine.hpp"
#include "serve/think_nudge.hpp"

#include "serve/capture.hpp"
#include "serve/log.hpp"
#include "serve/resume_loss.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <cstdio>

namespace strix {

namespace {
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

Engine::Engine(LmBackend &backend, const Tokenizer &tok, const Options &options)
    : be_(backend),
      tok_(tok),
      options_(options),
      cache_(options.cache),
      capacity_(backend.capacity()),
      im_start_(tok.id_of("<|im_start|>")),
      im_end_(tok.id_of("<|im_end|>")),
      eot_(tok.id_of("<|endoftext|>")),
      think_end_(tok.id_of("</think>")),
      masker_(tok, backend.logits_row()),
      thinking_stop_(tok.encode(kThinkingStop)),
      nudge_{tok.encode(kThinkNudge1), tok.encode(kThinkNudge2)} {
    STRIX_CHECK(backend.logits_row() >= tok.size(), "Engine: logits rows of ", backend.logits_row(),
                " are shorter than the tokenizer's ", tok.size(), " tokens");
    STRIX_CHECK(backend.max_chunk() >= (int64_t)thinking_stop_.size() && backend.capacity() >= 2, "Engine: backend chunk ",
                backend.max_chunk(), " (the thinking stop is ", thinking_stop_.size(), " tokens), capacity ", backend.capacity());
    STRIX_CHECK(std::count(thinking_stop_.begin(), thinking_stop_.end(), tok.id_of("</think>")) == 1,
                "Engine: the thinking stop text doesn't tokenize to exactly one </think>");
    for (const auto &n : nudge_)
        STRIX_CHECK(!n.empty() && (int64_t)n.size() <= backend.max_chunk() &&
                        std::count(n.begin(), n.end(), tok.id_of("</think>")) == 0,
                    "Engine: a thinking nudge tokenizes to ", n.size(), " tokens (backend chunk ", backend.max_chunk(),
                    ") or contains </think> - it must leave the think block open");
    STRIX_CHECK(options_.mtp_margin >= 0.0f, "Engine: mtp_margin ", options_.mtp_margin, ", expected >= 0");
    STRIX_CHECK(options_.mtp_draft >= 1 && options_.mtp_draft <= 15, "Engine: mtp_draft ", options_.mtp_draft,
                ", expected 1..15");
    STRIX_CHECK(options_.capture_dir.empty() || std::filesystem::is_directory(options_.capture_dir),
                "Engine: capture_dir '", options_.capture_dir, "' is not a directory");
    (void)ThinkWatch{options_.think_nudge_policy};  // a bad nudge policy fails here, at startup, not on a request
    be_.reset();
    idle_since_ = now_s();
    thread_ = std::thread([this] { worker(); });
}

Engine::Engine(LmBackend &backend, const Tokenizer &tok, PromptCache *cache, float mtp_margin)
    : Engine(backend, tok, [&] {
          Options o;
          o.cache = cache, o.mtp_margin = mtp_margin;
          return o;
      }()) {}

Engine::~Engine() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    for (Job &j : queue_) {
        GenerationResult r;
        r.finish_reason = "error", r.error = "server shutting down";
        j.sink->on_done(r);
    }
}

void Engine::submit(GenerationRequest req, std::shared_ptr<GenerationSink> sink) {
    STRIX_CHECK(sink != nullptr, "Engine::submit: null sink");
    STRIX_CHECK(!req.prompt.empty(), "Engine::submit: empty prompt");
    STRIX_CHECK(req.thinking_budget >= -1, "Engine::submit: thinking budget ", req.thinking_budget, " (-1 = none)");
    STRIX_CHECK(req.max_tokens >= 1 && (int64_t)req.prompt.size() + req.max_tokens <= capacity_,
                "Engine::submit: prompt ", req.prompt.size(), " + max_tokens ", req.max_tokens, " exceed the capacity ",
                capacity_, " (the API layer clamps)");
    for (int32_t id : req.prompt)
        STRIX_CHECK(id >= 0 && id < tok_.size(), "Engine::submit: prompt token ", id, " outside [0, ", tok_.size(), ")");
    {
        std::lock_guard<std::mutex> lock(mu_);
        STRIX_CHECK(!stop_, "Engine::submit: shutting down");
        // One line from the HTTP thread for a request that waits; its block prints when it starts.
        if (stats_.busy || !queue_.empty())
            slog(LogLevel::Info, "Request %lld queued: %s ahead of it (Request %lld running)", (long long)req.id,
                 fmt_n((long long)queue_.size() + stats_.busy).c_str(), (long long)running_id_);
        queue_.push_back({std::move(req), std::move(sink), now_s()});
        ++stats_.queued;
    }
    cv_.notify_one();
}

EngineStats Engine::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    EngineStats s = stats_;
    s.idle_seconds = s.busy || s.queued ? 0.0 : std::max(0.0, now_s() - idle_since_);
    return s;
}

void Engine::worker() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            job = std::move(queue_.front());
            queue_.pop_front();
            --stats_.queued, stats_.busy = 1;
            running_id_ = job.req.id;
        }
        run(job);
        std::lock_guard<std::mutex> lock(mu_);
        stats_.busy = 0, running_id_ = 0;
        idle_since_ = now_s();
        stats_.live_tokens = (int64_t)seq_.size();
        stats_.snapshot_pos = be_.snapshot_pos(LmBackend::kTurnSlot);
    }
}

int64_t Engine::resume_point(const std::vector<int32_t> &prompt, int64_t id, bool &from_disk, bool &from_ram) {
    from_disk = false, from_ram = false;
    const int64_t P = (int64_t)prompt.size();
    int64_t lcp = 0;
    while (lcp < (int64_t)seq_.size() && lcp < P && seq_[(size_t)lcp] == prompt[(size_t)lcp]) ++lcp;
    // In memory: the live state if the prompt extends it, else the better of the snapshots.
    int64_t best = 0, how = -2;  // -1 = live, >= 0 = slot
    if (lcp == (int64_t)seq_.size() && lcp < P && be_.pos() == lcp) best = lcp, how = -1;
    for (int slot : {LmBackend::kTurnSlot, LmBackend::kSystemSlot, LmBackend::kUserSlot}) {
        const int64_t sp = be_.snapshot_pos(slot);
        if (sp > best && sp <= lcp && sp < P && be_.can_restore_snapshot(slot)) best = sp, how = slot;
    }
    // In the prompt cache (RAM, else disk), if it covers more.
    if (cache_) {
        const int64_t dn = cache_->best(prompt, P - 1);
        if (dn > best) {
            const double t0 = now_s();
            size_t bytes = 0;
            bool ram = false;
            int64_t delta_from = 0;
            const bool loaded = cache_->with_state(prompt, dn, [&](const HostBuffer &state, int64_t from, int64_t to) {
                be_.import_state(state, to, from);  // a delta entry: its base (0, base), then the delta (base, dn)
                bytes += state.size();
                if (from > 0) delta_from = from;
            }, &ram);
            if (loaded) {
                slog_row(LogLevel::Info, "resume", "%s of %s tokens from the prompt cache's %s", fmt_n(dn).c_str(),
                         fmt_n(P).c_str(), ram ? "RAM" : "disk");
                const std::string what = delta_from > 0 ? "the entry of " + fmt_n(delta_from) + " tokens + a delta"
                                                         : std::string("a whole entry");
                slog_row(LogLevel::Info, "", "%s, %s MB loaded in %s ms", what.c_str(), fmt_rate(bytes / 1e6, 0).c_str(),
                         fmt_rate((now_s() - t0) * 1e3, 0).c_str());
                from_ram = ram;
                seq_.assign(prompt.begin(), prompt.begin() + dn);
                from_disk = true;
                return dn;
            }
        }
    }
    (void)id;
    const auto none = [&] {
        slog_row(LogLevel::Info, "resume", "none: no saved state is a prefix of this prompt - all %s tokens prefilled",
                 fmt_n(P).c_str());
    };
    if (how == -1 && best > 0) {
        slog_row(LogLevel::Info, "resume", "%s of %s tokens from the live session (where the last request ended)",
                 fmt_n(best).c_str(), fmt_n(P).c_str());
        return best;
    }
    if (how == -1) {  // the live session, but nothing of it matches
        none();
        return 0;
    }
    if (how >= 0) {
        be_.restore_snapshot((int)how);
        seq_.resize((size_t)best);
        slog_row(LogLevel::Info, "resume", "%s of %s tokens from the GPU snapshot at %s", fmt_n(best).c_str(),
                 fmt_n(P).c_str(),
                 how == LmBackend::kTurnSlot     ? "the previous prompt's end"
                 : how == LmBackend::kSystemSlot ? "the system prompt's end"
                                                 : "the start of the previous prompt's last user message");
        return best;
    }
    be_.reset();
    seq_.clear();
    none();
    return 0;
}

void Engine::capture(const GenerationRequest &req, const std::vector<CaptureStep> &steps, const std::string &reason) {
    // A side channel: a failure here is logged as an error and never touches the session or the response.
    try {
        const int64_t P = (int64_t)req.prompt.size();
        STRIX_CHECK((int64_t)seq_.size() >= P && std::equal(req.prompt.begin(), req.prompt.end(), seq_.begin()),
                    "Engine::capture: the session (", seq_.size(), " tokens) doesn't start with the ", P, "-token prompt");
        CaptureRecord rec;
        rec.id = req.id, rec.prompt_n = P, rec.tokens = seq_, rec.steps = steps;
        rec.tools = req.parser.tools && req.parser.tools->is_array() ? (int64_t)req.parser.tools->as_array("tools").size() : 0;
        rec.one_shot = req.one_shot, rec.user_turn = req.user_turn, rec.finish = reason;
        rec.mtp_draft = options_.mtp_draft, rec.mtp_margin = options_.mtp_margin;
        const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
        write_capture(options_.capture_dir, rec, ms);
    } catch (const std::exception &e) {
        slog_row(LogLevel::Error, "capture", "ERROR: failed: %s", e.what());
    }
}

void Engine::report_resume_loss(const std::vector<int32_t> &prompt, int64_t start, int64_t live_common, int64_t live_n,
                                int64_t id) {
    const int64_t P = (int64_t)prompt.size();
    if (P - 1 - start < kResumeLossReportTokens) return;  // nothing big enough to lose: skip the cache scan
    int64_t common = live_common, their_n = live_n;
    std::string what = "the live session";
    if (cache_) {
        const PromptCache::Nearest n = cache_->nearest(prompt);
        if (n.common > common) {
            common = n.common, their_n = n.tokens;
            what = std::string("the prompt cache's ") +
                   (n.kind == PromptCache::Kind::System ? "system-prefix" : n.kind == PromptCache::Kind::Checkpoint ? "user-turn" : "turn") +
                   " entry (" + (n.in_ram ? "RAM" : "disk") + ")";
        }
    }
    const std::string text = resume_loss_text(prompt, start, common, their_n, what, im_start_);
    if (text.empty()) return;
    const int64_t lost = std::min(common, P - 1) - start;
    const bool warn = lost >= kResumeLossWarnTokens;
    // "resume lost N tokens: why" - the label says "resume"; the why goes on its own row.
    std::string head = text.rfind("resume ", 0) == 0 ? text.substr(7) : text, why;
    if (const size_t colon = head.find(": "); colon != std::string::npos) why = head.substr(colon + 2), head.resize(colon);
    slog_row(warn ? LogLevel::Warning : LogLevel::Info, "resume", "%s%s", warn ? "WARNING: " : "", head.c_str());
    if (!why.empty()) slog_row(warn ? LogLevel::Warning : LogLevel::Info, "", "%s", why.c_str());
    (void)id;
}

namespace {
// A finished request's detail rows, before its cache saves: MTP and PLE (debug), a dropped tool call.
void log_detail(const GenerationResult &r) {
    if (r.mtp_drafted_tokens > 0) {
        char per_forward[48] = "";
        if (r.decode_forwards > 0)
            std::snprintf(per_forward, sizeof per_forward, ", %.2f tokens per forward",
                          (double)(r.completion_tokens - r.fed_tokens) / (double)r.decode_forwards);
        slog_row(LogLevel::Debug, "mtp", "%s/%s drafts accepted (%.1f%%)%s", fmt_n(r.mtp_accepted_tokens).c_str(),
                 fmt_n(r.mtp_drafted_tokens).c_str(), 100.0 * (double)r.mtp_accepted_tokens / (double)r.mtp_drafted_tokens,
                 per_forward);
        slog_row(LogLevel::Debug, "", "%s rollbacks; drafts accepted before the reject 0/1/2/3+: %s/%s/%s/%s",
                 fmt_n(r.mtp_rollbacks).c_str(), fmt_n(r.mtp_reject_at[0]).c_str(), fmt_n(r.mtp_reject_at[1]).c_str(),
                 fmt_n(r.mtp_reject_at[2]).c_str(), fmt_n(r.mtp_reject_at[3]).c_str());
    }
    const BackendStats &b = r.backend;
    if (b.ple_gathers > 0) {
        // The n-gram (PLE) table's rows: from the RAM row cache, else read from the table file - on demand inside a
        // forward, or ahead of it by a prefetch. A forward that reached the PLE layer before its rows waited.
        const bool slow = b.ple_wait_seconds > 1.0;
        const LogLevel level = slow ? LogLevel::Warning : LogLevel::Debug;
        slog_row(level, "ple", "%s row lookups, %s distinct, %.1f%% from the RAM row cache",
                 fmt_n(b.ngram_rows_requested).c_str(), fmt_n(b.ngram_rows_unique).c_str(),
                 b.ngram_rows_unique > 0 ? 100.0 * (double)b.ngram_rows_cached / (double)b.ngram_rows_unique : 0.0);
        slog_row(level, "", "%s read from the table file on demand, %s prefetched", fmt_n(b.ngram_rows_read).c_str(),
                 fmt_n(b.ngram_rows_prefetched).c_str());
        slog_row(level, "", "%sforwards waited for rows %s times, %s s in total", slow ? "WARNING: " : "",
                 fmt_n(b.ple_waits).c_str(), fmt_rate(b.ple_wait_seconds, 3).c_str());
    }
    if (r.dropped_partial_call)
        slog_row(LogLevel::Warning, "tools", "WARNING: a tool call was still open when the generation ended (%s)",
                 r.finish_reason.c_str());
}

// The end of a request's block, after its cache saves: the "done" row - the times a client sees (the total first, in
// a fixed column) and how it ended - and the closing rule.
void log_footer(const GenerationResult &r) {
    const LogLevel level = r.finish_reason == "error" ? LogLevel::Error
                           : r.finish_reason == "cancelled" ? LogLevel::Warning
                                                            : LogLevel::Info;
    std::string how = r.finish_reason;
    if (r.tool_calls > 0) how += " (" + std::to_string(r.tool_calls) + (r.tool_calls == 1 ? " call)" : " calls)");
    if (!r.error.empty()) how = "ERROR: failed (the error row above says why)";
    char first[48] = "";
    if (r.first_token_ms > 0) std::snprintf(first, sizeof first, ", %s s first token", fmt_rate(r.first_token_ms / 1e3, 2).c_str());
    slog_row(level, "done", "%9s s total%s, %s, queue %s s", fmt_rate(r.total_ms / 1e3, 2).c_str(), first, how.c_str(),
             fmt_rate(r.queue_ms / 1e3, 2).c_str());
    slog(LogLevel::Info, "%s", kLogRule);
}

}  // namespace

void Engine::run(Job &job) {
    GenerationRequest &req = job.req;
    GenerationSink &sink = *job.sink;
    GenerationResult res;
    const double t_start = now_s();
    res.queue_ms = (t_start - job.enqueued) * 1e3;
    res.prompt_tokens = (int64_t)req.prompt.size();
    const BackendStats b0 = be_.backend_stats();
    // The request's log block (serve/log.hpp): the rule, the id and the HTTP layer's rows now; the footer (log_footer)
    // after the saves - whichever way run() returns - then a blank line before the next block.
    // The blank line is one no-break space (U+00A0): journald drops a line that is empty or only ASCII whitespace
    // after the "<N>" prefix.
    slog(LogLevel::Info, "%s", kLogRule);
    slog(LogLevel::Info, "Request %lld", (long long)req.id);
    for (const auto &[label, text] : req.log_header) slog_row(LogLevel::Info, label.c_str(), "%s", text.c_str());
    struct BlockEnd {
        const GenerationResult &r;
        ~BlockEnd() {
            log_footer(r);
            slog(LogLevel::Info, "\xc2\xa0");
        }
    } block_end{res};
    const auto finish = [&](const std::string &reason) {
        res.finish_reason = reason;
        res.total_ms = (now_s() - job.enqueued) * 1e3;
        const BackendStats b = be_.backend_stats();
        res.backend.ngram_rows_requested = b.ngram_rows_requested - b0.ngram_rows_requested;
        res.backend.ngram_rows_unique = b.ngram_rows_unique - b0.ngram_rows_unique;
        res.backend.ngram_rows_cached = b.ngram_rows_cached - b0.ngram_rows_cached;
        res.backend.ngram_rows_read = b.ngram_rows_read - b0.ngram_rows_read;
        res.backend.ngram_rows_prefetched = b.ngram_rows_prefetched - b0.ngram_rows_prefetched;
        res.backend.ple_prefetches = b.ple_prefetches - b0.ple_prefetches;
        res.backend.ple_prefetch_skipped = b.ple_prefetch_skipped - b0.ple_prefetch_skipped;
        res.backend.ple_prefetch_seconds = b.ple_prefetch_seconds - b0.ple_prefetch_seconds;
        res.backend.ple_gathers = b.ple_gathers - b0.ple_gathers, res.backend.ple_waits = b.ple_waits - b0.ple_waits;
        res.backend.ple_gather_seconds = b.ple_gather_seconds - b0.ple_gather_seconds;
        res.backend.ple_wait_seconds = b.ple_wait_seconds - b0.ple_wait_seconds;
        res.backend.ple_wait_max_seconds = b.ple_wait_max_seconds;
        {
            std::lock_guard<std::mutex> lock(mu_);
            (reason == "error" ? stats_.requests_error : reason == "cancelled" ? stats_.requests_cancelled : stats_.requests_ok)++;
            stats_.generated_tokens += res.completion_tokens;
            stats_.mtp_drafted_tokens += res.mtp_drafted_tokens;
            stats_.think_nudges += res.think_nudges;
            stats_.mtp_accepted_tokens += res.mtp_accepted_tokens;
            stats_.mtp_rollbacks += res.mtp_rollbacks;
            for (int64_t i = 0; i < kMtpMaxDraft; ++i) stats_.mtp_reject_at[i] += res.mtp_reject_at[i];
            stats_.decode_seconds += res.decode_ms / 1e3;
            if (res.decode_ms > 0 && res.completion_tokens > 1)
                stats_.last_decode_tps = (double)(res.completion_tokens - 1) / (res.decode_ms / 1e3);
        }
        log_detail(res);
        sink.on_done(res);
    };
    try {
        sink.on_start();
        const std::vector<int32_t> &prompt = req.prompt;
        const int64_t P = (int64_t)prompt.size();
        bool from_disk = false, from_ram = false;
        int64_t live_common = 0;  // before resume_point moves the live session
        while (live_common < (int64_t)seq_.size() && live_common < P && seq_[(size_t)live_common] == prompt[(size_t)live_common])
            ++live_common;
        const int64_t live_n = (int64_t)seq_.size();
        const int64_t start = resume_point(prompt, req.id, from_disk, from_ram);
        report_resume_loss(prompt, start, live_common, live_n, req.id);
        res.cached_tokens = start;
        // Snapshot points: the prompt's last <|im_start|> (where the generation prompt begins; the next turn resumes
        // there), its second-to-last <|im_start|> (the start of the last user message - the user-turn
        // checkpoint: a client rewriting that message, e.g. dropping a reminder it appended, still matches everything
        // before it), and - for the disk cache - the second <|im_start|> overall (the end of the system
        // prompt, shared by every conversation that has it), unless the disk already holds that prefix.
        int64_t cut = -1, ustart = -1, sys = -1;
        for (int64_t k = P - 1; k > 0; --k)
            if (prompt[(size_t)k] == im_start_) {
                if (cut < 0) cut = k;
                else if (ustart < 0) ustart = k;
                sys = k;
            }
        // A prompt without <|im_start|> (a raw /v1/completions prompt): the turn point is the prompt's end, so a
        // next request that extends it - a benchmark priming a context, then adding to it - resumes there instead of
        // prefilling the primed part again. Chat prompts always have one; nothing changes for them.
        if (cut < 0) cut = P;
        const bool save_sys = cache_ && sys > start && sys < cut && !cache_->contains(prompt.data(), sys);
        // The user-turn checkpoint: only for user turns, only past the system prefix (the first user message's
        // start is the system entry) and not behind the resume point (then it was saved by an earlier request).
        const int64_t ust = cache_ && req.user_turn && ustart > sys && ustart >= start ? ustart : -1;
        // Logits rows as the sampler takes them: for greedy / top_k 1..20 (the served defaults) each row's top 20
        // candidates, reduced where the logits are (kernels/logits_topk on the real model) - ~160 bytes a row to the
        // host instead of ~1 MB and a full-row scan; same tokens either way.
        const bool cands = Sampler(req.sampling, tok_.size()).takes_candidates();
        const int64_t n_valid = tok_.size();
        LogitRows logits;
        // Structured output (response_format / tool_choice): every logits row from the prompt's last one on is masked
        // to what the grammar allows there (serve/structured_output.hpp); `so` follows each generated or fed token.
        std::unique_ptr<StructuredOutput> so;
        if (req.grammar)
            so = std::make_unique<StructuredOutput>(req.grammar, masker_, tok_, think_end_, std::vector<int32_t>{im_end_, eot_},
                                                    !req.parser.thinking);
        LogitMasks row_masks;  // the masks of the forward being made (kept alive across the call)
        const auto masks_for = [&](const std::vector<int32_t> &pending) -> const LogitMasks * {
            if (!so) return nullptr;
            row_masks = so->masks_for(pending);
            return &row_masks;
        };
        // A chunk ends at the backend's max chunk, and at the turn / user-start / system snapshot points.
        auto chunk_end = [&](int64_t pos) {
            int64_t end = std::min(P, pos + be_.max_chunk());
            if (cut > pos && cut < end) end = cut;
            if (ust > pos && ust < end) end = ust;
            if (save_sys && sys > pos && sys < end) end = sys;
            return end;
        };
        if (start == ust)  // resumed exactly at the user turn: its start state is the live state right now
            be_.save_snapshot(LmBackend::kUserSlot);
        for (int64_t pos = start; pos < P;) {
            const int64_t end = chunk_end(pos);
            if (end < P)
                be_.set_lookahead(std::vector<int32_t>(prompt.begin() + end, prompt.begin() + chunk_end(end)));
            logits = be_.forward_rows(std::vector<int32_t>(prompt.begin() + pos, prompt.begin() + end), end == P ? 1 : 0,
                                      cands, n_valid, end == P ? masks_for({}) : nullptr);
            seq_.insert(seq_.end(), prompt.begin() + pos, prompt.begin() + end);
            pos = end;
            if (pos == cut) be_.save_snapshot(LmBackend::kTurnSlot);
            if (pos == ust) be_.save_snapshot(LmBackend::kUserSlot);
            if (save_sys && pos == sys) be_.save_snapshot(LmBackend::kSystemSlot);
            sink.on_progress(pos, P);
            if (pos < P && sink.cancelled()) {
                res.prompt_ms = (now_s() - t_start) * 1e3;
                return finish("cancelled");
            }
        }
        const double t_first = now_s();
        res.prompt_ms = (t_first - t_start) * 1e3;
        res.first_token_ms = (t_first - job.enqueued) * 1e3;
        // The rate leads the row, right-aligned in a fixed column, so the numbers a block is read for (this and
        // "generate") stack up vertically across requests. Only for a real prefill: below ~1k tokens the time is
        // mostly fixed cost and the rate misleads - then "-" stands in the column.
        char rate[48];
        if (P - start >= 1024 && res.prompt_ms > 0)
            std::snprintf(rate, sizeof rate, "%9s t/s", fmt_rate((double)(P - start) / (res.prompt_ms / 1e3), 0).c_str());
        else
            std::snprintf(rate, sizeof rate, "%9s t/s", "-");
        slog_row(LogLevel::Info, "prefill", "%s   %s new tokens in %s s", rate, fmt_n(P - start).c_str(),
                 fmt_rate(res.prompt_ms / 1e3, 2).c_str());
        {
            std::lock_guard<std::mutex> lock(mu_);
            stats_.prompt_tokens += P, stats_.cached_tokens += start, stats_.prefill_tokens += P - start;
            (start > 0 ? stats_.cache_hits : stats_.cache_misses)++;
            if (from_disk) ++stats_.disk_hits, stats_.disk_tokens += start, stats_.ram_hits += from_ram;
            stats_.prefill_seconds += res.prompt_ms / 1e3;
            if (P > start) stats_.last_prefill_tps = (double)(P - start) / (res.prompt_ms / 1e3);
        }

        Sampler sampler(req.sampling, tok_.size());
        OutputParser parser(tok_, req.parser);
        ThinkWatch watch{options_.think_nudge_policy};
        // A sampled token just fed to the parser: the nudge watch follows the reasoning ones.
        const auto watch_tok = [&](int32_t t) {
            if (options_.think_nudge && parser.in_reasoning()) watch.observe(t, tok_.decode({t}));
        };
        std::vector<OutputEvent> events;
        std::string reason;
        std::vector<CaptureStep> steps;  // what each decode step did, for --capture-dir
        auto step = [&](int64_t drafted, int64_t accepted, int64_t emitted) {
            if (!options_.capture_dir.empty())
                steps.push_back({(uint8_t)drafted, (uint8_t)accepted, (uint8_t)emitted});
        };
        // A token sampled at a rejected draft's position - already counted, fed to the parser and the response format,
        // but not yet forwarded: the next step starts from it (drafts from it, then verifies it as row 0) instead of a
        // forward of its own (strixite PR #1).
        int32_t carry = -1;
        for (;;) {
            if (sink.cancelled()) {
                reason = "cancelled";
                break;
            }
            // An injection due (the budget's stop or a nudge, below) while a token is carried: forward the carried token
            // alone first - the old rollback path, for this step only - so the injection lands after it. Waiting for a
            // step without a carry instead could wait for ever: with every draft rejected, every step carries one.
            if (carry >= 0) {
                const bool budget_due = req.thinking_budget >= 0 && !res.thinking_budget_hit && parser.in_reasoning() &&
                                        parser.reasoning_tokens() >= req.thinking_budget;
                const bool nudge_due = options_.think_nudge && parser.in_reasoning() && watch.due() != 0;
                if (budget_due || nudge_due) {
                    ++res.decode_forwards;
                    logits = be_.forward_rows({carry}, 1, cands, n_valid, masks_for({}));
                    seq_.push_back(carry);
                    step(0, 0, 1);
                    carry = -1;
                }
            }
            // Thinking budget spent: the engine closes the think block itself (fed, not sampled), then goes on sampling. Never
            // while a carried token waits (flushed just above when due): the injection would be forwarded ahead of it.
            if (carry < 0 && req.thinking_budget >= 0 && !res.thinking_budget_hit && parser.in_reasoning() &&
                parser.reasoning_tokens() >= req.thinking_budget) {
                const int64_t n = (int64_t)thinking_stop_.size();
                if (res.completion_tokens + n >= req.max_tokens) {  // no room left for it and an answer
                    reason = "length";
                    break;
                }
                res.thinking_budget_hit = true;
                res.fed_tokens += n;
                for (int32_t id : thinking_stop_) {
                    parser.feed(id, events);
                    if (so) so->feed(id);
                }
                if (!events.empty()) sink.on_events(events), events.clear();
                res.completion_tokens += n;
                logits = be_.forward_rows(thinking_stop_, 1, cands, n_valid, masks_for({}));
                seq_.insert(seq_.end(), thinking_stop_.begin(), thinking_stop_.end());
                step(kThinkingStopStep, 0, n);
                continue;
            }
            // Thinking going in circles: feed a nudge into the think block (never a </think>), then go on sampling.
            // Not while a carried token waits (as above).
            if (carry < 0 && options_.think_nudge && parser.in_reasoning()) {
                if (const int d = watch.due()) {
                    const std::vector<int32_t> &nt = nudge_[d - 1];
                    const int64_t n = (int64_t)nt.size();
                    if (res.completion_tokens + n < req.max_tokens) {
                        slog_row(LogLevel::Info, "nudge", "#%d fed at %s thinking tokens: going in circles", d,
                                 fmt_n(watch.thinking_tokens()).c_str());
                        slog_row(LogLevel::Info, "", "%.0f%% of the last %s thinking tokens repeat earlier ones",
                                 100.0 * watch.window_rate(), fmt_n((long long)options_.think_nudge_policy.window).c_str());
                        for (int32_t t : nt) {
                            parser.feed(t, events);
                            if (so) so->feed(t);
                        }
                        if (!events.empty()) sink.on_events(events), events.clear();
                        res.completion_tokens += n;
                        ++res.think_nudges;
                        res.fed_tokens += n;
                        logits = be_.forward_rows(nt, 1, cands, n_valid, masks_for({}));
                        seq_.insert(seq_.end(), nt.begin(), nt.end());
                        step(kThinkingStopStep, 0, n);
                        watch.fired();
                        continue;
                    }
                }
            }
            int32_t id;
            if (carry >= 0) {  // sampled, counted and fed at the rejected position: straight to drafting
                id = carry;
                carry = -1;
            } else {
                id = sampler.sample(logits, 0);
                ++res.completion_tokens;
                if (id == im_end_ || id == eot_) {
                    reason = "stop";
                    break;
                }
                if (so) so->feed(id);
                const bool stopped = parser.feed(id, events);
                watch_tok(id);
                if (!events.empty()) sink.on_events(events), events.clear();
                if (stopped) {
                    reason = "stop";
                    break;
                }
                if (res.completion_tokens >= req.max_tokens) {
                    reason = "length";
                    break;
                }
            }

            // MTP: up to mtp_draft greedy drafts from the head (chained), each while its top-1 margin reaches
            // mtp_margin (and no end-of-turn); one verify forward of id + the drafts. Each position's token is still
            // sampled from the trunk's own logits (row j) - a draft only saves the forward when it matches.
            if (be_.has_mtp()) {
                const int64_t room = req.max_tokens - res.completion_tokens;  // >= 1 here
                const int64_t steps = std::min(options_.mtp_draft, room);
                std::vector<int32_t> ids{id};
                // Each token's PLE rows start loading as soon as it is known, while the next draft runs - so the
                // verify (or id's own forward, when no draft passes) finds them in the row cache.
                be_.prefetch_ple(ids, 0);
                StructuredOutput::Cursor draft_at = so ? so->cursor() : StructuredOutput::Cursor{};
                for (int64_t step = 0; step < steps; ++step) {
                    const Top2 t = be_.forward_mtp_top2(ids.back(), step);
                    STRIX_CHECK(!t.nan, "Engine: NaN in the MTP draft logits (step ", step, ")");
                    const int32_t d = t.best;
                    if (t.best_v - t.second_v < options_.mtp_margin || d == im_end_ || d == eot_) break;
                    // A draft the response format refuses would be rejected anyway: don't verify it.
                    if (so) {
                        if (!so->allows(draft_at, d)) break;
                        draft_at = so->after(draft_at, d);
                    }
                    ids.push_back(d);
                    if (step + 1 < steps) be_.prefetch_ple(ids, (int64_t)ids.size() - 1);  // the last: nothing to hide behind
                }
                const int64_t k = (int64_t)ids.size() - 1;
                if (k > 0) {
                    res.mtp_drafted_tokens += k;
                    ++res.decode_forwards;
                    const LogitRows ml = be_.forward_verify_rows(
                        ids, k + 1, cands, n_valid, masks_for(std::vector<int32_t>(ids.begin() + 1, ids.end())));
                    STRIX_CHECK(ml.rows == k + 1, "Engine: verify of ", k + 1, " tokens returned ", ml.rows,
                                " logits rows");
                    // Accept drafts while the sampled token is the draft.
                    int64_t j = 0;
                    int32_t v = -1;
                    bool ended = false;
                    for (; j < k; ++j) {
                        v = sampler.sample(ml, j);
                        if (v != ids[(size_t)j + 1]) break;
                        ++res.mtp_accepted_tokens, ++res.completion_tokens;
                        if (so) so->feed(v);
                        const bool st = parser.feed(v, events);
                        watch_tok(v);
                        if (!events.empty()) sink.on_events(events), events.clear();
                        if (st || res.completion_tokens >= req.max_tokens) {
                            reason = st ? "stop" : "length";
                            ended = true;
                            ++j;
                            break;
                        }
                    }
                    if (ended) {  // keep id + the j accepted drafts in the session, drop the rest
                        if (j < k) be_.keep_verify_prefix(ids, j + 1);
                        else be_.keep_verify();
                        seq_.insert(seq_.end(), ids.begin(), ids.begin() + j + 1);
                        step(k, j, j + 1);
                        break;
                    }
                    if (j == k) {  // all accepted: the last row is the next token's logits
                        be_.keep_verify();
                        seq_.insert(seq_.end(), ids.begin(), ids.end());
                        step(k, k, k + 1);
                        logits = ml.row_of(k);
                        continue;
                    }
                    // Draft j rejected; v is this position's sampled token. Keep id and the j accepted drafts from the
                    // verify (the backend doesn't run them again - keep_verify_prefix). The session is then "after
                    // ids[j]" - what a step's drafting starts from - so v carries into the next step instead of a
                    // forward of its own (strixite PR #1): one forward less per rollback.
                    ++res.mtp_rollbacks, ++res.completion_tokens, ++res.mtp_reject_at[std::min<int64_t>(j, kMtpMaxDraft - 1)];
                    {  // v's PLE rows load while the prefix is kept (its forward - the next verify's row 0 - comes soon)
                        std::vector<int32_t> next(ids.begin(), ids.begin() + j + 1);
                        next.push_back(v);
                        be_.prefetch_ple(next, j + 1);
                    }
                    be_.keep_verify_prefix(ids, j + 1);
                    seq_.insert(seq_.end(), ids.begin(), ids.begin() + j + 1);
                    if (v == im_end_ || v == eot_) {
                        step(k, j, j + 1);
                        reason = "stop";
                        break;
                    }
                    if (so) so->feed(v);
                    const bool st = parser.feed(v, events);
                    watch_tok(v);
                    if (!events.empty()) sink.on_events(events), events.clear();
                    step(k, j, j + 1);  // v is emitted by the next step
                    if (st || res.completion_tokens >= req.max_tokens) {
                        reason = st ? "stop" : "length";
                        break;
                    }
                    carry = v;
                    continue;
                }
            }
            ++res.decode_forwards;
            logits = be_.forward_rows({id}, 1, cands, n_valid, masks_for({}));
            seq_.push_back(id);
            step(0, 0, 1);
        }
        parser.finish(events);
        if (!events.empty()) sink.on_events(events);
        res.reasoning_tokens = parser.reasoning_tokens();
        res.dropped_partial_call = parser.dropped_partial_call();
        res.malformed_tool_calls = parser.malformed_calls();
        res.decode_ms = (now_s() - t_first) * 1e3;
        res.tool_calls = parser.tool_calls();
        // A streamed call that went out malformed is still a call the client received.
        if (reason == "stop" && parser.tool_calls() + parser.malformed_calls() > 0) reason = "tool_calls";
        char gen_rate[48];  // the rate leads the row, in the same column as the prefill one (serve/log.hpp)
        std::snprintf(gen_rate, sizeof gen_rate, "%9s t/s",
                      fmt_rate(res.decode_ms > 0 && res.completion_tokens > 1 ? (res.completion_tokens - 1) / (res.decode_ms / 1e3) : 0.0,
                               1)
                          .c_str());
        slog_row(LogLevel::Info, "generate", "%s   %s tokens (think %s) in %s s", gen_rate,
                 fmt_n(res.completion_tokens).c_str(), fmt_n(res.reasoning_tokens).c_str(),
                 fmt_rate(res.decode_ms / 1e3, 2).c_str());
        finish(reason);
        if (!options_.capture_dir.empty() && reason != "cancelled") capture(req, steps, reason);
        // After the response: the new prefixes go to the prompt cache's RAM tier (the disk only later, if at all).
        if (cache_ && reason != "cancelled") {
            const double t0 = now_s();
            int saved = 0, rows = 0;  // rows: the first gets the "cache" label
            const auto save = [&](int slot, int64_t n, PromptCache::Kind kind, const char *what) {
                // Already cached first: a request resumed exactly at its turn entry has no snapshot there, and needs none.
                const char *skip = n <= 0                                ? "no such point in the prompt"
                                   : cache_->contains(prompt.data(), n) ? nullptr
                                   : be_.snapshot_pos(slot) != n         ? "no snapshot there"
                                   : !be_.can_restore_snapshot(slot)     ? "snapshot no longer valid"
                                                                         : "";
                if (skip == nullptr) return;  // already cached
                if (*skip) {
                    slog_row(LogLevel::Debug, rows++ ? "" : "cache", "%s, %s tokens: not saved - %s (snapshot at %s)",
                             what, fmt_n(n).c_str(), skip, fmt_n(be_.snapshot_pos(slot)).c_str());
                    return;
                }
                const double t_e = now_s();
                // The replaced turn's buffer first (grown by this turn's few MB), else a spare: a fresh multi-GB buffer
                // costs ~100 ms per GB of page population on the request path.
                // A Turn of a deep conversation goes as a delta on the whole entry it extends (PromptCache "Delta entries").
                const int64_t base_n = cache_->delta_base(prompt.data(), n, kind);
                HostBuffer state = kind == PromptCache::Kind::System ? HostBuffer()
                                                                      : cache_->take_replaced_buffer(prompt.data(), n, base_n);
                if (state.capacity() == 0) state = cache_->take_buffer();
                be_.export_snapshot(slot, state, base_n);
                const double ms = (now_s() - t_e) * 1e3;  // > 0.5 s: the request path waited - worth seeing
                const LogLevel level = ms > 500 ? LogLevel::Warning : LogLevel::Debug;
                // To the RAM tier only: the disk gets an entry later (idle, RAM pressure, shutdown), if at all.
                slog_row(level, rows++ ? "" : "cache", "%s, %s tokens -> RAM%s", what, fmt_n(n).c_str(),
                         req.one_shot ? " only (a one-shot request: never written to disk)" : "");
                const std::string shape = base_n > 0 ? "+" + fmt_n(n - base_n) + " tokens on the " +
                                                            fmt_n(base_n) + "-token entry"
                                                     : std::string("whole");
                slog_row(level, "", "%s%s, %s MB, export %s ms", ms > 500 ? "WARNING: slow export - " : "", shape.c_str(),
                         fmt_rate(state.size() / 1e6, 0).c_str(), fmt_rate(ms, 0).c_str());
                rows++;
                if (!cache_->put(std::vector<int32_t>(prompt.begin(), prompt.begin() + n), kind, std::move(state),
                                 req.one_shot, base_n, req.id))
                    return;
                ++saved;
            };
            if (save_sys) save(LmBackend::kSystemSlot, sys, PromptCache::Kind::System, "the system prompt's end");
            // The checkpoint is the start of the user message, not its end: clients rewrite the latest user message
            // on the next one (opencode's plan mode appends a reminder to it and strips it later), so an entry that
            // includes it never matches again. The turn itself is a plain Turn, replaced by the tool loop's next.
            if (ust >= 0) save(LmBackend::kUserSlot, ust, PromptCache::Kind::Checkpoint, "the last user message's start");
            save(LmBackend::kTurnSlot, cut, PromptCache::Kind::Turn, "the reply's start");
            std::lock_guard<std::mutex> lock(mu_);
            stats_.ram_saves += saved, stats_.export_seconds += now_s() - t0;
        }
    } catch (const std::exception &e) {
        // A failed forward leaves the session unusable until reset: start clean for the next request.
        slog_row(LogLevel::Error, "error", "ERROR: %s", e.what());
        try {
            be_.reset();
        } catch (const std::exception &e2) {
            slog_row(LogLevel::Error, "", "ERROR: the reset after it failed too: %s", e2.what());
        }
        seq_.clear();
        res.error = e.what();
        finish("error");
    }
}

}  // namespace strix
