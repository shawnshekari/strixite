#include "serve/engine.hpp"
#include "serve/seq_backend.hpp"
#include "serve/think_nudge.hpp"

#include "serve/capture.hpp"
#include "serve/log.hpp"
#include "serve/resume_loss.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <cstdio>

namespace strix {

namespace {
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

bool ends_sentence(std::string_view tail) {
    if (tail.empty()) return false;
    if (tail.back() == '\n') return true;
    const size_t line = tail.rfind('\n');
    const std::string_view cur = line == std::string_view::npos ? tail : tail.substr(line + 1);
    if (std::count(cur.begin(), cur.end(), '`') % 2 != 0) return false;  // inside inline code
    size_t n = cur.size();
    while (n > 0 && std::string_view(")]\"'*").find(cur[n - 1]) != std::string_view::npos) --n;  // closing marks
    if (n < 2 || std::string_view(".!?").find(cur[n - 1]) == std::string_view::npos) return false;
    const unsigned char before = (unsigned char)cur[n - 2];
    return std::isalpha(before) || before == ')' || before == '`' || before == '"' || before == '\'' || before >= 0x80;
}

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
      nudge_{tok.encode(think_nudge1_text(options.think_nudge_wording)), tok.encode(kThinkNudge2)},
      think_close_(tok.encode(kThinkCloseBeforeCall)),
      call_begin_(tok.id_of("<tool_call>")) {
    STRIX_CHECK(backend.logits_row() >= tok.size(), "Engine: logits rows of ", backend.logits_row(),
                " are shorter than the tokenizer's ", tok.size(), " tokens");
    STRIX_CHECK(options.shared_chunk >= 0, "Engine: shared_chunk ", options.shared_chunk, ", expected >= 0 (0 = off)");
    STRIX_CHECK(backend.max_chunk() >= (int64_t)thinking_stop_.size() && backend.capacity() >= 2, "Engine: backend chunk ",
                backend.max_chunk(), " (the thinking stop is ", thinking_stop_.size(), " tokens), capacity ", backend.capacity());
    STRIX_CHECK(std::count(thinking_stop_.begin(), thinking_stop_.end(), tok.id_of("</think>")) == 1,
                "Engine: the thinking stop text doesn't tokenize to exactly one </think>");
    for (const auto &n : nudge_)
        STRIX_CHECK(!n.empty() && (int64_t)n.size() <= backend.max_chunk() &&
                        std::count(n.begin(), n.end(), tok.id_of("</think>")) == 0,
                    "Engine: a thinking nudge tokenizes to ", n.size(), " tokens (backend chunk ", backend.max_chunk(),
                    ") or contains </think> - it must leave the think block open");
    STRIX_CHECK(!think_close_.empty() && think_close_.front() == think_end_ &&
                    std::count(think_close_.begin(), think_close_.end(), think_end_) == 1 &&
                    (int64_t)think_close_.size() + 1 <= backend.max_chunk(),
                "Engine: kThinkCloseBeforeCall tokenizes to ", think_close_.size(),
                " tokens, expected </think> first and only once, and room for it + <tool_call> in a backend chunk of ",
                backend.max_chunk());
    STRIX_CHECK(options_.mtp_margin >= 0.0f, "Engine: mtp_margin ", options_.mtp_margin, ", expected >= 0");
    STRIX_CHECK(options_.mtp_draft >= 1 && options_.mtp_draft <= 15, "Engine: mtp_draft ", options_.mtp_draft,
                ", expected 1..15");
    STRIX_CHECK(options_.capture_dir.empty() || std::filesystem::is_directory(options_.capture_dir),
                "Engine: capture_dir '", options_.capture_dir, "' is not a directory");
    (void)ThinkWatch{options_.think_nudge_policy};  // a bad nudge policy fails here, at startup, not on a request
    be_.reset();
    idle_since_ = now_s();
    const int n = be_.n_seqs();
    STRIX_CHECK(n >= 1 && n <= 8, "Engine: the backend holds ", n, " conversations, expected 1..8");
    if (n > 1) arbiter_ = std::make_unique<GpuArbiter>(be_);
    for (int k = 0; k < n; ++k) {
        auto sl = std::make_unique<Slot>();
        sl->index = k;
        if (n > 1) sl->proxy = std::make_unique<SeqBackend>(*arbiter_, k), sl->be = sl->proxy.get();
        else sl->be = &be_;  // one slot: the backend itself, exactly as before
        slots_.push_back(std::move(sl));
    }
    stats_.slots = n, stats_.slot_request.assign((size_t)n, 0), stats_.slot_live_tokens.assign((size_t)n, 0);
    stats_.slot_snapshot_pos.assign((size_t)n, -1);
    for (auto &sl : slots_) sl->thread = std::thread([this, s = sl.get()] { worker(*s); });
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
    for (auto &sl : slots_)
        if (sl->thread.joinable()) sl->thread.join();
    for (Job &j : queue_) {
        GenerationResult r;
        r.finish_reason = "error", r.error = "server shutting down";
        deliver(j, r);  // never through a destructor, whatever the sink does
    }
}

void Engine::submit(GenerationRequest req, std::shared_ptr<GenerationSink> sink) {
    STRIX_CHECK(sink != nullptr, "Engine::submit: null sink");
    STRIX_CHECK(!req.prompt.empty(), "Engine::submit: empty prompt");
    STRIX_CHECK(req.thinking_budget >= -1, "Engine::submit: thinking budget ", req.thinking_budget, " (-1 = none)");
    STRIX_CHECK(req.thinking_budget_grace >= 0 && (req.thinking_budget_grace == 0 || req.thinking_budget >= 0),
                "Engine::submit: thinking budget grace ", req.thinking_budget_grace, " (expected >= 0, and 0 without a budget: ",
                req.thinking_budget, ")");
    STRIX_CHECK(req.max_tokens >= 1 && (int64_t)req.prompt.size() + req.max_tokens <= capacity_,
                "Engine::submit: prompt ", req.prompt.size(), " + max_tokens ", req.max_tokens, " exceed the capacity ",
                capacity_, " (the API layer clamps)");
    for (int32_t id : req.prompt)
        STRIX_CHECK(id >= 0 && id < tok_.size(), "Engine::submit: prompt token ", id, " outside [0, ", tok_.size(), ")");
    {
        std::lock_guard<std::mutex> lock(mu_);
        STRIX_CHECK(!stop_, "Engine::submit: shutting down");
        STRIX_CHECK(fatal_.empty(), "Engine::submit: the GPU failed on an earlier request (", fatal_,
                    "); nothing runs until the server restarts");
        // One line from the HTTP thread for a request that waits; its block prints when it starts.
        if (stats_.busy >= stats_.slots || !queue_.empty()) {
            std::string running;
            for (const auto &sl : slots_)
                if (sl->running_id) running += (running.empty() ? "" : ", ") + std::to_string(sl->running_id);
            slog(LogLevel::Info, "Request %lld queued: %s ahead of it (Request%s %s running)", (long long)req.id,
                 fmt_n((long long)queue_.size() + stats_.busy).c_str(), stats_.busy > 1 ? "s" : "", running.c_str());
        }
        queue_.push_back({std::move(req), std::move(sink), now_s()});
        ++stats_.queued;
        dispatch_locked();
    }
    cv_.notify_all();
}

void Engine::dispatch_locked() {
    while (!queue_.empty() && fatal_.empty()) {
        const GenerationRequest &req = queue_.front().req;
        const std::vector<int32_t> &key = req.image_pos3.empty() ? req.prompt : req.cache_key;
        // The slot, among the idle ones (an image prompt: slot 0 only, image positions are slot 0's): the one whose
        // state the prompt continues (its tokens a prefix of the prompt - the same conversation, longest first), else
        // an empty one, else the least recently used (its conversation goes back to the prompt cache). A shared
        // prefix alone (a system prompt and tool list every conversation of a client has) doesn't pick a slot: it
        // would evict another live conversation while a slot sits empty.
        // A prompt that continues a busy slot's running request (its whole prompt a prefix: the client's next turn,
        // sent as soon as the reply streamed while the slot still saves its cache) waits for that slot - on another
        // slot it would re-prefill from the prompt cache at best.
        if (slots_.size() > 1) {
            const Slot *mine = nullptr;
            for (const auto &sl : slots_) {
                const std::vector<int32_t> &rk = sl->running_key;
                if (sl->busy && !rk.empty() && rk.size() < key.size() && std::equal(rk.begin(), rk.end(), key.begin()))
                    mine = sl.get();
            }
            if (mine != nullptr) {
                if (held_logged_ != req.id) {
                    held_logged_ = req.id;
                    slog(LogLevel::Info, "Request %lld waits for slot %d: it continues Request %lld's conversation "
                         "(%s tokens), still running there", (long long)req.id, mine->index, (long long)mine->running_id,
                         fmt_n((long long)mine->running_key.size()).c_str());
                }
                return;  // FIFO: the requests behind it wait too
            }
        }
        Slot *cont = nullptr, *empty = nullptr, *lru = nullptr;
        int64_t cont_common = 0;
        for (auto &sl : slots_) {
            if (sl->busy) continue;
            if (!req.image_pos3.empty() && sl->index != 0) continue;
            const size_t n = sl->seq.size();  // an idle slot's tokens don't change (its worker waits for a job)
            if (n == 0) {
                if (!empty) empty = sl.get();
                continue;
            }
            // The same conversation: the prompt continues the state, or reaches its last turn point (the client
            // re-rendered the reply: the engine resumes from that snapshot).
            int64_t c = 0;
            while (c < (int64_t)n && c < (int64_t)key.size() && sl->seq[(size_t)c] == key[(size_t)c]) ++c;
            const bool same = (c == (int64_t)n && n < key.size()) || (sl->turn_pos > 0 && c >= sl->turn_pos);
            if (same && (!cont || c > cont_common)) cont = sl.get(), cont_common = c;
            if (!lru || sl->last_used < lru->last_used) lru = sl.get();
        }
        Slot *best = cont ? cont : empty ? empty : lru;
        if (best == nullptr) return;  // FIFO: the oldest request waits for a slot, the ones behind it too
        std::string why;
        if (slots_.size() > 1) {
            std::string idle;  // the other idle slots and what they hold
            for (const auto &sl : slots_)
                if (sl.get() != best && !sl->busy)
                    idle += (idle.empty() ? "" : ", ") + std::to_string(sl->index) + " (" + fmt_n((long long)sl->seq.size()) + ")";
            why = best == cont   ? "continues its live conversation (" + fmt_n(cont_common) + " of " + fmt_n((long long)best->seq.size()) +
                                       " tokens match)"
                  : best == empty ? std::string("an empty slot")
                                  : "least recently used: its " + fmt_n((long long)best->seq.size()) +
                                        "-token conversation goes back to the prompt cache";
            why += idle.empty() ? "; no other slot idle" : "; idle too: " + idle + " tokens";
        }
        best->running_key = key;  // before the move: key refers into the queued request
        best->job = std::make_unique<Job>(std::move(queue_.front()));
        queue_.pop_front();
        best->job->slot_why = std::move(why);
        best->busy = true, best->running_id = best->job->req.id, best->last_used = now_s();
        --stats_.queued, ++stats_.busy;
        stats_.slot_request[(size_t)best->index] = best->running_id;
    }
}

EngineStats Engine::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    EngineStats s = stats_;
    s.idle_seconds = s.busy || s.queued ? 0.0 : std::max(0.0, now_s() - idle_since_);
    if (arbiter_) {
        const GpuArbiter::Stats a = arbiter_->stats();
        s.verify_batches = a.verify_batches, s.verify_batch_runs = a.verify_runs;
        s.verify_solo_mrope = a.verify_solo_mrope;
        s.decoding_slots = arbiter_->decoding_count();
    }
    return s;
}

void Engine::deliver(Job &job, const GenerationResult &r) {
    // result_sent first: a sink that throws from on_done loses that response and must not get a second
    // delivery attempt (GenerationSink's contract, serve/engine.hpp).
    job.result_sent = true;
    try {
        job.sink->on_done(r);
    } catch (const std::exception &ex) {
        slog(LogLevel::Error, "a request's sink threw from on_done: %s - the client's response is lost "
                              "(sinks must not throw)",
             ex.what());
    } catch (...) {
        slog(LogLevel::Error, "a request's sink threw a non-standard error from on_done - the client's response "
                              "is lost (sinks must not throw)");
    }
}

void Engine::worker(Slot &sl) {
    if (slots_.size() > 1) log_slot_tag() = sl.index;  // this thread's log lines carry the slot
    for (;;) {
        std::unique_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&] { return stop_ || sl.job != nullptr; });
            if (sl.job == nullptr) return;  // stopping: an assigned job still runs
            job = std::move(sl.job);
        }
        bool released = false;  // the slot's busy bookkeeping ran (normal path, fatal path, or the guard)
        // The per-job bookkeeping under mu_ (the slot goes idle, stats move, the queue gets its chance).
        const auto release_slot = [&](int64_t snap) {
            if (released) return;
            released = true;
            std::lock_guard<std::mutex> lock(mu_);
            sl.busy = false, sl.running_id = 0, sl.running_key.clear(), --stats_.busy;
            idle_since_ = now_s();
            sl.turn_pos = snap;
            stats_.slot_request[(size_t)sl.index] = 0,
                stats_.slot_live_tokens[(size_t)sl.index] = (int64_t)sl.seq.size();
            stats_.slot_snapshot_pos[(size_t)sl.index] = snap;
            stats_.live_tokens = 0;
            for (int64_t v : stats_.slot_live_tokens) stats_.live_tokens += v;
            dispatch_locked();
        };
        // The last line: an exception must never escape a slot's worker. The
        // production abort: one slot's by-design K/V growth refusal poisoned the arbiter, and another slot's
        // post-run advisory snapshot_pos threw into this loop uncaught -> std::terminate -> the whole server
        // died with every live conversation. Advisory calls are now non-throwing (GpuArbiter::run, Kind::Hint),
        // so escaping here needs a genuinely unexpected error - and still must not take the process down:
        // fail the request exactly once, mark the slot for reset, keep serving the rest.
        const auto escape = [&](const std::string &what) {
            slog_row(LogLevel::Error, "error", "the worker caught an error outside the request path: %s - the "
                                                "request is failed, the slot resets before its next job",
                     what.c_str());
            if (job && !job->result_sent) {
                GenerationResult r;
                r.total_ms = (now_s() - job->enqueued) * 1e3;
                r.finish_reason = "error", r.error = "engine: unexpected error while this request ran: " + what;
                deliver(*job, r);
                std::lock_guard<std::mutex> lock(mu_);
                ++stats_.requests_error;
            }
            sl.seq.clear();
            sl.needs_reset = true;  // its state is suspect: reset before anything runs on it again
            release_slot(-1);        // no trustworthy snapshot position after an escape
            cv_.notify_all();
        };
        try {
            if (sl.needs_reset || (sl.proxy && sl.proxy->stale())) {  // another slot's call failed midway while this
                                                                       // one sat idle, or the guard escaped above
                sl.needs_reset = false;
                try {
                    sl.be->reset();
                } catch (const std::exception &ex) {
                    slog(LogLevel::Error, "slot %d: the reset after another slot's failure failed: %s", sl.index,
                         ex.what());
                }
                sl.seq.clear();
            }
            run(*job, sl);
            if (!sl.fatal_pending.empty()) {
                std::deque<Job> failed;
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    fatal_ = sl.fatal_pending;
                    failed.swap(queue_);
                    stats_.queued = 0, --stats_.busy;
                    sl.busy = false, sl.running_id = 0, sl.running_key.clear(), stats_.slot_request[(size_t)sl.index] = 0;
                }
                released = true;  // that was this slot's idle bookkeeping; the guard must not redo it
                for (Job &j : failed) {
                    GenerationResult r;
                    r.finish_reason = "error", r.error = "the GPU failed on an earlier request (" + sl.fatal_pending +
                                                         "); the server is restarting";
                    deliver(j, r);
                }
                if (options_.on_fatal) options_.on_fatal(sl.fatal_pending);
                return;  // nothing runs on this context again
            }
            const int64_t snap = sl.be->snapshot_pos(LmBackend::kTurnSlot);  // -1 when poisoned (a Hint): turn_pos
                                                                             // "none", never a save against it
            release_slot(snap);
            cv_.notify_all();
        } catch (const std::exception &ex) {
            escape(ex.what());
        } catch (...) {
            escape("a non-standard exception (no message)");
        }
    }
}

std::string Engine::fatal_error() const {
    std::lock_guard<std::mutex> lock(mu_);
    return fatal_;
}

int64_t Engine::resume_point(Slot &sl, const std::vector<int32_t> &prompt, int64_t id, bool &from_disk, bool &from_ram) {
    from_disk = false, from_ram = false;
    const int64_t P = (int64_t)prompt.size();
    int64_t lcp = 0;
    while (lcp < (int64_t)sl.seq.size() && lcp < P && sl.seq[(size_t)lcp] == prompt[(size_t)lcp]) ++lcp;
    // In memory: the live state if the prompt extends it, else the better of the snapshots.
    int64_t best = 0, how = -2;  // -1 = live, >= 0 = slot
    if (lcp == (int64_t)sl.seq.size() && lcp < P && sl.be->pos() == lcp) best = lcp, how = -1;
    for (int slot : {LmBackend::kTurnSlot, LmBackend::kSystemSlot, LmBackend::kUserSlot}) {
        const int64_t sp = sl.be->snapshot_pos(slot);
        if (sp > best && sp <= lcp && sp < P && sl.be->can_restore_snapshot(slot)) best = sp, how = slot;
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
                sl.be->import_state(state, to, from);  // a delta entry: its base (0, base), then the delta (base, dn)
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
                sl.seq.assign(prompt.begin(), prompt.begin() + dn);
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
        sl.be->restore_snapshot((int)how);
        sl.seq.resize((size_t)best);
        slog_row(LogLevel::Info, "resume", "%s of %s tokens from the GPU snapshot at %s", fmt_n(best).c_str(),
                 fmt_n(P).c_str(),
                 how == LmBackend::kTurnSlot     ? "the previous prompt's end"
                 : how == LmBackend::kSystemSlot ? "the system prompt's end"
                                                 : "the start of the previous prompt's last user message");
        return best;
    }
    sl.be->reset();
    sl.seq.clear();
    none();
    return 0;
}

void Engine::capture(Slot &sl, const GenerationRequest &req, const std::vector<CaptureStep> &steps, const std::string &reason) {
    // A side channel: a failure here is logged as an error and never touches the session or the response.
    try {
        const int64_t P = (int64_t)req.prompt.size();
        STRIX_CHECK((int64_t)sl.seq.size() >= P && std::equal(req.prompt.begin(), req.prompt.end(), sl.seq.begin()),
                    "Engine::capture: the session (", sl.seq.size(), " tokens) doesn't start with the ", P, "-token prompt");
        CaptureRecord rec;
        rec.id = req.id, rec.prompt_n = P, rec.tokens = sl.seq, rec.steps = steps;
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
    // Only when it happens: a conversation that lost batching to its own history is worth a line, one that
    // never did is not. This is why a slot verifies never show up in a batch, in the block that pays for it.
    if (r.verify_solo_mrope > 0)
        slog_row(LogLevel::Info, "batching",
                 "alone %s (mRoPE position state: an image in this conversation - a multi-run forward takes scalar "
                 "positions only), shared %s",
                 fmt_n(r.verify_solo_mrope).c_str(), fmt_n(r.verify_shared_runs).c_str());
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

void Engine::run(Job &job, Slot &sl) {
    GenerationRequest &req = job.req;
    GenerationSink &sink = *job.sink;
    GenerationResult res;
    int64_t generated_counted = 0;  // of res.completion_tokens, already in stats_.generated_tokens (per decode step)
    const double t_start = now_s();
    res.queue_ms = (t_start - job.enqueued) * 1e3;
    res.prompt_tokens = (int64_t)req.prompt.size();
    const BackendStats b0 = sl.be->backend_stats();
    // The request's log block (serve/log.hpp): the rule, the id and the HTTP layer's rows now; the footer (log_footer)
    // after the saves - whichever way run() returns - then a blank line before the next block.
    // The blank line is one no-break space (U+00A0): journald drops a line that is empty or only ASCII whitespace
    // after the "<N>" prefix.
    slog(LogLevel::Info, "%s", kLogRule);
    slog(LogLevel::Info, "Request %lld", (long long)req.id);
    for (const auto &[label, text] : req.log_header) slog_row(LogLevel::Info, label.c_str(), "%s", text.c_str());
    if (!job.slot_why.empty()) slog_row(LogLevel::Info, "slot", "%s", job.slot_why.c_str());
    if (!options_.degraded.empty())
        slog_row(LogLevel::Warning, "degraded", "WARNING: performance degraded - %s (the startup WARNINGs say what each costs)",
                 options_.degraded.c_str());
    struct BlockEnd {
        const GenerationResult &r;
        ~BlockEnd() {
            log_footer(r);
            slog(LogLevel::Info, "\xc2\xa0");
        }
    } block_end{res};
    bool finished = false;  // finish() ran once: the stats counted, the result delivered (or its delivery was
                            // attempted) - it is never re-counted and never re-delivered
    const auto finish = [&](const std::string &reason) {
        if (finished) {  // the catch reached finish a second time: the first one's delivery happened (or threw in
                         // the sink - result_sent then says) - keep the first result, log the second trouble
            slog_row(LogLevel::Error, "error", "ERROR after the result was already delivered (reason '%s'): the "
                                               "failure handling itself failed - keeping the first result",
                     reason.c_str());
            return;
        }
        finished = true;
        res.finish_reason = reason;
        res.total_ms = (now_s() - job.enqueued) * 1e3;
        const BackendStats b = sl.be->backend_stats();
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
            stats_.generated_tokens += res.completion_tokens - generated_counted;  // the rest of the per-step count
            stats_.mtp_drafted_tokens += res.mtp_drafted_tokens;
            stats_.think_nudges += res.think_nudges;
            stats_.think_end_blocked += res.think_end_blocked;
            stats_.think_closed_for_call += res.think_closed_for_call;
            stats_.mtp_accepted_tokens += res.mtp_accepted_tokens;
            stats_.mtp_rollbacks += res.mtp_rollbacks;
            for (int64_t i = 0; i < kMtpMaxDraft; ++i) stats_.mtp_reject_at[i] += res.mtp_reject_at[i];
            stats_.decode_seconds += res.decode_ms / 1e3;
            if (res.decode_ms > 0 && res.completion_tokens > 1)
                stats_.last_decode_tps = (double)(res.completion_tokens - 1) / (res.decode_ms / 1e3);
        }
        log_detail(res);
        job.result_sent = true;  // the worker's guard leaves the one-delivery duty to this call
        sink.on_done(res);       // sinks must not throw (GenerationSink's contract, serve/engine.hpp)
    };
    try {
        sink.on_start();
        // Image prompt (vision, Phase D): resume / cache matching and sl.seq run on the key stream (image_pad
        // runs replaced by content-derived pseudo ids, serve/vision_request.cpp), the forwards on the real
        // ids. Text prompts: both are req.prompt - the path is bit-identical and allocation-free.
        const bool has_images = !req.image_pos3.empty();
        STRIX_CHECK(has_images == !req.prefill_images.empty(),
                    "Engine::run: ", req.prefill_images.size(), " prefill images with ",
                    req.image_pos3.size() / 3, " triple rows - an image prompt carries both");
        if (has_images) {
            STRIX_CHECK(sl.be->has_vision(), "Engine::run: an image prompt on a backend without a vision tower");
            STRIX_CHECK(req.image_pos3.size() == 3 * req.prompt.size(),
                        "Engine::run: ", req.image_pos3.size() / 3, " walk rows for a prompt of ",
                        req.prompt.size());
            STRIX_CHECK(!req.cache_key.empty() && req.cache_key.size() == req.prompt.size(),
                        "Engine::run: an image prompt must carry its cache-key stream (",
                        req.cache_key.size(), " ids for a prompt of ", req.prompt.size(), ")");
            for (const PrefillImage &im : req.prefill_images)
                STRIX_CHECK(im.row_start >= 0 && im.rows >= 1 && im.row_start + im.rows <= (int64_t)req.prompt.size() &&
                                im.prefill.grid_t == 1 && im.prefill.grid_h >= 2 && im.prefill.grid_w >= 2,
                            "Engine::run: image block rows [", im.row_start, ", ", im.row_start + im.rows,
                            ") outside the prompt of ", req.prompt.size(), " (grid ", im.prefill.grid_h, "x",
                            im.prefill.grid_w, ")");
        }
        const std::vector<int32_t> &prompt = has_images ? req.cache_key : req.prompt;
        const int64_t P = (int64_t)prompt.size();
        bool from_disk = false, from_ram = false;
        int64_t live_common = 0;  // before resume_point moves the live session
        while (live_common < (int64_t)sl.seq.size() && live_common < P && sl.seq[(size_t)live_common] == prompt[(size_t)live_common])
            ++live_common;
        const int64_t live_n = (int64_t)sl.seq.size();
        int64_t start = resume_point(sl, prompt, req.id, from_disk, from_ram);
        if (has_images && start > 0) {  // a state saved mid-image (or a key collision): the image rows would
            bool mid_image = false;     // not be in it - prefill the whole prompt instead
            for (const PrefillImage &im : req.prefill_images)
                mid_image |= start > im.row_start && start < im.row_start + im.rows;
            if (mid_image) {
                sl.be->reset();
                sl.seq.clear();
                slog_row(LogLevel::Info, "resume", "lost: a saved state ends inside an image block - all %s "
                         "tokens prefilled", fmt_n(P).c_str());
                start = 0;
            }
        }
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
        // An image prompt (Phase D) also ends at the image block edges: one forward carries one whole block
        // (Phase B's one-image-per-forward contract), and a block ends at a max-chunk multiple before any
        // forward can start inside it - so every image forward is exactly the block, the text before it in
        // the forward it follows.
        auto block_for = [&](int64_t pos) -> const PrefillImage * {
            for (const PrefillImage &im : req.prefill_images)
                if (im.row_start <= pos && pos < im.row_start + im.rows) return &im;
            return nullptr;
        };
        auto chunk_end = [&](int64_t pos) {
            int64_t step = sl.be->max_chunk();
            // Not for an image prompt: its blocks (up to a max chunk) must each fit one forward.
            if (!has_images && arbiter_ && options_.shared_chunk > 0 && arbiter_->others_decoding(sl.index))
                step = std::min(step, options_.shared_chunk);
            int64_t end = std::min(P, pos + step);
            if (cut > pos && cut < end) end = cut;
            if (ust > pos && ust < end) end = ust;
            if (save_sys && sys > pos && sys < end) end = sys;
            if (has_images) {
                if (const PrefillImage *im = block_for(pos)) {
                    const int64_t blk_end = im->row_start + im->rows;
                    STRIX_CHECK(blk_end <= end, "Engine::run: the image block at row ", im->row_start,
                                " ends at ", blk_end, ", past the forward end ", end, " (max chunk ",
                                sl.be->max_chunk(), ") - the block must fit one forward");
                    end = blk_end;
                }
                for (const PrefillImage &im : req.prefill_images)
                    if (im.row_start > pos && im.row_start < end) end = im.row_start;
            }
            return end;
        };
        const double t_prefill = now_s();
        const auto progress = [&](int64_t pos) { sink.on_prefill({P, start, pos, (now_s() - t_prefill) * 1e3}); };
        progress(start);
        if (start == ust)  // resumed exactly at the user turn: its start state is the live state right now
            sl.be->save_snapshot(LmBackend::kUserSlot);
        for (int64_t pos = start; pos < P;) {
            const int64_t end = chunk_end(pos);
            if (end < P)  // the real ids: the lookahead feeds ple_ngram_ids, and the negative cache-key ids
                sl.be->set_lookahead(std::vector<int32_t>(req.prompt.begin() + end,  // are matchers, never model
                                  req.prompt.begin() + chunk_end(end)));          // inputs (vision)
            if (const PrefillImage *im = block_for(pos)) {  // the whole block (chunk_end checked), in one forward
                STRIX_CHECK(pos == im->row_start && end == im->row_start + im->rows,
                            "Engine::run: the image forward covers rows [", pos, ", ", end, "), the block is [",
                            im->row_start, ", ", im->row_start + im->rows, ")");
                PrefillImage decoded;  // an image the HTTP layer knew (no pixels) that the resume didn't cover
                if (im->prefill.pixel_values.empty()) {
                    STRIX_CHECK(static_cast<bool>(im->prefill.decode), "Engine::run: the image block at row ",
                                im->row_start, " (grid ", im->prefill.grid_h, "x", im->prefill.grid_w,
                                ") carries neither pixel_values nor a decode - the request layer must give one");
                    const double t_dec = now_s();
                    decoded.prefill.pixel_values = im->prefill.decode();
                    decoded.prefill.grid_t = im->prefill.grid_t;
                    decoded.prefill.grid_h = im->prefill.grid_h, decoded.prefill.grid_w = im->prefill.grid_w;
                    decoded.row_start = im->row_start, decoded.rows = im->rows;
                    slog_row(LogLevel::Debug, "vision", "image at row %s decoded at prefill (%lldx%lld grid, %.0f ms): "
                             "the resume stopped before it", fmt_n(im->row_start).c_str(),
                             (long long)im->prefill.grid_h, (long long)im->prefill.grid_w, (now_s() - t_dec) * 1e3);
                    im = &decoded;
                }
                logits = sl.be->forward_image_prefill(
                    std::vector<int32_t>(req.prompt.begin() + pos, req.prompt.begin() + end), *im,
                    std::vector<int64_t>(req.image_pos3.begin() + 3 * pos, req.image_pos3.begin() + 3 * end),
                    end == P ? 1 : 0, cands, n_valid, end == P ? masks_for({}) : nullptr);
            } else {
                logits = sl.be->forward_rows(std::vector<int32_t>(req.prompt.begin() + pos, req.prompt.begin() + end),
                                          end == P ? 1 : 0, cands, n_valid,
                                          end == P ? masks_for({}) : nullptr);
            }
            sl.seq.insert(sl.seq.end(), prompt.begin() + pos, prompt.begin() + end);
            {  // per chunk, so /metrics' prefill counter moves during a long prefill (tools/strix_rate.py)
                std::lock_guard<std::mutex> lock(mu_);
                stats_.prefill_tokens += end - pos;
            }
            pos = end;
            if (pos == cut) sl.be->save_snapshot(LmBackend::kTurnSlot);
            if (pos == ust) sl.be->save_snapshot(LmBackend::kUserSlot);
            if (save_sys && pos == sys) sl.be->save_snapshot(LmBackend::kSystemSlot);
            progress(pos);
            if (pos < P && sink.cancelled()) {
                res.prompt_ms = (now_s() - t_start) * 1e3;
                return finish("cancelled");
            }
        }
        // Decoding from here: with several slots the arbiter waits for this slot's verifies to batch them. Ended
        // when the decode loop ends - not held through the response write and the prompt cache's export, where
        // the other slots would still defer to it (shared prefill chunks) with the GPU free.
        struct Decoding {
            SeqBackend *p;
            explicit Decoding(SeqBackend *q) : p(q) {
                if (p) p->set_decoding(true);
            }
            void end() {
                if (p) p->set_decoding(false);
                p = nullptr;
            }
            ~Decoding() { end(); }
        } decoding{sl.proxy.get()};
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
            stats_.prompt_tokens += P, stats_.cached_tokens += start;  // prefill_tokens: per chunk above
            (start > 0 ? stats_.cache_hits : stats_.cache_misses)++;
            if (from_disk) ++stats_.disk_hits, stats_.disk_tokens += start, stats_.ram_hits += from_ram;
            stats_.prefill_seconds += res.prompt_ms / 1e3;
            if (P > start) stats_.last_prefill_tps = (double)(P - start) / (res.prompt_ms / 1e3);
        }

        Sampler sampler(req.sampling, tok_.size());
        OutputParser::Config parser_cfg = req.parser;
        // Literal tags (config literal-tags) - not under a response_format grammar, whose think-block end is the token.
        parser_cfg.literal_tags = options_.literal_tags && !so;
        OutputParser parser(tok_, parser_cfg);
        ThinkWatch watch{options_.think_nudge_policy};
        // The think-block guard: no end of turn while the think block or a tool call is open - the model reaches for
        // <|im_end|> when it means to name a tag in inline code (`</tool_call>`) or in code it writes into a call
        // (`id_of("<|im_end|>")`), ending the turn mid thought or mid call.
        const std::vector<int32_t> end_ids{im_end_, eot_};
        const auto guard = [&](LogitRows &rows, int64_t r) {
            if (options_.think_end_guard && (parser.in_reasoning() || parser.in_call()) && ban_tokens(rows, r, end_ids))
                ++res.think_end_blocked;
        };
        // A sampled token just fed to the parser: the nudge watch follows the reasoning ones.
        // think_tail: the reasoning text's last bytes, for the budget's sentence-end wait (ends_sentence).
        std::string think_tail;
        const auto watch_tok = [&](int32_t t) {
            if (!parser.in_reasoning()) return;
            const std::string piece = tok_.decode({t});
            if (options_.think_nudge) watch.observe(t, piece);
            if (req.thinking_budget_grace > 0) {
                think_tail += piece;
                if (think_tail.size() > 512) think_tail.erase(0, think_tail.size() - 256);
            }
        };
        // The thinking budget's stop is due: the budget spent and - with a grace - the reasoning at a sentence end, or
        // the grace spent too.
        const auto stop_due = [&] {
            if (req.thinking_budget < 0 || res.thinking_budget_hit || !parser.in_reasoning()) return false;
            const int64_t n = parser.reasoning_tokens();
            if (n < req.thinking_budget) return false;
            return req.thinking_budget_grace == 0 || n >= req.thinking_budget + req.thinking_budget_grace ||
                   ends_sentence(think_tail);
        };
        std::vector<OutputEvent> events;
        std::string reason;
        std::vector<CaptureStep> steps;  // what each decode step did, for --capture-dir
        auto step = [&](int64_t drafted, int64_t accepted, int64_t emitted) {
            if (!options_.capture_dir.empty())
                steps.push_back({(uint8_t)drafted, (uint8_t)accepted, (uint8_t)emitted});
        };
        // A tool call begun inside the think block (Options::think_close_on_call): a sampled <tool_call> right after
        // a newline while the think block is open. Not under a response_format grammar, which owns the structure.
        const auto misplaced_call = [&](int32_t t) {
            return options_.think_close_on_call && !so && t == call_begin_ && parser.in_reasoning() && parser.after_newline();
        };
        // Feeds kThinkCloseBeforeCall, then the sampled <tool_call> (counted already, not yet fed or forwarded), in one
        // forward: the call then parses as one. False: no room left for it under max_tokens (the caller ends, length).
        const auto close_think_before_call = [&](int32_t call_tok) {
            STRIX_CHECK(call_tok == call_begin_ && parser.in_reasoning(), "Engine: closing the think block before token ",
                        call_tok, " (expected <tool_call> ", call_begin_, ") with the think block ",
                        parser.in_reasoning() ? "open" : "closed");
            const int64_t n = (int64_t)think_close_.size();
            if (res.completion_tokens + n >= req.max_tokens) return false;
            std::vector<int32_t> fed = think_close_;
            fed.push_back(call_tok);
            for (int32_t t : fed) parser.feed(t, events);  // no stop string can complete: none of it is content
            STRIX_CHECK(parser.in_call(), "Engine: the think block closed before a tool call, but the parser isn't in the call");
            if (!events.empty()) sink.on_events(events), events.clear();
            res.completion_tokens += n, res.fed_tokens += n;
            ++res.think_closed_for_call;
            ++res.decode_forwards;
            logits = sl.be->forward_rows(fed, 1, cands, n_valid, masks_for({}));
            sl.seq.insert(sl.seq.end(), fed.begin(), fed.end());
            step(kThinkingStopStep, 0, n);  // the fed close, then the sampled <tool_call> as a step of its own
            step(0, 0, 1);
            return true;
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
                const bool budget_due = stop_due();
                const bool nudge_due = options_.think_nudge && parser.in_reasoning() && watch.due() != 0;
                if (budget_due || nudge_due) {
                    ++res.decode_forwards;
                    logits = sl.be->forward_rows({carry}, 1, cands, n_valid, masks_for({}));
                    sl.seq.push_back(carry);
                    step(0, 0, 1);
                    carry = -1;
                }
            }
            // Thinking budget spent: the engine closes the think block itself (fed, not sampled), then goes on sampling. Never
            // while a carried token waits (flushed just above when due): the injection would be forwarded ahead of it.
            if (carry < 0 && stop_due()) {
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
                logits = sl.be->forward_rows(thinking_stop_, 1, cands, n_valid, masks_for({}));
                sl.seq.insert(sl.seq.end(), thinking_stop_.begin(), thinking_stop_.end());
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
                        slog_row(LogLevel::Debug, "", "%.0f%% of the last %s thinking tokens repeat earlier ones",
                                 100.0 * watch.window_rate(), fmt_n((long long)options_.think_nudge_policy.window).c_str());
                        for (int32_t t : nt) {
                            parser.feed(t, events);
                            if (so) so->feed(t);
                        }
                        if (!events.empty()) sink.on_events(events), events.clear();
                        res.completion_tokens += n;
                        ++res.think_nudges;
                        res.fed_tokens += n;
                        logits = sl.be->forward_rows(nt, 1, cands, n_valid, masks_for({}));
                        sl.seq.insert(sl.seq.end(), nt.begin(), nt.end());
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
                guard(logits, 0);
                id = sampler.sample(logits, 0);
                ++res.completion_tokens;
                if (id == im_end_ || id == eot_) {
                    reason = "stop";
                    break;
                }
                if (misplaced_call(id)) {
                    if (!close_think_before_call(id)) {
                        reason = "length";
                        break;
                    }
                    continue;
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
            // sampled from the trunk's own logits (row j) - a draft only saves the forward when it matches. No drafts
            // when id brought the budget's stop due: the stop follows id directly.
            if (res.completion_tokens > generated_counted) {  // /metrics' generated counter moves per step
                std::lock_guard<std::mutex> lock(mu_);
                stats_.generated_tokens += res.completion_tokens - generated_counted;
                generated_counted = res.completion_tokens;
            }
            if (sl.be->has_mtp() && !stop_due()) {
                const int64_t room = req.max_tokens - res.completion_tokens;  // >= 1 here
                // Several slots decoding: at most as many drafts as let every slot's verify run fit one batched
                // forward (max_verify_rows / decoding slots - 1: 2 slots 7, 3 slots 4, 4 slots 3). Past the budget the
                // arbiter splits the batch in two forwards, and each slot's longer chain runs alone on the GPU.
                int64_t draft_max = options_.mtp_draft;
                if (arbiter_) {
                    const int64_t n = std::max<int64_t>(1, arbiter_->decoding_count());
                    draft_max = std::min(draft_max, std::max<int64_t>(1, arbiter_->backend().max_verify_rows() / n - 1));
                }
                const int64_t steps = std::min(draft_max, room);
                std::vector<int32_t> ids{id};
                // Each token's PLE rows start loading as soon as it is known, while the next draft runs - so the
                // verify (or id's own forward, when no draft passes) finds them in the row cache.
                sl.be->prefetch_ple(ids, 0);
                StructuredOutput::Cursor draft_at = so ? so->cursor() : StructuredOutput::Cursor{};
                for (int64_t step = 0; step < steps; ++step) {
                    const Top2 t = sl.be->forward_mtp_top2(ids.back(), step);
                    STRIX_CHECK(!t.nan, "Engine: NaN in the MTP draft logits (step ", step, ")");
                    const int32_t d = t.best;
                    if (t.best_v - t.second_v < options_.mtp_margin || d == im_end_ || d == eot_) break;
                    // A <tool_call> drafted inside the think block might need the think close fed ahead of it: no draft.
                    if (d == call_begin_ && options_.think_close_on_call && parser.in_reasoning()) break;
                    // A draft the response format refuses would be rejected anyway: don't verify it.
                    if (so) {
                        if (!so->allows(draft_at, d)) break;
                        draft_at = so->after(draft_at, d);
                    }
                    ids.push_back(d);
                    if (step + 1 < steps) sl.be->prefetch_ple(ids, (int64_t)ids.size() - 1);  // the last: nothing to hide behind
                }
                const int64_t k = (int64_t)ids.size() - 1;
                if (k > 0) {
                    res.mtp_drafted_tokens += k;
                    ++res.decode_forwards;
                    LogitRows ml = sl.be->forward_verify_rows(
                        ids, k + 1, cands, n_valid, masks_for(std::vector<int32_t>(ids.begin() + 1, ids.end())));
                    STRIX_CHECK(ml.rows == k + 1, "Engine: verify of ", k + 1, " tokens returned ", ml.rows,
                                " logits rows");
                    // Accept drafts while the sampled token is the draft.
                    int64_t j = 0;
                    int32_t v = -1;
                    bool ended = false, cut = false;
                    for (; j < k; ++j) {
                        guard(ml, j);
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
                        // The budget's stop due after this draft, with drafts after it still unchecked: keep id and the
                        // j + 1 accepted drafts and stop accepting, so the stop lands right here (at a sentence end)
                        // rather than after the rest of the chain. Row j + 1 is the logits after the last kept draft.
                        if (j + 1 < k && stop_due()) {
                            ++j;
                            cut = true;
                            break;
                        }
                    }
                    if (cut) {  // at the budget's stop (above): id and j accepted drafts kept
                        sl.be->keep_verify_prefix(ids, j + 1);
                        sl.seq.insert(sl.seq.end(), ids.begin(), ids.begin() + j + 1);
                        step(k, j, j + 1);
                        logits = ml.row_of(j);
                        continue;
                    }
                    if (ended) {  // keep id + the j accepted drafts in the session, drop the rest
                        if (j < k) sl.be->keep_verify_prefix(ids, j + 1);
                        else sl.be->keep_verify();
                        sl.seq.insert(sl.seq.end(), ids.begin(), ids.begin() + j + 1);
                        step(k, j, j + 1);
                        break;
                    }
                    if (j == k) {  // all accepted: the last row is the next token's logits
                        sl.be->keep_verify();
                        sl.seq.insert(sl.seq.end(), ids.begin(), ids.end());
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
                        sl.be->prefetch_ple(next, j + 1);
                    }
                    sl.be->keep_verify_prefix(ids, j + 1);
                    sl.seq.insert(sl.seq.end(), ids.begin(), ids.begin() + j + 1);
                    if (v == im_end_ || v == eot_) {
                        step(k, j, j + 1);
                        reason = "stop";
                        break;
                    }
                    if (misplaced_call(v)) {  // fed with the close in a forward of its own, not carried
                        step(k, j, j + 1);
                        if (!close_think_before_call(v)) {
                            reason = "length";
                            break;
                        }
                        continue;
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
            if (arbiter_ && sl.be->has_mtp()) {
                // Several slots: a step with no draft is a verify of id alone, so it joins the other slots' batch. As a
                // plain forward it ran alone on the GPU and held their batch until it finished.
                LogitRows ml = sl.be->forward_verify_rows({id}, 1, cands, n_valid, masks_for({}));
                STRIX_CHECK(ml.rows == 1, "Engine: verify of 1 token (no draft) returned ", ml.rows, " logits rows");
                sl.be->keep_verify();
                logits = ml.row_of(0);
            } else {
                logits = sl.be->forward_rows({id}, 1, cands, n_valid, masks_for({}));
            }
            sl.seq.push_back(id);
            step(0, 0, 1);
        }
        decoding.end();
        if (sl.proxy) {  // the window closed with the decode: these numbers are this request own
            const GpuArbiter::VerifyWindow vw = sl.proxy->verify_window();
            res.verify_shared_runs = vw.shared, res.verify_solo_mrope = vw.solo_mrope;
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
        if (res.think_closed_for_call > 0)
            slog_row(LogLevel::Debug, "guard", "think block closed ahead of a tool call begun inside it %s time%s (</think> fed)",
                     fmt_n(res.think_closed_for_call).c_str(), res.think_closed_for_call == 1 ? "" : "s");
        if (res.think_end_blocked > 0)
            slog_row(LogLevel::Debug, "guard", "end of turn blocked inside the think block or a tool call %s time%s (it was the top token)",
                     fmt_n(res.think_end_blocked).c_str(), res.think_end_blocked == 1 ? "" : "s");
        finish(reason);
        // Captures replay token streams - an image prompt's stream carries image_pad rows without their
        // pixels, so image requests are not captured.
        if (!options_.capture_dir.empty() && reason != "cancelled" && !has_images) capture(sl, req, steps, reason);
        // After the response: the new prefixes go to the prompt cache's RAM tier (the disk only later, if at all).
        if (cache_ && reason != "cancelled") {
            const double t0 = now_s();
            int saved = 0, rows = 0;  // rows: the first gets the "cache" label
            const auto save = [&](int slot, int64_t n, PromptCache::Kind kind, const char *what) {
                // Already cached first: a request resumed exactly at its turn entry has no snapshot there, and needs none.
                const char *skip = n <= 0                                ? "no such point in the prompt"
                                   : cache_->contains(prompt.data(), n) ? nullptr
                                   : sl.be->snapshot_pos(slot) != n         ? "no snapshot there"
                                   : !sl.be->can_restore_snapshot(slot)     ? "snapshot no longer valid"
                                                                         : "";
                if (skip == nullptr) return;  // already cached
                if (*skip) {
                    slog_row(LogLevel::Debug, rows++ ? "" : "cache", "%s, %s tokens: not saved - %s (snapshot at %s)",
                             what, fmt_n(n).c_str(), skip, fmt_n(sl.be->snapshot_pos(slot)).c_str());
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
                // Room before the buffer grows (PromptCache::make_room_for): else no save - never past the margin.
                const uint64_t want = sl.be->snapshot_bytes(slot, base_n);
                if (want > state.capacity() && !cache_->make_room_for(want - state.capacity())) {
                    slog_row(LogLevel::Warning, rows++ ? "" : "cache",
                             "%s, %s tokens: not saved - no room in RAM (needs %s GB more; MemAvailable %s GiB, margin %s "
                             "GiB)", what, fmt_n(n).c_str(), fmt_rate((want - state.capacity()) / 1e9, 1).c_str(),
                             fmt_rate(cache_->mem_available_now() / (double)(1ull << 30), 1).c_str(),
                             fmt_rate(cache_->options().ram_margin / (double)(1ull << 30), 0).c_str());
                    slog_row(LogLevel::Warning, "", "%s", cache_->ram_summary().c_str());
                    cache_->give_back(std::move(state));
                    return;
                }
                sl.be->export_snapshot(slot, state, base_n);
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
    } catch (const GpuFatalError &e) {
        // The GPU context is gone (strixite issue #6: an aperture violation, then hipErrorIllegalAddress on every
        // request until a restart): no reset - it would fail too. The worker fails the queue and calls on_fatal.
        slog_row(LogLevel::Error, "error", "FATAL: %s", e.what());
        slog_row(LogLevel::Error, "", "FATAL: no request can run on this GPU context: failing the queued ones, refusing "
                 "new ones%s", options_.on_fatal ? ", the server exits for its supervisor to restart it" : "");
        sl.fatal_pending = e.what();
        sl.seq.clear();
        res.error = e.what();
        finish("error");
    } catch (const std::exception &e) {
        // A failed forward leaves the session unusable until reset: start clean for the next request.
        slog_row(LogLevel::Error, "error", "ERROR: %s", e.what());
        try {
            sl.be->reset();
        } catch (const std::exception &e2) {
            slog_row(LogLevel::Error, "", "ERROR: the reset after it failed too: %s", e2.what());
        }
        sl.seq.clear();
        res.error = e.what();
        finish("error");
    }
}

}  // namespace strix
