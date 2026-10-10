#pragma once

// The serving loop: one worker thread owns the model and runs requests
// one at a time in arrival order (FIFO). Per request: resume from the longest reusable prefix, prefill the rest in
// chunks, then sample / parse / stream token by token until an end-of-turn token, a stop string, max_tokens or a
// cancel.
//
// Prefix reuse ("Sequence state"): the engine knows the tokens in the session (the last prompt + what was fed of
// its generation). A new prompt resumes from the live state when it extends those tokens exactly, else from the
// snapshot - saved at the prompt's last <|im_start|> (the start of the generation prompt, a stable token
// boundary: the next request re-renders the assistant turn from there, dropping or rewriting its reasoning) - when
// the new prompt still matches up to it, else from scratch.
//
// The model sits behind LmBackend, so the loop and the HTTP layer above it are testable without a GPU.

#include "common/check.hpp"
#include "serve/capture.hpp"
#include "serve/output_parser.hpp"
#include "serve/prompt_cache.hpp"
#include "serve/grammar.hpp"
#include "serve/sampler.hpp"
#include "serve/structured_output.hpp"
#include "serve/think_nudge.hpp"
#include "serve/token_mask.hpp"
#include "serve/tokenizer.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strix {

// Counters a backend keeps for /metrics and the request log - cumulative, readable from any thread. PLE n-gram rows:
// requested by forwards, distinct per gather, served from the row cache, read from the table file (SSD / page cache);
// gathers and their seconds; forwards that waited for their rows at the PLE layer (the GPU idles meanwhile);
// prefetches (LmBackend::prefetch_ple): rows they read from the table file into the cache (a gather then counts them
// as cached - rows read from the file in all: ngram_rows_read + ngram_rows_prefetched), jobs, seconds, skipped.
struct BackendStats {
    int64_t ngram_rows_requested = 0, ngram_rows_unique = 0, ngram_rows_cached = 0, ngram_rows_read = 0;
    int64_t ngram_rows_prefetched = 0;
    int64_t ple_gathers = 0, ple_waits = 0, ple_prefetches = 0, ple_prefetch_skipped = 0;
    double ple_gather_seconds = 0, ple_wait_seconds = 0, ple_wait_max_seconds = 0, ple_prefetch_seconds = 0;
};

class LmBackend {
public:
    virtual ~LmBackend() = default;
    virtual int64_t capacity() const = 0;    // positions
    virtual int64_t max_chunk() const = 0;   // tokens per forward
    virtual int64_t logits_row() const = 0;  // floats per logits row (>= the tokenizer's size)
    virtual int64_t pos() const = 0;
    virtual void reset() = 0;
    // Runs ids at pos().., advances pos(); returns the last token's logits if want_logits, else nothing.
    virtual std::vector<float> forward(const std::vector<int32_t> &ids, bool want_logits) = 0;
    // Hint before a forward: the ids of the forward after it (a long prompt's next chunk), so the backend can
    // prepare for them during this one. Only a hint - nothing may depend on it being used.
    virtual void set_lookahead(const std::vector<int32_t> &next_ids) { (void)next_ids; }
    virtual std::vector<float> forward(const std::vector<int32_t> &ids, int64_t n_logits) {
        return forward(ids, n_logits > 0);
    }
    virtual bool has_mtp() const { return false; }
    // The MTP head's logits for the token after token_id (step 0: token_id is the last sampled token; step > 0:
    // chained, token_id is step - 1's draft - Qwen4ExpSession::forward_mtp).
    virtual std::vector<float> forward_mtp(int32_t token_id, int64_t step) {
        (void)token_id, (void)step;
        return {};
    }
    // The draft's top2() (serve/sampler.hpp) - what the engine uses of it. The default takes forward_mtp's logits;
    // a backend may reduce them where they are (Qwen4ExpSession::forward_mtp_top2).
    virtual Top2 forward_mtp_top2(int32_t token_id, int64_t step) {
        const std::vector<float> ml = forward_mtp(token_id, step);
        STRIX_CHECK(!ml.empty(), "LmBackend::forward_mtp_top2: the backend's MTP draft (step ", step, ") returned no logits");
        return top2(ml.data(), (int64_t)ml.size());
    }
    // Verifying drafts: a forward whose state changes can be undone (drop) or kept - exactly one of the two before
    // anything else. The default saves / restores snapshot slot kDraftSlot; a backend may do it cheaper.
    virtual std::vector<float> forward_verify(const std::vector<int32_t> &ids, int64_t n_logits) {
        save_snapshot(kDraftSlot);
        return forward(ids, n_logits);
    }
    // forward / forward_verify with the logits as the sampler takes them (LogitRows): with cands, a backend may return
    // each row's top Sampler::kCandidates candidates over ids [0, n_valid) instead of the full row (the engine asks
    // only when Sampler::takes_candidates() - the sampled tokens are the same either way). The default returns full
    // rows; Qwen4ExpBackend reduces them on the GPU (kernels/logits_topk).
    // masks (structured output): null, or one allowed-token mask per returned row (LogitMasks; masks->rows ==
    // n_logits) - the rows then hold allowed ids only (candidates from those alone, or full rows with the rest -inf).
    virtual LogitRows forward_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                                   const LogitMasks *masks = nullptr) {
        (void)cands, (void)n_valid;
        LogitRows l = LogitRows::from_full(forward(ids, n_logits), logits_row());
        if (masks != nullptr) apply_masks(l, *masks);
        return l;
    }
    virtual LogitRows forward_verify_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid,
                                          const LogitMasks *masks = nullptr) {
        (void)cands, (void)n_valid;
        LogitRows l = LogitRows::from_full(forward_verify(ids, n_logits), logits_row());
        if (masks != nullptr) apply_masks(l, *masks);
        return l;
    }
    virtual BackendStats backend_stats() const { return {}; }
    // A hint (Qwen4ExpSession::prefetch_ple): the next forward starts with ids (while a verify awaits its outcome:
    // ids start where the verify did); fetch the inputs of ids[first, size) it reads from slow storage now. Changes
    // nothing a forward computes; the default does nothing.
    virtual void prefetch_ple(const std::vector<int32_t> &ids, int64_t first) { (void)ids, (void)first; }
    virtual void keep_verify() {}
    virtual void drop_verify() { restore_snapshot(kDraftSlot); }
    // The third way out: keep only the verify's first `rows` of `ids` (1 <= rows < ids.size()) - the state as if
    // just those had been forwarded (an MTP rejection). The default drops the verify and forwards them again; a
    // backend may keep them without re-running (Qwen4ExpSession::keep_verify_prefix).
    virtual void keep_verify_prefix(const std::vector<int32_t> &ids, int64_t rows) {
        drop_verify();
        forward(std::vector<int32_t>(ids.begin(), ids.begin() + rows), (int64_t)0);
    }
    // Snapshot slots (Qwen4ExpSnapshot): save at pos(), restore if still valid. kTurnSlot: the prompt's generation
    // prompt; kSystemSlot: the end of the system prompt (for the disk cache); kDraftSlot: MTP rollback; kUserSlot:
    // the start of the prompt's last user message - the user-turn checkpoint (for the disk cache): a client that
    // later rewrites that message (e.g. drops a reminder it appended) still resumes here, not at the system prefix.
    static constexpr int kTurnSlot = 0, kSystemSlot = 1, kDraftSlot = 2, kUserSlot = 3;
    virtual void save_snapshot(int slot) = 0;
    virtual int64_t snapshot_pos(int slot) const = 0;  // -1 if none
    virtual bool can_restore_snapshot(int slot) const = 0;
    virtual void restore_snapshot(int slot) = 0;
    // States for the prompt cache: a slot's state (KV below its position included) out to host bytes, and host bytes
    // in as the state for positions [0, n) (pos() = n; every slot invalid after). from > 0: a delta - only what changed
    // since position `from` (the KV rows [from, n) and every per-step part; Qwen4ExpSession::state_bytes), imported
    // right after its base (import_state(base, from), then import_state(delta, n, from)).
    virtual void export_snapshot(int slot, HostBuffer &out, int64_t from = 0) = 0;  // resizes out to the state's size
    virtual void import_state(const HostBuffer &state, int64_t n, int64_t from = 0) = 0;
    // What a saved state fits: model, weights, layout, activation dtype (PromptCache refuses others).
    virtual std::string state_fingerprint() const = 0;
    virtual std::string describe() const = 0;  // for /health and logs
};

struct GenerationRequest {
    std::vector<int32_t> prompt;
    int64_t max_tokens = 0;  // >= 1; prompt + max_tokens <= capacity (the API layer clamps)
    SamplingParams sampling;
    OutputParser::Config parser;  // parser.tools must outlive the request
    // Reasoning tokens allowed before the engine closes the think block itself (feeding kThinkingStop); -1 =
    // no budget. The API layer sets it (an explicit budget, or the answer room).
    int64_t thinking_budget = -1;
    // The prompt's last message is the user's (not tool results): the prefix up to that message's start goes to
    // the disk cache as a checkpoint (PromptCache::Kind::Checkpoint), kept when later turns extend it.
    bool user_turn = false;
    // One message and no tools (a batch client's one-off prompt, nothing a later turn extends): its prompt cache
    // entries stay in RAM and never reach the disk (PromptCache "RAM-only entries").
    bool one_shot = false;
    // Structured output (response_format / tool_choice): the compiled grammar
    // the answer must follow, or null. Shared by requests with the same schema; the engine thread alone uses it.
    std::shared_ptr<grammar::GrammarAutomaton> grammar;
    int64_t id = 0;       // the HTTP layer's request id, for log lines ("Request <id>"); 0 = none given
    bool stream = false;  // for the log only (the HTTP layer streams or not)
    // The HTTP layer's rows for the top of the request's log block (label, text; an empty label goes on the row
    // above), printed by the engine thread when the request starts - so they never land inside another request's block.
    std::vector<std::pair<std::string, std::string>> log_header;
};

// What the engine feeds when a thinking budget is spent: Qwen's documented early-stop text, then the think close.
constexpr const char *kThinkingStop =
    "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly "
    "now.\n</think>\n\n";

constexpr int64_t kMtpMaxDraft = 4;  // rejection-point buckets: j = 0, 1, 2 and 3 or more accepted drafts

struct GenerationResult {
    std::string finish_reason;  // stop | length | tool_calls | cancelled | error
    std::string error;          // finish_reason == error
    int64_t prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0, reasoning_tokens = 0;
    int64_t mtp_drafted_tokens = 0, mtp_accepted_tokens = 0, mtp_rollbacks = 0;
    int64_t mtp_reject_at[kMtpMaxDraft] = {};  // rollbacks by drafts j accepted before the rejected one (last: j >= 3)
    double queue_ms = 0, prompt_ms = 0, decode_ms = 0;
    // From the submit to the end of the prefill (the first token's logits are ready) / to the response's end.
    double first_token_ms = 0, total_ms = 0;
    // Trunk forwards of sampled tokens in the decode (a verify counts once), and the tokens the engine fed itself
    // (thinking stop, nudges - each one forward of its own): (completion - fed) / forwards = tokens per forward.
    int64_t decode_forwards = 0, fed_tokens = 0;
    int64_t tool_calls = 0;       // tool calls in the answer
    bool dropped_partial_call = false;
    int64_t malformed_tool_calls = 0;  // streamed calls that went out but weren't well-formed (OutputParser)
    BackendStats backend;  // this request's share (wait_max: the longest wait of the session so far)
    bool thinking_budget_hit = false;  // the engine closed the think block (kThinkingStop was fed)
    int64_t think_nudges = 0;          // thinking nudges fed (serve/think_nudge.hpp)
};

// The request's side of the loop, called on the engine thread.
class GenerationSink {
public:
    virtual ~GenerationSink() = default;
    virtual void on_start() = 0;                               // left the queue
    virtual void on_progress(int64_t done, int64_t total) = 0;  // after each prefill chunk
    virtual void on_events(std::vector<OutputEvent> &events) = 0;
    virtual void on_done(const GenerationResult &r) = 0;
    virtual bool cancelled() = 0;  // polled between chunks and tokens
};

struct EngineStats {
    int64_t requests_ok = 0, requests_error = 0, requests_cancelled = 0;
    int64_t prompt_tokens = 0, cached_tokens = 0, prefill_tokens = 0, generated_tokens = 0;
    int64_t mtp_drafted_tokens = 0, mtp_accepted_tokens = 0, mtp_rollbacks = 0;
    int64_t mtp_reject_at[kMtpMaxDraft] = {};
    int64_t think_nudges = 0;  // thinking nudges fed
    int64_t cache_hits = 0, cache_misses = 0;
    double prefill_seconds = 0, decode_seconds = 0;
    double last_prefill_tps = 0, last_decode_tps = 0;
    int64_t queued = 0, busy = 0;
    // Seconds since the last request finished (or the engine started); 0 while one runs or waits. `busy` alone reads
    // 1 through a steady agent session, so a batch client sampling it saw a "stuck" flag; this lets it
    // wait for a real gap (e.g. idle_seconds >= 2).
    double idle_seconds = 0;
    int64_t live_tokens = 0, snapshot_pos = -1;
    // Prompts resumed from the prompt cache (either tier; ram_hits: of those, from its RAM tier) / states put in its
    // RAM tier (the disk gets them later, if at all).
    int64_t disk_hits = 0, disk_tokens = 0, ram_saves = 0, ram_hits = 0;
    double export_seconds = 0;
};

class Engine {
public:
    struct Options {
        PromptCache *cache = nullptr;
        float mtp_margin = 1.5f;  // logit margin (top1 - top2) required to draft MTP token
        int64_t mtp_draft = 1;    // most drafts per verify (the head chained), 1..15
        // Non-empty: every finished request's token ids and MTP steps are written there (serve/capture.hpp). The
        // directory must exist.
        std::string capture_dir;
        // The thinking nudge (serve/think_nudge.hpp): on in service; off for like-for-like
        // comparisons with engines that don't have it.
        bool think_nudge = true;
        ThinkWatch::Policy think_nudge_policy;  // when it fires (config think-nudge-rate / -min-tokens)
    };

    Engine(LmBackend &backend, const Tokenizer &tok, const Options &options);
    Engine(LmBackend &backend, const Tokenizer &tok, PromptCache *cache = nullptr, float mtp_margin = 1.5f);
    ~Engine();  // finishes the running request, fails the queued ones, joins the worker
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    void submit(GenerationRequest req, std::shared_ptr<GenerationSink> sink);
    EngineStats stats() const;
    BackendStats backend_stats() const { return be_.backend_stats(); }  // thread-safe per the backend's contract
    int64_t capacity() const { return capacity_; }
    const LmBackend &backend() const { return be_; }
    const PromptCache *prompt_cache() const { return cache_; }
    float mtp_margin() const { return options_.mtp_margin; }

private:
    struct Job {
        GenerationRequest req;
        std::shared_ptr<GenerationSink> sink;
        double enqueued;
    };
    void worker();
    void run(Job &job);
    // Sets the backend up to prefill from the returned position: the longest reusable prefix among the live state,
    // the two snapshot slots and the prompt cache (restoring / importing / resetting as needed). from_disk: from the
    // prompt cache (either tier); from_ram: from its RAM tier.
    int64_t resume_point(const std::vector<int32_t> &prompt, int64_t id, bool &from_disk, bool &from_ram);
    // Logs where the prompt left the state it shared the most with (the live session before resume_point, or a
    // prompt cache entry) when that cost >= kResumeLossReportTokens of prefill (serve/resume_loss.hpp).
    // Writes the finished request's tokens and MTP steps to options_.capture_dir (serve/capture.hpp); logs, never
    // throws.
    void capture(const GenerationRequest &req, const std::vector<CaptureStep> &steps, const std::string &reason);
    void report_resume_loss(const std::vector<int32_t> &prompt, int64_t start, int64_t live_common, int64_t live_n,
                            int64_t id);

    LmBackend &be_;
    const Tokenizer &tok_;
    const Options options_;
    PromptCache *cache_;
    const int64_t capacity_;
    const int32_t im_start_, im_end_, eot_;
    const int32_t think_end_;  // </think>
    const TokenMasker masker_;  // structured output's token masks over the logits row (built once, ~35 ms)
    const std::vector<int32_t> thinking_stop_;  // kThinkingStop, tokenized
    const std::vector<int32_t> nudge_[2];       // kThinkNudge1 / 2, tokenized
    std::vector<int32_t> seq_;  // the tokens in the backend (seq_.size() == be_.pos())
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> queue_;
    bool stop_ = false;
    EngineStats stats_;
    double idle_since_ = 0;  // when the last request finished (steady clock, s); under mu_
    int64_t running_id_ = 0;  // the id of the request the worker runs (while stats_.busy); under mu_
    std::thread thread_;
};

}  // namespace strix
