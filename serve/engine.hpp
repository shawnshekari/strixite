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
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
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
    // Vision tower (Qwen4ExpSession::vision_stats): images = tower forwards (one per image), the patch / merged-token
    // counts they ran, and the time they took (the journal line's span: upload + tower + scatter).
    int64_t vision_images = 0, vision_patches = 0, vision_merged_tokens = 0;
    double vision_tower_seconds = 0, vision_tower_max_seconds = 0;
};

// One tower input for an image prefill (vision): the rows the tower runs over (FP32,
// the producer is serve/vision_preprocess) and the grid. Deliberately model-agnostic types (floats + ints) -
// the engine passes these to the backend, never interprets them.
struct VisionPrefill {
    std::vector<float> pixel_values;
    int64_t grid_t = 1, grid_h = 0, grid_w = 0;
    // Set when pixel_values were left empty because this server decoded the same image bytes before
    // (vision::DecodeMemo - usually the image is behind the resume point and the pixels are never read): builds them
    // from the kept bytes. The engine calls it only for a block its prefill reaches (a resume that came up short of
    // the image), on the worker, and hands the result to the backend as is.
    std::function<std::vector<float>()> decode;
};

// One image at its row span in the prompt (stream order). row_start is the first image_token_id row, rows the
// merged count - the engine aligns prefill chunk boundaries to these spans and hands each image to exactly
// one forward (Phase B's one-image-per-forward contract).
struct PrefillImage {
    VisionPrefill prefill;
    int64_t row_start = -1, rows = 0;
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
    // Vision (image prompts). max_positions: the model's position ceiling
    // (the walk refuses past it; 0 = unknown, the check skips). has_vision: image requests may be built at all
    // (the HTTP layer's gate and /health). forward_image_prefill: one prefill forward carrying one image (the
    // block's rows, pos3 the walk's triples for these rows) with the logits as forward_rows takes them. The
    // default refuses naming the missing tower - a text-only backend never runs one.
    virtual bool has_vision() const { return false; }
    virtual int64_t max_positions() const { return 0; }
    virtual LogitRows forward_image_prefill(const std::vector<int32_t> &ids, const PrefillImage &image,
                                            const std::vector<int64_t> &pos3, int64_t n_logits, bool cands,
                                            int64_t n_valid, const LogitMasks *masks = nullptr) {
        (void)ids, (void)image, (void)pos3, (void)n_logits, (void)cands, (void)n_valid, (void)masks;
        STRIX_FAIL("the backend ", describe(), " has no vision tower - the image prompt is refused");
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
    // ------------------------------------------------ several conversations (feat/multi-sequence)
    // A backend holds n_seqs() conversations ("seqs"; not the snapshot slots below), each its own state at its own
    // position; the calls above are seq 0's. The batched MTP step: every decoding seq drafts its own chain
    // (forward_mtp_top2(seq, ..), chains stay sequential per seq), then one verify_batch() verifies all their runs
    // in one forward (the weights read once for every run), then each run is resolved on its own - keep_verify(seq,
    // rows, carry) or drop_verify(seq) - before anything else runs. The defaults are one seq (0) on the calls above,
    // so a single-conversation backend serves the same step.
    virtual int n_seqs() const { return 1; }
    virtual int64_t pos(int seq) const {
        check_seq(seq, "pos");
        return pos();
    }
    // A plain forward of seq's next ids (a prefill chunk; one seq a forward).
    virtual LogitRows forward_rows(int seq, const std::vector<int32_t> &ids, int64_t n_logits, bool cands,
                                   int64_t n_valid, const LogitMasks *masks = nullptr) {
        check_seq(seq, "forward_rows");
        return forward_rows(ids, n_logits, cands, n_valid, masks);
    }
    virtual Top2 forward_mtp_top2(int seq, int32_t token_id, int64_t step) {
        check_seq(seq, "forward_mtp_top2");
        return forward_mtp_top2(token_id, step);
    }
    virtual void prefetch_ple(int seq, const std::vector<int32_t> &ids, int64_t first) {
        check_seq(seq, "prefetch_ple");
        prefetch_ple(ids, first);
    }
    // One run of a batched verify: seq's ids (the token before the drafts, then the drafts), every row's logits
    // back; masks: null, or one mask per row (masks->rows == ids.size()).
    struct VerifyRun {
        int seq = 0;
        std::vector<int32_t> ids;
        const LogitMasks *masks = nullptr;
    };
    // The runs' logits rows, one LogitRows per run in the runs' order. Distinct seqs; the backend's row budget
    // (max_verify_rows()) holds all of them - the caller packs.
    virtual std::vector<LogitRows> verify_batch(const std::vector<VerifyRun> &runs, bool cands, int64_t n_valid) {
        STRIX_CHECK(runs.size() == 1, "LmBackend::verify_batch: ", runs.size(), " runs on a one-conversation backend (",
                    describe(), "), expected 1");
        check_seq(runs[0].seq, "verify_batch");
        verify_ids_ = runs[0].ids;
        return {forward_verify_rows(runs[0].ids, (int64_t)runs[0].ids.size(), cands, n_valid, runs[0].masks)};
    }
    virtual int64_t max_verify_rows() const { return 16; }  // rows (and logits rows) of one verify_batch forward
    // Can seq's next verify share a verify_batch with another conversation's? More than one run means scalar
    // positions only (Qwen4ExpSession::forward_batch's contract), so a conversation whose state carries mRoPE
    // position state - one that has run an image, or resumed one that did - runs alone, and the arbiter packs
    // accordingly instead of letting the batch throw and take every run in it down (issue #29). A backend
    // without that state batches everything.
    virtual bool batchable(int seq) const { return true; }
    // Resolve seq's run of the pending verify_batch: keep its first `rows` rows (all of them: the whole verify).
    // carry: the token seq's next forward starts with (an MTP rejection's corrected sample; -1: none) - a backend
    // may check it there.
    virtual void keep_verify(int seq, int64_t rows, int32_t carry) {
        check_seq(seq, "keep_verify");
        (void)carry;
        STRIX_CHECK(rows >= 1 && rows <= (int64_t)verify_ids_.size(), "LmBackend::keep_verify: keep ", rows,
                    " rows of a ", verify_ids_.size(), "-row verify, expected 1..", verify_ids_.size());
        if (rows == (int64_t)verify_ids_.size()) keep_verify();
        else keep_verify_prefix(verify_ids_, rows);
        verify_ids_.clear();
    }
    virtual void drop_verify(int seq) {
        check_seq(seq, "drop_verify");
        drop_verify();
        verify_ids_.clear();
    }
    // A conversation's own state: reset it alone (the others untouched), its own snapshot slots, its prompt-cache
    // export / import. failed(): an earlier call failed midway - every seq's state may be half-written, so only a
    // reset() of the whole backend continues.
    virtual void reset(int seq) {
        check_seq(seq, "reset");
        reset();
    }
    virtual bool failed() const { return false; }
    virtual void save_snapshot(int seq, int slot) {
        check_seq(seq, "save_snapshot");
        save_snapshot(slot);
    }
    virtual int64_t snapshot_pos(int seq, int slot) const {
        check_seq(seq, "snapshot_pos");
        return snapshot_pos(slot);
    }
    virtual bool can_restore_snapshot(int seq, int slot) const {
        check_seq(seq, "can_restore_snapshot");
        return can_restore_snapshot(slot);
    }
    virtual void restore_snapshot(int seq, int slot) {
        check_seq(seq, "restore_snapshot");
        restore_snapshot(slot);
    }
    virtual void export_snapshot(int seq, int slot, HostBuffer &out, int64_t from = 0) {
        check_seq(seq, "export_snapshot");
        export_snapshot(slot, out, from);
    }
    virtual uint64_t snapshot_bytes(int seq, int slot, int64_t from = 0) const {
        check_seq(seq, "snapshot_bytes");
        return snapshot_bytes(slot, from);
    }
    virtual void import_state(int seq, const HostBuffer &state, int64_t n, int64_t from = 0) {
        check_seq(seq, "import_state");
        import_state(state, n, from);
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
    // The bytes export_snapshot(slot, out, from) will size out to (the prompt cache makes room for them first).
    virtual uint64_t snapshot_bytes(int slot, int64_t from = 0) const = 0;
    virtual void import_state(const HostBuffer &state, int64_t n, int64_t from = 0) = 0;
    // What a saved state fits: model, weights, layout, activation dtype (PromptCache refuses others).
    virtual std::string state_fingerprint() const = 0;
    virtual std::string describe() const = 0;  // for /health and logs

protected:
    void check_seq(int seq, const char *what) const {
        STRIX_CHECK(seq >= 0 && seq < n_seqs(), "LmBackend::", what, ": seq ", seq, ", expected 0..", n_seqs() - 1,
                    " (", describe(), ")");
    }

private:
    std::vector<int32_t> verify_ids_;  // the one-seq default's pending verify_batch run (keep_verify's prefix)
};

struct GenerationRequest {
    std::vector<int32_t> prompt;
    // Image prompt (vision): per prompt row the (t, h, w) section positions of the mRoPE walk
    // (serve/vision_request.cpp, the get_rope_index rule) and the images with their spans, stream
    // order. Both empty: a text prompt - today's path, bit-identical, zero cost. The
    // engine refuses them on a backend without a tower (submit-time check) and re-checks spans vs ids rows.
    std::vector<int64_t> image_pos3;
    std::vector<PrefillImage> prefill_images;
    // The stream resume / prompt-cache matching runs on: prompt with each image's pad run replaced by its
    // content-derived pseudo ids (serve/vision_request.cpp). Empty = text prompt (prompt is used as is).
    std::vector<int32_t> cache_key;
    int64_t max_tokens = 0;  // >= 1; prompt + max_tokens <= capacity (the API layer clamps)
    SamplingParams sampling;
    OutputParser::Config parser;  // parser.tools must outlive the request
    // Reasoning tokens allowed before the engine closes the think block itself (feeding kThinkingStop); -1 =
    // no budget. The API layer sets it (an explicit budget, or the answer room).
    int64_t thinking_budget = -1;
    // Past the budget, at most this many more reasoning tokens while the engine waits for the end of a sentence
    // (ends_sentence) before it feeds the stop; 0 = the stop right at the budget. The server's thinking cap sets it
    // (config thinking-cap-grace); a request's own budget leaves it 0.
    int64_t thinking_budget_grace = 0;
    // The prompt's last message is the user's (not tool results): the prefix up to that message's start goes to
    // the disk cache as a checkpoint (PromptCache::Kind::Checkpoint), kept when later turns extend it.
    bool user_turn = false;
    // One message and no tools (a batch client's one-off prompt, nothing a later turn extends): its prompt cache
    // entries stay in RAM and never reach the disk (PromptCache "RAM-only entries").
    bool one_shot = false;
    // Structured output (response_format / tool_choice): the compiled grammar
    // the answer must follow, or null. Shared by requests with the same schema; several slots' workers drive one
    // concurrently (GrammarAutomaton is internally synchronized - serve/grammar.hpp, review item 2).
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

// The reasoning text so far (its tail - the last few hundred bytes are enough) ends a sentence: it ends in a newline,
// or in . ! ? (closing quotes / brackets / * after it allowed) right after a letter, `)` or a closing quote, outside
// inline code (an even number of backticks since the last newline). Digits before the stop don't count ("version 3."
// may go on "3.14"), nor does a dot inside a name the next token continues ("engine." + "cpp" - most such names
// tokenize as "engine" + ".cpp", so the dot never ends the tail).
bool ends_sentence(std::string_view tail);

// What the engine feeds ahead of a tool call begun inside the think block (Engine::Options::think_close_on_call):
// the template's think close; the newline before it is the model's own.
constexpr const char *kThinkCloseBeforeCall = "</think>\n\n";

constexpr int64_t kMtpMaxDraft = 4;  // rejection-point buckets: j = 0, 1, 2 and 3 or more accepted drafts

struct GenerationResult {
    std::string finish_reason;  // stop | length | tool_calls | cancelled | error
    std::string error;          // finish_reason == error
    int64_t prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0, reasoning_tokens = 0;
    int64_t mtp_drafted_tokens = 0, mtp_accepted_tokens = 0, mtp_rollbacks = 0;
    // How this request verifies ran (its decode window in GpuArbiter): shared with another slot run, or ran
    // alone because the conversation carries mRoPE position state and cannot share a forward.
    int64_t verify_shared_runs = 0, verify_solo_mrope = 0;
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
    int64_t think_end_blocked = 0;     // rows where the guard took out an end of turn that was the top token
    int64_t think_closed_for_call = 0;  // think blocks closed ahead of a tool call begun inside them (think_close_on_call)
};

// Where the prefill stands (GenerationSink::on_prefill): `processed` of the prompt's `total` tokens are in the KV
// state, `cached` of them reused from the live session or the prompt cache rather than prefilled; `ms` since the
// prefill started (once the resume point was found and any prompt cache state loaded).
struct PrefillProgress {
    int64_t total = 0, cached = 0, processed = 0;
    double ms = 0;
};

// The request's side of the loop, called on the engine thread. Implementations must not throw: the engine calls
// on_done exactly once per request (the worker's last-line guard does not re-deliver after a delivery was
// attempted), so a throwing sink loses that request's response and takes nothing else down with it.
class GenerationSink {
public:
    virtual ~GenerationSink() = default;
    virtual void on_start() = 0;                               // left the queue
    virtual void on_progress(int64_t done, int64_t total) { (void)done, (void)total; }  // after each prefill chunk
    // The same with the cached tokens and the time, once more at the resume point: the engine calls this one.
    virtual void on_prefill(const PrefillProgress &p) { on_progress(p.processed, p.total); }
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
    int64_t think_end_blocked = 0;  // GenerationResult::think_end_blocked, summed
    int64_t think_closed_for_call = 0;  // GenerationResult::think_closed_for_call, summed
    int64_t cache_hits = 0, cache_misses = 0;
    double prefill_seconds = 0, decode_seconds = 0;
    double last_prefill_tps = 0, last_decode_tps = 0;
    int64_t queued = 0, busy = 0;  // busy: requests running (0..slots)
    int64_t slots = 1;             // conversations served at once (the backend's n_seqs)
    // Per slot: the request it runs (0: idle), the tokens its state holds and its snapshot's position (-1: none).
    std::vector<int64_t> slot_request, slot_live_tokens, slot_snapshot_pos;
    int64_t verify_batches = 0, verify_batch_runs = 0;  // batched MTP verifies (GpuArbiter) and their runs
    // Of those, the runs that went alone on a conversation that can never share a forward (mRoPE position
    // state, issue #29): the batching the arbiter refuses, stated instead of inferred from the other two.
    int64_t verify_solo_mrope = 0;
    int64_t decoding_slots = 0;  // slots in their decode loop (GpuArbiter::decoding_count; 0 with one slot)
    // Seconds since the last request finished (or the engine started); 0 while one runs or waits. `busy` alone reads
    // 1 through a steady agent session, so a batch client sampling it saw a "stuck" flag; this lets it
    // wait for a real gap (e.g. idle_seconds >= 2).
    double idle_seconds = 0;
    int64_t live_tokens = 0;  // the sum of slot_live_tokens: every slot's state, not the last one to finish
    // Prompts resumed from the prompt cache (either tier; ram_hits: of those, from its RAM tier) / states put in its
    // RAM tier (the disk gets them later, if at all).
    int64_t disk_hits = 0, disk_tokens = 0, ram_saves = 0, ram_hits = 0;
    double export_seconds = 0;
};

class GpuArbiter;  // serve/seq_backend.hpp (it builds on LmBackend above)
class SeqBackend;

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
        std::string think_nudge_wording = "commit";  // nudge 1's text (config think-nudge-wording; think_nudge1_text)
        // The think-block guard (config think-end-guard): <|im_end|> / <|endoftext|> are never sampled while the think
        // block or a tool call is open (ban_tokens). Off only for like-for-like comparisons with engines that don't have it.
        bool think_end_guard = true;
        // OutputParser::Config::literal_tags (config literal-tags): a tag not right after a newline is text.
        bool literal_tags = true;
        // Called once, on the worker thread, after a request failed with GpuFatalError (the GPU context is lost, every
        // later request would fail): that request and the queued ones have their error responses, new ones are
        // refused. strix_server stops serving and exits non-zero, for its supervisor to restart it. Empty: the
        // engine only refuses.
        std::function<void(const std::string &)> on_fatal;
        // Non-empty: what startup turned down (serve/memory_plan.hpp, low-memory adapt). Every request's log block
        // carries it as a "performance degraded" WARNING row, and /health reports it.
        std::string degraded;
        // Close the think block before a tool call begun inside it (config think-close-on-call): a
        // <tool_call> sampled right after a newline while the think block is open is reasoning text to the parser,
        // so no call runs; the model then wrote call after call into its thinking (52 in one opencode turn) while the
        // guard kept the turn from ending, until the client cancelled. With this on, the engine feeds
        // kThinkCloseBeforeCall ahead of that <tool_call> and the call is parsed as one. Off only for like-for-like
        // comparisons. (Last, so the positional Options{...} in the tests keep their meaning.)
        bool think_close_on_call = true;
        // Prefill chunk while another slot is decoding (config chunk-shared; several slots only; 0 = the backend's
        // max chunk always). A chunk holds the GPU: at 16,384 tokens the other slots' decode stopped ~10 s a chunk.
        // Read per chunk, so a prefill alone runs full chunks again as soon as the others finish.
        int64_t shared_chunk = 2048;
    };

    Engine(LmBackend &backend, const Tokenizer &tok, const Options &options);
    Engine(LmBackend &backend, const Tokenizer &tok, PromptCache *cache = nullptr, float mtp_margin = 1.5f);
    ~Engine();  // finishes the running request, fails the queued ones, joins the worker
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    // The backend's n_seqs() slots each serve one conversation at a time: a worker per slot, requests in arrival
    // order (FIFO) to an idle slot - the one the prompt continues, else an empty one, else the least recently used
    // (dispatch_locked); an image prompt waits for slot 0
    // (image positions are slot 0's); a prompt that continues a busy slot's running request waits for that slot. Several slots share the GPU through a GpuArbiter (serve/seq_backend.hpp),
    // which batches their MTP verifies.

    void submit(GenerationRequest req, std::shared_ptr<GenerationSink> sink);
    EngineStats stats() const;
    BackendStats backend_stats() const { return be_.backend_stats(); }  // thread-safe per the backend's contract
    int64_t capacity() const { return capacity_; }
    const LmBackend &backend() const { return be_; }
    const PromptCache *prompt_cache() const { return cache_; }
    float mtp_margin() const { return options_.mtp_margin; }
    // The GpuFatalError that stopped the engine (empty: none); submit refuses every request after it.
    std::string fatal_error() const;
    const std::string &degraded() const { return options_.degraded; }

private:
    struct Job {
        GenerationRequest req;
        std::shared_ptr<GenerationSink> sink;
        double enqueued;
        std::string slot_why;  // why dispatch picked its slot (the block's "slot" row; several slots only)
        // Set right before on_done is called (run()'s finish, or the worker's last-line guard): the guard
        // delivers the error result only when no delivery has happened, so a request gets it exactly once.
        bool result_sent = false;
    };
    // One conversation slot: its backend view (the backend itself with one slot, else a SeqBackend), the tokens its
    // state holds, its worker and the request it was given.
    struct Slot {
        int index = 0;
        LmBackend *be = nullptr;
        std::unique_ptr<SeqBackend> proxy;  // several slots only
        std::vector<int32_t> seq;           // the tokens in the slot's state (seq.size() == be->pos())
        std::thread thread;
        std::unique_ptr<Job> job;     // assigned, not started; under mu_
        bool busy = false;            // a request assigned or running; under mu_
        int64_t running_id = 0;       // under mu_
        std::vector<int32_t> running_key;  // the running request's prompt (its cache key); empty when idle; under mu_
        double last_used = 0;         // when it was last given a request (steady clock, s); under mu_
        int64_t turn_pos = -1;        // its turn snapshot's position after its last request (-1: none); under mu_
        std::string fatal_pending;    // set by run()'s handler (the slot's thread only)
        bool needs_reset = false;     // the worker's guard saw an error escape the request path: reset before the
                                      // next job (the slot's thread only) - an unclean state must not carry over
    };
    void worker(Slot &sl);
    void run(Job &job, Slot &sl);
    // Marks the job's result sent and delivers it, logging (never propagating) a throwing sink (GenerationSink's
    // no-throw contract). The one path to on_done outside run()'s finish.
    static void deliver(Job &job, const GenerationResult &r);
    void dispatch_locked();  // queued requests to idle slots (FIFO; continued conversation, else empty, else LRU)
    // Sets the backend up to prefill from the returned position: the longest reusable prefix among the live state,
    // the two snapshot slots and the prompt cache (restoring / importing / resetting as needed). from_disk: from the
    // prompt cache (either tier); from_ram: from its RAM tier.
    int64_t resume_point(Slot &sl, const std::vector<int32_t> &prompt, int64_t id, bool &from_disk, bool &from_ram);
    // Logs where the prompt left the state it shared the most with (the live session before resume_point, or a
    // prompt cache entry) when that cost >= kResumeLossReportTokens of prefill (serve/resume_loss.hpp).
    // Writes the finished request's tokens and MTP steps to options_.capture_dir (serve/capture.hpp); logs, never
    // throws.
    void capture(Slot &sl, const GenerationRequest &req, const std::vector<CaptureStep> &steps, const std::string &reason);
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
    const std::vector<int32_t> think_close_;    // kThinkCloseBeforeCall, tokenized
    const int32_t call_begin_;                  // <tool_call>
    std::unique_ptr<GpuArbiter> arbiter_;  // several slots only
    std::vector<std::unique_ptr<Slot>> slots_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> queue_;
    bool stop_ = false;
    std::string fatal_;  // the GpuFatalError's text once one was thrown; under mu_
    EngineStats stats_;
    double idle_since_ = 0;  // when the last request finished (steady clock, s); under mu_
    int64_t held_logged_ = 0;  // the request whose wait for its conversation's busy slot was logged; under mu_
};

}  // namespace strix
