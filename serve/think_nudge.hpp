#pragma once

// The thinking nudge (a permanent engine feature for this model, 2026-10-01): when a turn's thinking goes in circles,
// the engine feeds one first-person sentence into the think block - never a </think>, the model decides whether to wrap
// up. Measured first: on terminal-bench captures a 1k window of >= 25% self-repeated 8-grams past 3k thinking tokens
// fired on 21 / 31 of the cobol rabbit hole's long turns and 5 / 55 of other tasks' long turns, never on opencode;
// resumed with the nudge, 7 / 12 runs wrapped up 5-40x sooner with the same kind of next step, 5 ignored it (as long as
// without).
//
// ThinkWatch follows one turn's reasoning tokens and says when a nudge is due:
//   nudge 1: past min_tokens with the last `window` tokens' self-copy rate >= rate;
//   nudge 2 (firmer): past max(2 x, + min_gap) the first nudge's position, with the rate condition again;
// No length backstop (removed after measuring it): its 3 fires
// (16.4k / 16.4k / 4.1k thinking tokens with ~1-5% self-copy) were ignored twice and once made the model end its turn
// inside the think block - an empty reply. Long thinking that isn't circling is not this mechanism's job.
// at most max_nudges. A due nudge waits for a paragraph break ("\n\n" in the text so far) up to boundary_wait tokens,
// so it never lands mid-sentence. self-copy: the n-gram ending at a token occurred earlier in this turn's thinking.

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace strix {

// What the engine feeds (model-specific: Qwen3.8-Flash-Next's thinking voice, first person, no </think>).
constexpr const char *kThinkNudge1 =
    "I've been reasoning about this for a long time and I keep going over the same ground. I should commit to the "
    "most promising approach now and act on it.\n\n";
// Nudge 1's alternative wording ("decide"): names the decision and the next action. In fork tests at the served
// trigger it ended thinking more often than kThinkNudge1, and wrote the answer inside the think block more often too.
constexpr const char *kThinkNudgeDecide =
    "I have enough information to act. I'll state my decision and the exact next command, run it, and let the output "
    "tell me if I was wrong.\n\n";
constexpr const char *kThinkNudge2 =
    "I'm still going around in circles. The task may not need this level of detail - I'll take the simplest approach "
    "that meets the stated requirements, act on it, and check the result.\n\n";

// Nudge 1's text for the config's think-nudge-wording: "commit" (kThinkNudge1) or "decide" (kThinkNudgeDecide).
// Throws on any other name.
const char *think_nudge1_text(const std::string &wording);

class ThinkWatch {
public:
    struct Policy {
        int64_t min_tokens = 3072;     // no rate trigger before this many thinking tokens
        int64_t window = 1024;         // tokens the self-copy rate is taken over
        double rate = 0.25;            // the rate that makes a nudge due
        int64_t min_gap = 4096;        // nudge 2: at least this many tokens after nudge 1 (and 2 x its position)
        int64_t boundary_wait = 256;   // a due nudge waits at most this long for a paragraph break
        int ngram = 8;
        int max_nudges = 2;
    };
    explicit ThinkWatch(Policy p);
    // One reasoning token (sampled, not fed by the engine) and its decoded text.
    void observe(int32_t id, const std::string &text);
    // 0: nothing to feed now; 1 / 2: feed nudge 1 / 2 now.
    int due() const;
    void fired();  // the engine fed the nudge due() named
    int64_t thinking_tokens() const { return n_; }
    double window_rate() const { return (double)window_hits_ / (double)p_.window; }
    int nudges() const { return nudges_; }

private:
    Policy p_;
    int64_t n_ = 0;
    std::vector<int32_t> last_;           // the previous ngram - 1 tokens (ring)
    size_t last_at_ = 0;
    std::unordered_set<uint64_t> seen_;   // hashes of this turn's n-grams
    std::vector<uint8_t> hits_;           // ring of the last `window` self-copy flags
    int64_t window_hits_ = 0;
    int nudges_ = 0;
    int64_t first_at_ = -1;               // thinking tokens at nudge 1
    int64_t armed_at_ = -1;               // thinking tokens when the pending nudge became due (-1: none due)
    bool para_ = false;                   // the text so far ends a paragraph ("\n\n")
    char tail_ = 0;                       // the text's last character
    bool condition() const;
};

}  // namespace strix
