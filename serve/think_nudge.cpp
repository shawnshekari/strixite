#include "serve/think_nudge.hpp"

#include "common/check.hpp"

#include <algorithm>

namespace strix {

const char *think_nudge1_text(const std::string &wording) {
    if (wording == "commit") return kThinkNudge1;
    if (wording == "decide") return kThinkNudgeDecide;
    STRIX_CHECK(false, "think_nudge1_text: think-nudge-wording '", wording, "' is not one of: commit, decide");
    return nullptr;
}

ThinkWatch::ThinkWatch(Policy p) : p_(p) {
    STRIX_CHECK(p_.min_tokens >= 0 && p_.window >= 16 && p_.rate > 0 && p_.rate <= 1 &&
                    p_.min_gap >= 0 && p_.boundary_wait >= 0 && p_.ngram >= 2 && p_.ngram <= 32 && p_.max_nudges >= 0 &&
                    p_.max_nudges <= 2,
                "ThinkWatch: policy min_tokens ", p_.min_tokens, ", window ", p_.window, ", rate ", p_.rate, ", min_gap ", p_.min_gap, ", boundary_wait ", p_.boundary_wait, ", ngram ", p_.ngram,
                ", max_nudges ", p_.max_nudges, " (expected window >= 16, 0 < rate <= 1, ngram 2..32, max_nudges 0..2)");
    last_.assign((size_t)(p_.ngram - 1), -1);
    hits_.assign((size_t)p_.window, 0);
}

bool ThinkWatch::condition() const {
    const bool repeating = n_ >= p_.min_tokens && n_ >= p_.window && window_rate() >= p_.rate;
    if (nudges_ == 0) return repeating;
    const int64_t after = std::max(2 * first_at_, first_at_ + p_.min_gap);
    return n_ >= after && repeating;
}

void ThinkWatch::observe(int32_t id, const std::string &text) {
    // The n-gram ending here: the previous ngram - 1 tokens (oldest first) and this one, hashed (FNV-1a over ids).
    bool full = true;
    uint64_t h = 1469598103934665603ull;
    for (size_t k = 0; k < last_.size(); ++k) {
        const int32_t t = last_[(last_at_ + k) % last_.size()];
        if (t < 0) full = false;
        h = (h ^ (uint32_t)t) * 1099511628211ull;
    }
    h = (h ^ (uint32_t)id) * 1099511628211ull;
    last_[last_at_] = id;
    last_at_ = (last_at_ + 1) % last_.size();
    const uint8_t hit = full && !seen_.insert(h).second ? 1 : 0;
    uint8_t &slot = hits_[(size_t)(n_ % p_.window)];
    window_hits_ += (int64_t)hit - (int64_t)slot;
    slot = hit;
    ++n_;
    // Paragraph break: "\n\n" inside this token's text, or across it and the previous one.
    for (char c : text) {
        para_ = c == '\n' && tail_ == '\n';
        tail_ = c;
    }
    if (!text.empty() && text.back() != '\n') para_ = false;
    if (armed_at_ < 0 && nudges_ < p_.max_nudges && condition()) armed_at_ = n_;
}

int ThinkWatch::due() const {
    if (armed_at_ < 0) return 0;
    return para_ || n_ - armed_at_ >= p_.boundary_wait ? nudges_ + 1 : 0;
}

void ThinkWatch::fired() {
    STRIX_CHECK(armed_at_ >= 0, "ThinkWatch::fired: no nudge was due");
    if (nudges_ == 0) first_at_ = n_;
    ++nudges_;
    armed_at_ = -1;
    para_ = true, tail_ = '\n';  // the nudge text ends with a paragraph break
}

}  // namespace strix
