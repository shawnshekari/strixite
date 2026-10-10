#pragma once

// Qwen3.8-Flash-Next's tokenizer, hand-rolled, loaded from the checkpoint's
// tokenizer.json and checked against it at load: byte-level BPE (248044 tokens, 247587 merges) plus 33 added
// tokens. Encoding, as the reference tokenizer does it:
//   1. added tokens are split out of the raw text first (leftmost-longest; none of this model's strip or
//      normalize), so "<|im_start|>" in the text is that one token;
//   2. each piece between them is NFC-normalized, then split by the pre-tokenizer pattern
//        (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N} | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//        | \s*[\r\n]+ | \s+(?!\S) | \s+
//      (leftmost-first alternation with backtracking, implemented by hand in pretokenize());
//   3. each split's UTF-8 bytes are merged pairwise by merge rank (lowest first, leftmost on a tie).
// The vocabulary's byte-level strings (GPT-2's byte <-> printable mapping) are turned back into raw bytes at load,
// so BPE works on bytes. Decoding a token gives its raw bytes (a token can end inside a UTF-8 sequence).

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strix {

class Tokenizer {
public:
    explicit Tokenizer(const std::string &tokenizer_json);

    std::vector<int32_t> encode(std::string_view text) const;  // text must be valid UTF-8 (throws otherwise)
    // The same, but an added token counts only where it lies wholly outside plain_spans (byte ranges [begin, end),
    // sorted, not overlapping, inside text): there its text is encoded as ordinary text. render_chat reports the
    // spans of what users, tools and tool schemas wrote, so a file holding "<|im_end|>" stays text (2026-10-09).
    std::vector<int32_t> encode(std::string_view text, const std::vector<std::pair<size_t, size_t>> &plain_spans) const;
    const std::string &token_bytes(int32_t id) const;          // raw bytes; throws on an unknown id
    std::string decode(const std::vector<int32_t> &ids) const;  // concatenated bytes (may be invalid UTF-8 at a cut)
    int32_t id_of(std::string_view token_text) const;          // an added token's id or a vocab token (byte-level
                                                               // form); throws if absent
    int32_t size() const { return (int32_t)bytes_.size(); }     // vocab + added tokens
    int32_t byte_token(uint8_t b) const { return byte_id_[b]; }  // the single-byte token of b
    bool is_added(int32_t id) const { return id >= n_vocab_; }

    // The pre-tokenizer's split of NFC-normalized code points into [begin, end) spans (exposed for tests).
    static std::vector<std::pair<size_t, size_t>> pretokenize(const std::vector<uint32_t> &cps);

private:
    int32_t n_vocab_ = 0;
    std::vector<std::string> bytes_;                    // id -> raw bytes
    std::unordered_map<std::string, int32_t> by_bytes_;  // raw bytes -> vocab id (not added tokens)
    std::unordered_map<std::string, int32_t> by_text_;   // byte-level text / added content -> id
    std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges_;  // (left << 32 | right) -> (rank, merged id)
    std::vector<std::pair<std::string, int32_t>> added_;  // content, id; longest first
    int32_t byte_id_[256];
    bool added_first_[256] = {};  // bytes an added token starts with

    void bpe(std::string_view piece, std::vector<int32_t> &out) const;
    void encode_plain(std::string_view text, std::vector<int32_t> &out) const;
};

}  // namespace strix
