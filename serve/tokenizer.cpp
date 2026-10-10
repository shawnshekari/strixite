#include "serve/tokenizer.hpp"

#include "common/check.hpp"
#include "serve/json.hpp"
#include "serve/unicode.hpp"

#include <algorithm>
#include <fstream>
#include <queue>
#include <sstream>

namespace strix {

namespace {

// The pre-tokenizer pattern this implementation hand-codes; a tokenizer.json with any other is refused.
constexpr const char *kPattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|"
    "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// GPT-2's byte-level alphabet: printable bytes map to themselves, the other 68 to U+0100.. in byte order.
std::vector<uint32_t> byte_to_cp() {
    std::vector<uint32_t> m(256);
    uint32_t next = 256;
    for (uint32_t b = 0; b < 256; ++b) {
        const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        m[b] = printable ? b : next++;
    }
    return m;
}

std::string read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    STRIX_CHECK(f.good(), "Tokenizer: can't open '", path, "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    STRIX_CHECK(!f.bad(), "Tokenizer: read of '", path, "' failed");
    return ss.str();
}

const json::Value &need(const json::Value &v, const char *key, const std::string &where) {
    const json::Value *x = v.find(key);
    STRIX_CHECK(x != nullptr, "Tokenizer: '", where, "' has no '", key, "'");
    return *x;
}

void expect_str(const json::Value &v, const char *key, const std::string &want, const std::string &where) {
    const std::string &got = need(v, key, where).as_string(where + "." + key);
    STRIX_CHECK(got == want, "Tokenizer: ", where, ".", key, " is '", got, "', this tokenizer implements '", want, "'");
}

void expect_bool(const json::Value &v, const char *key, bool want, const std::string &where) {
    const bool got = need(v, key, where).as_bool(where + "." + key);
    STRIX_CHECK(got == want, "Tokenizer: ", where, ".", key, " is ", got, ", this tokenizer implements ", want);
}

void check_byte_level(const json::Value &v, const std::string &where) {
    expect_str(v, "type", "ByteLevel", where);
    expect_bool(v, "add_prefix_space", false, where);
    expect_bool(v, "use_regex", false, where);
}

bool apostrophe_s(uint32_t c) { return c == 's' || c == 'S' || c == 0x17F; }  // (?i) also folds U+017F LONG S to s
bool ci(uint32_t c, char lower) { return c == (uint32_t)lower || c == (uint32_t)(lower - 32); }

}  // namespace

Tokenizer::Tokenizer(const std::string &path) {
    const std::string text = read_file(path);
    const json::Value root = json::Value::parse(text);
    const std::string p = "'" + path + "'";
    // Only what this implementation does is accepted.
    const json::Value &model = need(root, "model", p);
    expect_str(model, "type", "BPE", p + ".model");
    STRIX_CHECK(need(model, "unk_token", p).is_null(), "Tokenizer: ", p, ".model.unk_token must be null");
    expect_bool(model, "byte_fallback", false, p + ".model");
    expect_bool(model, "ignore_merges", false, p + ".model");
    expect_str(model, "continuing_subword_prefix", "", p + ".model");
    expect_str(model, "end_of_word_suffix", "", p + ".model");
    STRIX_CHECK(need(model, "dropout", p).is_null(), "Tokenizer: ", p, ".model.dropout must be null");
    expect_str(need(root, "normalizer", p), "type", "NFC", p + ".normalizer");
    const json::Value &pre = need(root, "pre_tokenizer", p);
    expect_str(pre, "type", "Sequence", p + ".pre_tokenizer");
    const auto &pres = need(pre, "pretokenizers", p).as_array(p + ".pre_tokenizer.pretokenizers");
    STRIX_CHECK(pres.size() == 2, "Tokenizer: ", p, ": ", pres.size(), " pre-tokenizers, expected Split + ByteLevel");
    expect_str(pres[0], "type", "Split", p + ".pretokenizers[0]");
    expect_str(need(pres[0], "pattern", p), "Regex", kPattern, p + ".pretokenizers[0].pattern");
    expect_str(pres[0], "behavior", "Isolated", p + ".pretokenizers[0]");
    expect_bool(pres[0], "invert", false, p + ".pretokenizers[0]");
    check_byte_level(pres[1], p + ".pretokenizers[1]");
    check_byte_level(need(root, "decoder", p), p + ".decoder");

    // Byte-level text -> raw bytes.
    const std::vector<uint32_t> b2c = byte_to_cp();
    std::unordered_map<uint32_t, uint8_t> c2b;
    for (uint32_t b = 0; b < 256; ++b) c2b[b2c[b]] = (uint8_t)b;
    const auto to_bytes = [&](const std::string &tok) {
        std::string out;
        for (uint32_t c : unicode::decode_utf8(tok, "Tokenizer: vocab token")) {
            const auto it = c2b.find(c);
            STRIX_CHECK(it != c2b.end(), "Tokenizer: vocab token '", tok, "' has U+", std::hex, c, std::dec,
                        ", outside the byte-level alphabet");
            out += (char)it->second;
        }
        return out;
    };

    const auto &vocab = need(model, "vocab", p).as_object(p + ".model.vocab");
    n_vocab_ = (int32_t)vocab.size();
    bytes_.assign((size_t)n_vocab_, std::string());
    std::vector<bool> seen((size_t)n_vocab_, false);
    for (const auto &[tok, idv] : vocab) {
        const int64_t id = idv.as_int(p + ".model.vocab['" + tok + "']");
        STRIX_CHECK(id >= 0 && id < n_vocab_ && !seen[(size_t)id], "Tokenizer: vocab id ", id, " of '", tok,
                    "' out of [0, ", n_vocab_, ") or repeated");
        seen[(size_t)id] = true;
        bytes_[(size_t)id] = to_bytes(tok);
        STRIX_CHECK(by_bytes_.emplace(bytes_[(size_t)id], (int32_t)id).second, "Tokenizer: two vocab tokens decode to the same bytes ('", tok, "')");
        by_text_.emplace(tok, (int32_t)id);
    }
    for (int b = 0; b < 256; ++b) {
        const auto it = by_bytes_.find(std::string(1, (char)b));
        STRIX_CHECK(it != by_bytes_.end(), "Tokenizer: byte 0x", std::hex, b, std::dec, " has no token (byte-level BPE needs all 256)");
        byte_id_[b] = it->second;
    }

    const auto &merges = need(model, "merges", p).as_array(p + ".model.merges");
    merges_.reserve(merges.size() * 2);
    for (size_t r = 0; r < merges.size(); ++r) {
        std::string a, b;
        if (merges[r].is_string()) {
            const std::string &m = merges[r].as_string("merge");
            const size_t sp = m.find(' ');
            STRIX_CHECK(sp != std::string::npos && m.find(' ', sp + 1) == std::string::npos, "Tokenizer: merge ", r,
                        " '", m, "' isn't 'left right'");
            a = m.substr(0, sp), b = m.substr(sp + 1);
        } else {
            const auto &pair = merges[r].as_array("merge");
            STRIX_CHECK(pair.size() == 2, "Tokenizer: merge ", r, " has ", pair.size(), " parts, expected 2");
            a = pair[0].as_string("merge"), b = pair[1].as_string("merge");
        }
        const auto ia = by_text_.find(a), ib = by_text_.find(b), iab = by_text_.find(a + b);
        STRIX_CHECK(ia != by_text_.end() && ib != by_text_.end() && iab != by_text_.end(), "Tokenizer: merge ", r,
                    " ('", a, "' + '", b, "') refers to a token not in the vocab");
        merges_.emplace(((uint64_t)(uint32_t)ia->second << 32) | (uint32_t)ib->second,
                        std::pair{(int32_t)r, iab->second});  // a repeated pair keeps its first (lowest) rank
    }

    const auto &added = need(root, "added_tokens", p).as_array(p + ".added_tokens");
    for (size_t k = 0; k < added.size(); ++k) {
        const std::string w = p + ".added_tokens[" + std::to_string(k) + "]";
        const json::Value &a = added[k];
        const std::string &content = need(a, "content", w).as_string(w + ".content");
        const int64_t id = need(a, "id", w).as_int(w + ".id");
        for (const char *flag : {"single_word", "lstrip", "rstrip", "normalized"})
            expect_bool(a, flag, false, w + " ('" + content + "')");
        STRIX_CHECK(id == (int64_t)bytes_.size(), "Tokenizer: added token '", content, "' has id ", id, ", expected ",
                    bytes_.size(), " (added tokens follow the vocab in order)");
        STRIX_CHECK(!content.empty() && json::valid_utf8(content), "Tokenizer: added token ", id, " is empty or not UTF-8");
        bytes_.push_back(content);
        STRIX_CHECK(by_text_.emplace(content, (int32_t)id).second, "Tokenizer: added token '", content, "' repeats a vocab token");
        added_.emplace_back(content, (int32_t)id);
    }
    std::stable_sort(added_.begin(), added_.end(), [](const auto &x, const auto &y) { return x.first.size() > y.first.size(); });
    for (const auto &a : added_) added_first_[(unsigned char)a.first[0]] = true;
}

const std::string &Tokenizer::token_bytes(int32_t id) const {
    STRIX_CHECK(id >= 0 && id < size(), "Tokenizer::token_bytes: id ", id, " outside [0, ", size(), ")");
    return bytes_[(size_t)id];
}

std::string Tokenizer::decode(const std::vector<int32_t> &ids) const {
    std::string out;
    for (int32_t id : ids) out += token_bytes(id);
    return out;
}

int32_t Tokenizer::id_of(std::string_view token_text) const {
    const auto it = by_text_.find(std::string(token_text));
    STRIX_CHECK(it != by_text_.end(), "Tokenizer::id_of: no token '", token_text, "'");
    return it->second;
}

std::vector<std::pair<size_t, size_t>> Tokenizer::pretokenize(const std::vector<uint32_t> &c) {
    using unicode::char_class, unicode::is_white_space, unicode::kLetter, unicode::kMark, unicode::kNumber;
    const size_t n = c.size();
    const auto lm = [&](size_t k) { return (char_class(c[k]) & (kLetter | kMark)) != 0; };
    const auto nl = [&](size_t k) { return c[k] == '\r' || c[k] == '\n'; };
    // [^\s\p{L}\p{M}\p{N}]
    const auto punct = [&](size_t k) { return !is_white_space(c[k]) && char_class(c[k]) == 0; };
    std::vector<std::pair<size_t, size_t>> out;
    for (size_t i = 0; i < n;) {
        size_t end = i;
        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c[i] == '\'' && i + 1 < n) {
            const uint32_t a = c[i + 1], b = i + 2 < n ? c[i + 2] : 0;
            if (apostrophe_s(a) || ci(a, 't') || ci(a, 'm') || ci(a, 'd')) end = i + 2;
            else if ((ci(a, 'r') && ci(b, 'e')) || (ci(a, 'v') && ci(b, 'e')) || (ci(a, 'l') && ci(b, 'l'))) end = i + 3;
        }
        // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (end == i) {
            size_t j = i;
            const uint8_t k0 = char_class(c[i]);
            if (!nl(i) && !(k0 & (kLetter | kNumber)) && i + 1 < n && lm(i + 1)) j = i + 1;
            if (j < n && lm(j)) {
                while (j < n && lm(j)) ++j;
                end = j;
            }
        }
        // 3. \p{N}
        if (end == i && (char_class(c[i]) & kNumber)) end = i + 1;
        // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        if (end == i) {
            size_t j = i;
            if (c[i] == ' ' && i + 1 < n && punct(i + 1)) j = i + 1;
            if (punct(j)) {
                while (j < n && punct(j)) ++j;
                while (j < n && nl(j)) ++j;
                end = j;
            }
        }
        if (end == i && is_white_space(c[i])) {
            size_t w = i;
            while (w < n && is_white_space(c[w])) ++w;
            // 5. \s*[\r\n]+ - up to and including the last newline of the run.
            for (size_t k = w; k > i; --k)
                if (nl(k - 1)) { end = k; break; }
            // 6. \s+(?!\S) - the run, less its last character when a non-space follows.
            if (end == i) {
                if (w == n) end = w;
                else if (w - i >= 2) end = w - 1;
            }
            // 7. \s+
            if (end == i) end = w;
        }
        STRIX_CHECK(end > i, "Tokenizer::pretokenize: no alternative matched U+", std::hex, c[i], std::dec, " at ", i);
        out.emplace_back(i, end);
        i = end;
    }
    return out;
}

void Tokenizer::bpe(std::string_view piece, std::vector<int32_t> &out) const {
    const size_t n = piece.size();
    if (n == 1) {
        out.push_back(byte_id_[(unsigned char)piece[0]]);
        return;
    }
    struct Sym {
        int32_t id;
        int prev, next;
    };
    std::vector<Sym> s(n);
    for (size_t k = 0; k < n; ++k) s[k] = {byte_id_[(unsigned char)piece[k]], (int)k - 1, k + 1 < n ? (int)k + 1 : -1};
    struct Cand {
        int32_t rank;
        int pos;
        int32_t left, right, merged;
        bool operator>(const Cand &o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
    };
    std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> q;
    const auto consider = [&](int a) {
        if (a < 0 || s[(size_t)a].next < 0) return;
        const int b = s[(size_t)a].next;
        const auto it = merges_.find(((uint64_t)(uint32_t)s[(size_t)a].id << 32) | (uint32_t)s[(size_t)b].id);
        if (it != merges_.end()) q.push({it->second.first, a, s[(size_t)a].id, s[(size_t)b].id, it->second.second});
    };
    for (int k = 0; k + 1 < (int)n; ++k) consider(k);
    while (!q.empty()) {
        const Cand m = q.top();
        q.pop();
        Sym &a = s[(size_t)m.pos];
        if (a.id != m.left || a.next < 0 || s[(size_t)a.next].id != m.right) continue;  // stale
        const int b = a.next;
        a.id = m.merged;
        a.next = s[(size_t)b].next;
        if (a.next >= 0) s[(size_t)a.next].prev = m.pos;
        s[(size_t)b].id = -1;  // dead
        consider(a.prev);
        consider(m.pos);
    }
    for (int k = 0; k >= 0; k = s[(size_t)k].next) out.push_back(s[(size_t)k].id);
}

void Tokenizer::encode_plain(std::string_view text, std::vector<int32_t> &out) const {
    if (text.empty()) return;
    const std::vector<uint32_t> cps = unicode::nfc(unicode::decode_utf8(text, "Tokenizer::encode"));
    for (const auto &[b, e] : pretokenize(cps)) bpe(unicode::encode_utf8(cps, b, e), out);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const { return encode(text, {}); }

std::vector<int32_t> Tokenizer::encode(std::string_view text, const std::vector<std::pair<size_t, size_t>> &plain_spans) const {
    STRIX_CHECK(json::valid_utf8(text), "Tokenizer::encode: text (", text.size(), " bytes) is not valid UTF-8");
    for (size_t k = 0; k < plain_spans.size(); ++k) {
        const auto [b, e] = plain_spans[k];
        STRIX_CHECK(b <= e && e <= text.size() && (k == 0 || plain_spans[k - 1].second <= b), "Tokenizer::encode: plain span ",
                    k, " [", b, ", ", e, ") of ", plain_spans.size(), " - expected sorted, non-overlapping spans inside the ",
                    text.size(), "-byte text", k > 0 ? " (previous ends at " + std::to_string(plain_spans[k - 1].second) + ")" : "");
    }
    std::vector<int32_t> out;
    out.reserve(text.size() / 3 + 4);
    size_t plain = 0, span = 0;
    for (size_t i = 0; i < text.size();) {
        int32_t hit = -1;
        size_t len = 0;
        if (!added_first_[(unsigned char)text[i]]) {
            ++i;
            continue;
        }
        while (span < plain_spans.size() && plain_spans[span].second <= i) ++span;
        if (span < plain_spans.size() && plain_spans[span].first <= i) {  // inside a plain span: skip to its end
            i = plain_spans[span].second;
            continue;
        }
        const size_t room = span < plain_spans.size() ? plain_spans[span].first - i : text.size() - i;
        for (const auto &[content, id] : added_)  // longest first: the first hit is the longest here
            if (content.size() <= room && text.compare(i, content.size(), content) == 0) {
                hit = id, len = content.size();
                break;
            }
        if (hit < 0) {
            ++i;
            continue;
        }
        encode_plain(text.substr(plain, i - plain), out);
        out.push_back(hit);
        i += len, plain = i;
    }
    encode_plain(text.substr(plain), out);
    return out;
}

}  // namespace strix
