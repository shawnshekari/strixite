#include "serve/output_parser.hpp"

#include "common/check.hpp"
#include "serve/unicode.hpp"

#include <algorithm>

namespace strix {

namespace {

// Length of the longest prefix of s made of whole UTF-8 characters (s is otherwise well-formed).
size_t whole_chars(const std::string &s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= std::min<size_t>(3, n); ++back) {
        const unsigned char c = (unsigned char)s[n - back];
        if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking for the lead
        const size_t need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
        return need > back ? n - back : n;
    }
    return n;  // 4+ trailing continuation bytes can't be one character: leave them to fail at decode
}

// Byte offset where s's trailing white space starts (Python isspace), s whole characters.
size_t trailing_ws_start(const std::string &s) {
    const std::vector<uint32_t> cps = unicode::decode_utf8(s, "output text");
    size_t end = 0, at = 0;
    for (uint32_t c : cps) {
        std::string one;
        unicode::append_utf8(one, c);
        at += one.size();
        if (!unicode::python_isspace(c)) end = at;
    }
    return end;
}

void skip_ws(const std::string &s, size_t &i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
}

const json::Value *param_schema(const json::Value *tools, const std::string &fn, const std::string &param) {
    if (!tools || !tools->is_array()) return nullptr;
    for (const json::Value &t : tools->as_array("tools")) {
        const json::Value *f = t.find("function");
        const json::Value *name = f ? f->find("name") : nullptr;
        if (!name || !name->is_string() || name->as_string("name") != fn) continue;
        const json::Value *params = f->find("parameters");
        const json::Value *props = params ? params->find("properties") : nullptr;
        return props ? props->find(param) : nullptr;
    }
    return nullptr;
}

bool is_string_typed(const json::Value *schema) {
    if (!schema) return true;  // not in the schema: keep what the model wrote
    const json::Value *type = schema->find("type");
    if (!type) return false;
    if (type->is_string()) return type->as_string("type") == "string";
    if (!type->is_array()) return false;
    // ["string"] or ["string", "null"]: a string; anything else in the list could be JSON.
    bool str = false;
    for (const json::Value &v : type->as_array("type")) {
        if (!v.is_string()) return false;
        const std::string &t = v.as_string("type");
        if (t == "string") str = true;
        else if (t != "null") return false;
    }
    return str;
}

const std::string kFn = "<function=", kParam = "<parameter=", kParamEnd = "</parameter>", kFnEnd = "</function>";

bool starts_with(const std::string &s, size_t at, const std::string &prefix) { return s.compare(at, prefix.size(), prefix) == 0; }
// s[at..] is a proper prefix of tag (more bytes could still make it the tag).
bool could_become(const std::string &s, size_t at, const std::string &tag) {
    const size_t n = s.size() - at;
    return n < tag.size() && tag.compare(0, n, s, at, n) == 0;
}
// A JSON string's text without the quotes.
std::string json_escaped(std::string_view s) {
    std::string q;
    json::append_quoted(q, s);
    return q.substr(1, q.size() - 2);
}

}  // namespace

bool parse_tool_call(const std::string &body, const json::Value *tools, ToolCallOut &out) {
    size_t i = 0;
    skip_ws(body, i);
    if (body.compare(i, kFn.size(), kFn) != 0) return false;
    i += kFn.size();
    const size_t name_end = body.find('>', i);
    if (name_end == std::string::npos || name_end == i || body.find('\n', i) < name_end) return false;
    ToolCallOut call;
    call.name = body.substr(i, name_end - i);
    call.arguments = json::Value::object();
    i = name_end + 1;
    for (;;) {
        skip_ws(body, i);
        if (body.compare(i, kFnEnd.size(), kFnEnd) == 0) {
            i += kFnEnd.size();
            skip_ws(body, i);
            if (i != body.size()) return false;
            break;
        }
        if (body.compare(i, kParam.size(), kParam) != 0) return false;
        i += kParam.size();
        const size_t pn_end = body.find('>', i);
        if (pn_end == std::string::npos || pn_end == i || body.find('\n', i) < pn_end) return false;
        const std::string pname = body.substr(i, pn_end - i);
        i = pn_end + 1;
        const size_t v_end = body.find(kParamEnd, i);
        if (v_end == std::string::npos) return false;
        std::string value = body.substr(i, v_end - i);
        if (!value.empty() && value.front() == '\n') value.erase(0, 1);  // the format's newline after the tag
        if (!value.empty() && value.back() == '\n') value.pop_back();    // ... and before the closing tag
        i = v_end + kParamEnd.size();
        json::Value v = json::Value::string(value);
        if (!is_string_typed(param_schema(tools, call.name, pname))) {
            try {
                v = json::Value::parse(value);
            } catch (const Error &) {
                v = json::Value::string(value);
            }
        }
        call.arguments.set(pname, std::move(v));
    }
    out = std::move(call);
    return true;
}

OutputParser::OutputParser(const Tokenizer &tok, Config cfg)
    : tok_(tok), cfg_(std::move(cfg)), think_end_(tok.id_of("</think>")), call_begin_(tok.id_of("<tool_call>")),
      call_end_(tok.id_of("</tool_call>")), phase_(cfg_.thinking ? Phase::Reasoning : Phase::Content), rng_(cfg_.id_seed) {
    for (const std::string &s : cfg_.stop) {
        STRIX_CHECK(!s.empty() && json::valid_utf8(s), "OutputParser: stop strings must be non-empty UTF-8");
        max_stop_ = std::max(max_stop_, s.size());
    }
}

void OutputParser::add(Channel &c, const std::string &bytes, std::vector<OutputEvent> &out, bool *stopped) {
    c.bytes += bytes;
    const size_t n = whole_chars(c.bytes);
    std::string whole = c.bytes.substr(0, n);
    c.bytes.erase(0, n);
    if (!c.started && cfg_.trim_ws) {
        const std::vector<uint32_t> cps = unicode::decode_utf8(whole, "output text");
        size_t k = 0;
        while (k < cps.size() && unicode::python_isspace(cps[k])) ++k;
        whole = unicode::encode_utf8(cps, k, cps.size());
        if (whole.empty()) return;
        c.started = true;
    }
    c.pending += whole;
    if (stopped && !cfg_.stop.empty()) {
        size_t hit = std::string::npos;
        for (const std::string &s : cfg_.stop) hit = std::min(hit, c.pending.find(s));
        if (hit != std::string::npos) {
            c.pending.erase(hit);
            release(c, out, true);
            *stopped = true;
            return;
        }
    }
    // Hold back the last max_stop - 1 bytes (a stop string may still complete there), and the white space
    // before them (it's trailing if the text ends there).
    size_t cut = c.pending.size();
    if (stopped && max_stop_ > 1) cut -= std::min(cut, max_stop_ - 1);
    while (cut > 0 && cut < c.pending.size() && ((unsigned char)c.pending[cut] & 0xC0) == 0x80) --cut;
    const size_t keep = cfg_.trim_ws ? trailing_ws_start(c.pending.substr(0, cut)) : cut;
    if (keep == 0) return;
    out.push_back({c.kind, c.pending.substr(0, keep), {}});
    c.pending.erase(0, keep);
}

void OutputParser::release(Channel &c, std::vector<OutputEvent> &out, bool drop_trailing_ws) {
    c.bytes.clear();  // an incomplete character at the end can't be shown
    if (drop_trailing_ws && cfg_.trim_ws) c.pending.erase(trailing_ws_start(c.pending));
    if (!c.pending.empty()) out.push_back({c.kind, c.pending, {}});
    c.pending.clear();
}

bool OutputParser::feed(int32_t id, std::vector<OutputEvent> &out) {
    STRIX_CHECK(!stopped_, "OutputParser::feed after a stop string ended generation");
    const std::string &bytes = tok_.token_bytes(id);
    switch (phase_) {
    case Phase::Reasoning:
        ++reasoning_tokens_;
        if (id == think_end_) {
            release(reasoning_, out, true);
            phase_ = Phase::Content;
        } else {
            add(reasoning_, bytes, out, nullptr);
        }
        return false;
    case Phase::Content:
        if (id == call_begin_) {  // held content stays held: trimmed if the call parses, kept if it doesn't
            phase_ = Phase::Call;
            call_body_.clear();
            call_state_ = CallState::Header, call_at_ = name_scan_ = 0;
            return false;
        }
        add(content_, bytes, out, &stopped_);
        return stopped_;
    case Phase::Call:
        if (id != call_end_) {
            call_body_ += bytes;
            if (cfg_.stream_calls) advance_call(out);
            return false;
        }
        phase_ = Phase::Content;
        if (cfg_.stream_calls && call_state_ != CallState::Header && call_state_ != CallState::FnName &&
            call_state_ != CallState::NoHeader) {
            if (call_state_ == CallState::Done) {  // well-formed: the arguments close now
                out.push_back({OutputEvent::Kind::ToolCallArgs, "}", {}});
                ++tool_calls_;
            } else {
                ++malformed_calls_;
            }
            return false;
        }
        if (ToolCallOut call; json::valid_utf8(call_body_) && parse_tool_call(call_body_, cfg_.tools, call)) {
            call.id = new_call_id();
            release(content_, out, true);
            out.push_back({OutputEvent::Kind::ToolCall, {}, std::move(call)});
            ++tool_calls_;
        } else {  // not a call the model formed correctly: show it as the text it is
            add(content_, "<tool_call>" + call_body_ + "</tool_call>", out, &stopped_);
            return stopped_;
        }
        return false;
    }
    return false;
}

std::string OutputParser::new_call_id() {
    static const char *kAlnum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string id = "call_";
    for (int k = 0; k < 24; ++k) id += kAlnum[rng_() % 62];
    return id;
}

// The streamed form of parse_tool_call: the same grammar, read as far as call_body_ goes, each part sent once it
// can no longer change.
void OutputParser::advance_call(std::vector<OutputEvent> &out) {
    const std::string &b = call_body_;
    const auto args = [&](std::string piece) { out.push_back({OutputEvent::Kind::ToolCallArgs, std::move(piece), {}}); };
    // A name between call_at_ and the next '>': its end, npos to wait, or 0 when it can't be one (empty, or a newline
    // before the '>'). name_scan_: where the last look stopped (no '>' or newline before it).
    const auto name_end = [&]() -> size_t {
        const size_t at = std::max(call_at_, name_scan_), e = b.find_first_of(">\n", at);
        if (e == std::string::npos) {
            name_scan_ = b.size();
            return e;
        }
        return b[e] == '\n' || e == call_at_ ? 0 : e;
    };
    for (;;) {
        switch (call_state_) {
        case CallState::Header: {
            size_t i = call_at_;
            skip_ws(b, i);
            call_at_ = i;
            if (i == b.size() || could_become(b, i, kFn)) return;
            if (!starts_with(b, i, kFn)) {
                call_state_ = CallState::NoHeader;
                return;
            }
            call_at_ = name_scan_ = i + kFn.size(), call_state_ = CallState::FnName;
            break;
        }
        case CallState::FnName: {
            const size_t end = name_end();
            if (end == std::string::npos) return;
            const std::string name = end ? b.substr(call_at_, end - call_at_) : std::string();
            if (!end || !json::valid_utf8(name)) {
                call_state_ = CallState::NoHeader;
                return;
            }
            // The call goes out: the content before it is trimmed as when a buffered call parses.
            release(content_, out, true);
            ToolCallOut start;
            start.id = new_call_id(), start.name = call_name_ = name;
            out.push_back({OutputEvent::Kind::ToolCallStart, {}, std::move(start)});
            args("{");
            first_param_ = true, params_seen_.clear();
            call_at_ = end + 1, call_state_ = CallState::Next;
            break;
        }
        case CallState::Next: {
            size_t i = call_at_;
            skip_ws(b, i);
            call_at_ = i;
            if (i == b.size() || could_become(b, i, kFnEnd) || could_become(b, i, kParam)) return;
            if (starts_with(b, i, kFnEnd)) {  // the closing "}" waits for </tool_call>, as the buffered parse does
                call_at_ = i + kFnEnd.size(), call_state_ = CallState::Done;
            } else if (starts_with(b, i, kParam)) {
                call_at_ = name_scan_ = i + kParam.size(), call_state_ = CallState::Name;
            } else {
                call_state_ = CallState::Broken;
            }
            break;
        }
        case CallState::Name: {
            const size_t end = name_end();
            if (end == std::string::npos) return;
            param_name_ = end ? b.substr(call_at_, end - call_at_) : std::string();
            if (!end || !json::valid_utf8(param_name_) || !params_seen_.insert(param_name_).second) {
                call_state_ = CallState::Broken;
                break;
            }
            value_string_ = is_string_typed(param_schema(cfg_.tools, call_name_, param_name_));
            if (value_string_) {  // the key goes out now, the value as it comes
                std::string piece = first_param_ ? "" : ",";
                json::append_quoted(piece, param_name_);
                args(piece + ":\"");
                first_param_ = false;
            }
            call_at_ = value_scan_ = end + 1, value_started_ = false, call_state_ = CallState::Value;
            break;
        }
        case CallState::Value: {
            if (!value_started_) {  // the format's newline after the tag isn't part of the value
                if (call_at_ == b.size()) return;
                value_start_ = value_sent_ = call_at_ + (b[call_at_] == '\n');
                value_started_ = true;
            }
            const size_t v_end = b.find(kParamEnd, value_scan_);
            if (v_end == std::string::npos) value_scan_ = std::max(value_scan_, b.size() - std::min(b.size(), kParamEnd.size() - 1));
            if (v_end != std::string::npos) {
                size_t end = v_end;
                if (end > value_start_ && b[end - 1] == '\n') --end;  // ... nor the one before the closing tag
                if (value_string_) {
                    const std::string rest = b.substr(value_sent_, end - value_sent_);
                    if (!json::valid_utf8(rest)) {
                        call_state_ = CallState::Broken;
                        break;
                    }
                    args(json_escaped(rest) + "\"");
                } else {
                    const std::string raw = b.substr(value_start_, end - value_start_);
                    if (!json::valid_utf8(raw)) {
                        call_state_ = CallState::Broken;
                        break;
                    }
                    json::Value v = json::Value::string(raw);
                    try {
                        v = json::Value::parse(raw);
                    } catch (const Error &) {
                    }
                    std::string piece = first_param_ ? "" : ",";
                    json::append_quoted(piece, param_name_);
                    args(piece + ":" + v.dump());
                    first_param_ = false;
                }
                call_at_ = v_end + kParamEnd.size(), call_state_ = CallState::Next;
                break;
            }
            if (!value_string_) return;  // a typed value goes out whole
            // Send what can't change any more: not a tail that could still become "</parameter>", not a newline that
            // could still be the one before it, not part of a character.
            size_t safe = b.size();
            for (size_t k = std::min(kParamEnd.size() - 1, b.size() - value_sent_); k > 0; --k)
                if (kParamEnd.compare(0, k, b, b.size() - k, k) == 0) {
                    safe = b.size() - k;
                    break;
                }
            if (safe > value_sent_ && b[safe - 1] == '\n') --safe;
            safe = value_sent_ + whole_chars(b.substr(value_sent_, safe - value_sent_));
            if (safe == value_sent_) return;
            const std::string piece = b.substr(value_sent_, safe - value_sent_);
            if (!json::valid_utf8(piece)) {
                call_state_ = CallState::Broken;
                break;
            }
            args(json_escaped(piece));
            value_sent_ = safe;
            return;
        }
        case CallState::Done: {
            size_t i = call_at_;
            skip_ws(b, i);
            call_at_ = i;
            if (i == b.size()) return;
            call_state_ = CallState::Broken;  // text after </function>
            break;
        }
        case CallState::NoHeader:
        case CallState::Broken: return;
        }
    }
}

void OutputParser::finish(std::vector<OutputEvent> &out) {
    if (phase_ == Phase::Call) {
        dropped_call_ = true;
        // A streamed one went out: it stays a call, unterminated (finish_reason tool_calls unless it's length).
        if (cfg_.stream_calls && call_state_ != CallState::Header && call_state_ != CallState::FnName &&
            call_state_ != CallState::NoHeader)
            ++malformed_calls_;
    }
    release(reasoning_, out, true);
    release(content_, out, true);
}

}  // namespace strix
