// Streamed tool calls against the buffered parser (serve/output_parser.hpp), host only: the same generated text,
// fed as the tokenizer splits it, a byte at a time, and in random pieces, through an OutputParser with and without
// Config::stream_calls. The stream must put out what the buffered parser does for the same tokens with each call the
// stream left unterminated replaced by an empty call of the same name: the same events in the same order -
// reasoning, content, call ids, names and arguments - except those calls' arguments, which must not parse; and for
// well-formed text exactly what the buffered parser does. Fixed cases, random well-formed calls, then random calls
// (odd names, white space, schemas, thinking, stop strings, bad bytes) and random damage to them.
//
//   check_stream_calls <tokenizer.json> [random cases, default 20000] [seed]

#include "common/check.hpp"
#include "serve/json.hpp"
#include "serve/output_parser.hpp"
#include "serve/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace strix;

namespace {

const char *kSpecials[] = {"<tool_call>", "</tool_call>", "</think>"};

// What a parser put out, in order: adjacent pieces of the same kind joined; a whole ToolCall as a start (id, name) and
// its arguments, as a stream sends it.
struct Event {
    char kind;  // R reasoning, C content, S call start, A arguments
    std::string text, id;
    bool operator==(const Event &o) const { return kind == o.kind && text == o.text && id == o.id; }
};
struct Run {
    std::vector<Event> events;
    int64_t tool_calls = 0, malformed = 0;
    bool dropped = false, threw = false;
    // Streamed: each call region (<tool_call> in content to </tool_call>, or to the end): its token range, whether a
    // start went out, the name, and whether its arguments closed.
    struct Region {
        size_t begin, end;  // end: the </tool_call> token, or ids.size() if cut off
        bool started = false, closed = false;
        std::string name;
    };
    std::vector<Region> regions;
    std::vector<int> arg_pieces;  // per call: ToolCallArgs events (streamed)
};

void push(std::vector<Event> &ev, char kind, const std::string &text, const std::string &id = {}) {
    if (kind != 'S' && !ev.empty() && ev.back().kind == kind) ev.back().text += text;
    else ev.push_back({kind, text, id});
}

// Text -> token ids: special markers as their tokens, the rest per `mode` (0 the tokenizer's split, 1 a byte each,
// 2 random pieces at character boundaries, each encoded alone).
std::vector<int32_t> tokens(const Tokenizer &tok, const std::string &text, int mode, std::mt19937_64 &rng) {
    std::vector<int32_t> ids;
    const auto plain = [&](const std::string &s) {
        if (s.empty()) return;
        if (mode == 1 || !json::valid_utf8(s)) {  // the tokenizer takes UTF-8 only: other bytes as byte tokens
            for (unsigned char c : s) ids.push_back(tok.byte_token(c));
            return;
        }
        if (mode == 0) {
            for (int32_t t : tok.encode(s)) ids.push_back(t);
            return;
        }
        size_t at = 0;
        while (at < s.size()) {
            size_t n = std::min(s.size() - at, (size_t)(1 + rng() % 7));
            while (at + n < s.size() && ((unsigned char)s[at + n] & 0xC0) == 0x80) ++n;
            const std::string piece = s.substr(at, n);
            if (json::valid_utf8(piece))
                for (int32_t t : tok.encode(piece)) ids.push_back(t);
            else
                for (unsigned char c : piece) ids.push_back(tok.byte_token(c));
            at += n;
        }
    };
    size_t at = 0;
    for (;;) {
        size_t best = std::string::npos;
        const char *which = nullptr;
        for (const char *sp : kSpecials)
            if (size_t p = text.find(sp, at); p < best) best = p, which = sp;
        if (!which) break;
        plain(text.substr(at, best - at));
        ids.push_back(tok.id_of(which));
        at = best + std::string(which).size();
    }
    plain(text.substr(at));
    return ids;
}

Run run(const Tokenizer &tok, const std::vector<int32_t> &ids, const json::Value *tools, bool thinking, bool stream,
        const std::vector<std::string> &stop) {
    OutputParser::Config cfg;
    cfg.thinking = thinking, cfg.tools = tools, cfg.id_seed = 42, cfg.stream_calls = stream, cfg.stop = stop;
    OutputParser p(tok, cfg);
    const int32_t begin = tok.id_of("<tool_call>"), end = tok.id_of("</tool_call>");
    Run r;
    bool in_call = false;
    const auto take = [&](std::vector<OutputEvent> &ev) {
        for (const OutputEvent &e : ev) {
            if (e.kind == OutputEvent::Kind::Reasoning) push(r.events, 'R', e.text);
            else if (e.kind == OutputEvent::Kind::Content) push(r.events, 'C', e.text);
            else if (e.kind == OutputEvent::Kind::ToolCall) {
                push(r.events, 'S', e.call.name, e.call.id);
                push(r.events, 'A', e.call.arguments.dump());
            } else if (e.kind == OutputEvent::Kind::ToolCallStart) {
                push(r.events, 'S', e.call.name, e.call.id);
                r.arg_pieces.push_back(0);
                if (in_call) r.regions.back().started = true, r.regions.back().name = e.call.name;
            } else {
                STRIX_CHECK(!r.events.empty() && (r.events.back().kind == 'S' || r.events.back().kind == 'A'),
                            "ToolCallArgs outside a call");
                push(r.events, 'A', e.text);
                ++r.arg_pieces.back();
            }
        }
        ev.clear();
    };
    std::vector<OutputEvent> ev;
    try {
        for (size_t i = 0; i < ids.size(); ++i) {
            const bool opens = !in_call && ids[i] == begin && !p.in_reasoning();
            const bool stopped = p.feed(ids[i], ev);
            if (opens) in_call = true, r.regions.push_back({i, ids.size(), false, false, {}});
            if (in_call && ids[i] == end && !opens) {
                in_call = false;
                r.regions.back().end = i;
                for (const OutputEvent &e : ev) r.regions.back().closed |= e.kind == OutputEvent::Kind::ToolCallArgs && e.text == "}";
            }
            take(ev);
            if (stopped) break;
        }
        p.finish(ev);
        take(ev);
    } catch (const Error &) {  // invalid UTF-8 reaching a text channel (the engine fails the request)
        r.threw = true;
    }
    r.tool_calls = p.tool_calls(), r.malformed = p.malformed_calls(), r.dropped = p.dropped_partial_call();
    return r;
}

bool parses_as_object(const std::string &s) {
    try {
        return json::Value::parse(s).is_object();
    } catch (const Error &) {
        return false;
    }
}

// Unterminated arguments as a stream leaves them: not JSON, but an object's opening that "}" or "\"}" closes (no
// escape or character split, nothing raw).
bool open_prefix(const std::string &s) {
    return !parses_as_object(s) && (parses_as_object(s + "}") || parses_as_object(s + "\"}"));
}

int failures = 0;
void fail(const std::string &what, const std::string &text, int mode) {
    if (++failures <= 20) std::fprintf(stderr, "FAIL (mode %d): %s\n--- text ---\n%s\n------------\n", mode, what.c_str(), text.c_str());
}

std::string show(const std::vector<Event> &ev) {
    std::string s;
    for (const Event &e : ev) s += std::string(1, e.kind) + "[" + e.text + "]";
    return s;
}

// The oracle: the stream must put out exactly what the buffered parser does for the same tokens with each call the
// stream left unterminated (malformed or cut off) replaced by an empty call of the same name - same events in the
// same order (reasoning, content, call ids and names, arguments), except that call's arguments, which must not parse.
// long_value: the text holds a string value long enough that it must have gone out in more than one piece.
void check(const Tokenizer &tok, const std::string &text, const json::Value *tools, bool thinking, bool well_formed,
           std::mt19937_64 &rng, const std::vector<std::string> &stop = {}, bool long_value = false) {
    for (int mode = 0; mode < 3; ++mode) {
        const std::vector<int32_t> ids = tokens(tok, text, mode, rng);
        const Run b = run(tok, ids, tools, thinking, false, stop), s = run(tok, ids, tools, thinking, true, stop);
        std::vector<int32_t> neutral;
        std::vector<bool> open_call, cut_call;  // per streamed call, in order: left unterminated; cut off by the end
        size_t at = 0;
        for (const Run::Region &g : s.regions) {
            if (!g.started) continue;
            open_call.push_back(!g.closed), cut_call.push_back(!g.closed && g.end == ids.size());
            if (g.closed) continue;
            if (g.end < ids.size()) {  // left open though it ended: the buffered parser must reject it, or it repeats a name
                std::string body;
                for (size_t k = g.begin + 1; k < g.end; ++k) body += tok.token_bytes(ids[k]);
                ToolCallOut c;
                size_t tags = 0;
                for (size_t q = body.find("<parameter="); q != std::string::npos; q = body.find("<parameter=", q + 1)) ++tags;
                if (json::valid_utf8(body) && parse_tool_call(body, tools, c) && c.arguments.as_object("arguments").size() == tags)
                    fail("the stream left open a call the buffered parser takes", text, mode);
            }
            neutral.insert(neutral.end(), ids.begin() + (long)at, ids.begin() + (long)g.begin);
            neutral.push_back(tok.id_of("<tool_call>"));
            for (int32_t t : tok.encode("<function=" + g.name + "></function>")) neutral.push_back(t);
            if (g.end < ids.size()) neutral.push_back(tok.id_of("</tool_call>"));
            at = g.end < ids.size() ? g.end + 1 : ids.size();
        }
        neutral.insert(neutral.end(), ids.begin() + (long)at, ids.end());
        const Run n = run(tok, neutral, tools, thinking, false, stop);
        if (s.threw != n.threw) fail(s.threw ? "streamed threw, neutralized buffered didn't" : "neutralized buffered threw, stream didn't", text, mode);
        if (s.threw || n.threw) continue;
        // A call cut off by the end is dropped by the buffered parser: compare without it (its arguments mustn't parse).
        std::vector<Event> se;
        std::vector<size_t> se_call;  // the streamed call each kept event belongs to (0: none)
        for (size_t k = 0, call = 0; k < s.events.size(); ++k) {
            if (s.events[k].kind == 'S') ++call;
            const bool in_call = s.events[k].kind == 'S' || s.events[k].kind == 'A';
            if (in_call && call >= 1 && call <= cut_call.size() && cut_call[call - 1]) {
                if (s.events[k].kind == 'A' && !open_prefix(s.events[k].text))
                    fail("cut-off call's arguments parse, or aren't an object's opening", text, mode);
                continue;
            }
            if (!se.empty() && !in_call && se.back().kind == s.events[k].kind) se.back().text += s.events[k].text;
            else se.push_back(s.events[k]), se_call.push_back(in_call ? call : 0);
        }
        bool same = se.size() == n.events.size();
        for (size_t k = 0; same && k < se.size(); ++k) {
            const Event &x = se[k], &y = n.events[k];
            const size_t call = se_call[k];
            if (x.kind == 'A' && call >= 1 && call <= open_call.size() && open_call[call - 1])
                same = y.kind == 'A' && y.text == "{}" && open_prefix(x.text);
            else
                same = x == y;
        }
        if (!same) fail("events differ:\n  stream:  " + show(s.events) + "\n  neutral: " + show(n.events), text, mode);
        int64_t open = 0;
        for (bool o : open_call) open += o;
        if (s.malformed != open) fail("malformed " + std::to_string(s.malformed) + " vs unterminated " + std::to_string(open), text, mode);
        if (s.tool_calls != (int64_t)open_call.size() - open) fail("tool_calls isn't the closed calls", text, mode);
        if (s.dropped != n.dropped || (!b.threw && stop.empty() && s.dropped != b.dropped)) fail("dropped differs", text, mode);
        if (well_formed && (open != 0 || b.events != s.events)) fail("well-formed: stream differs from buffered:\n  " + show(b.events) + "\n  " + show(s.events), text, mode);
        if (long_value) {
            bool split = false;
            for (int pieces : s.arg_pieces) split |= pieces > 4;
            if (!split) fail("a long string value went out in one piece", text, mode);
        }
    }
}

std::string pick(std::mt19937_64 &rng, const std::vector<std::string> &v) { return v[rng() % v.size()]; }

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <tokenizer.json> [random cases] [seed]\n", argv[0]);
        return 2;
    }
    const Tokenizer tok(argv[1]);
    const int n_random = argc > 2 ? std::atoi(argv[2]) : 20000;
    std::mt19937_64 rng(argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1);
    const json::Value tools = json::Value::parse(R"([
      {"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{
        "path":{"type":"string"},"content":{"type":"string"},"mode":{"type":["string","null"]}}}}},
      {"type":"function","function":{"name":"run","parameters":{"type":"object","properties":{
        "cmd":{"type":"string"},"timeout":{"type":"integer"},"env":{"type":"object"},"args":{"type":"array"},
        "dry":{"type":"boolean"}}}}}])");

    // Fixed cases: well-formed (1) and not (0).
    const std::vector<std::pair<std::string, int>> fixed = {
        {"<tool_call>\n<function=run>\n<parameter=cmd>\nls -la\n</parameter>\n</function>\n</tool_call>", 1},
        {"Let me look.\n\n<tool_call>\n<function=run>\n<parameter=cmd>\nls\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>", 1},
        {"<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/a \"b\".txt\n</parameter>\n<parameter=content>\nline 1\n\tline \\2\n\n</par not yet\nend </parameter\n</parameter>\n</function>\n</tool_call>", 1},
        {"<tool_call>\n<function=write_file>\n<parameter=content>\n\n</parameter>\n<parameter=path>\n\n\n</parameter>\n</function>\n</tool_call>", 1},
        {"<tool_call>\n<function=write_file>\n<parameter=content></parameter>\n<parameter=path>x</parameter>\n</function>\n</tool_call>", 1},
        {"<tool_call>\n<function=write_file>\n<parameter=content>\nh\xc3\xa9llo \xe4\xb8\x96\xe7\x95\x8c \xf0\x9f\x98\x80\n</parameter>\n</function>\n</tool_call>", 1},
        {"<tool_call>\n<function=run>\n<parameter=env>\n{\"A\": 1, \"B\": [1, 2.50, \"x\"]}\n</parameter>\n<parameter=args>\nnot json\n</parameter>\n<parameter=dry>\ntrue\n</parameter>\n</function>\n</tool_call>", 1},
        {"<tool_call>\n<function=unknown_fn>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>", 1},
        {"<tool_call>\n<function=run>\n</function>\n</tool_call>", 1},
        {"<tool_call>\n<function=run>\n<parameter=cmd>\na\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=write_file>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>", 1},
        {"thinking about <tool_call> it</think>\n\nok <tool_call>\n<function=run>\n<parameter=cmd>\nx\n</parameter>\n</function>\n</tool_call>", 1},
        // Never a header: text in both paths.
        {"<tool_call>\nnot a call\n</tool_call> after", 0},
        {"<tool_call>\n<function=\n</function>\n</tool_call>", 0},
        {"<tool_call>\n<function=a\nb>\n</function>\n</tool_call>", 0},
        {"<tool_call></tool_call>", 0},
        // Header went out, then broken / cut off.
        {"<tool_call>\n<function=run>\n<parameter=cmd>\nls\n</function>\n</tool_call>", 0},
        {"<tool_call>\n<function=run>\njunk\n</function>\n</tool_call>", 0},
        {"<tool_call>\n<function=run>\n<parameter=cmd>\nls\n</parameter>\n</function>\ntrailing\n</tool_call>", 0},
        {"<tool_call>\n<function=run>\n<parameter=cmd>\nls -la /very/long", 0},
        {"<tool_call>\n<function=run>\n<parameter=cmd>\nls\n</parameter>\n</function>", 0},
        {"<tool_call>\n<function=run>\n<parameter=timeout>\n30", 0},
        {"<tool_call>\n<function=run>\n<parameter=cmd>\na\n</parameter>\n<parameter=cmd>\nb\n</parameter>\n</function>\n</tool_call>", 2},
        {"<tool_call>\n<function=run>\n<parameter=timeout>\n1\n</parameter>\n<parameter=timeout>\n2\n</parameter>\n</function>\n</tool_call>", 2},
    };
    for (const auto &[text, expect] : fixed) {
        const bool thinking = text.find("</think>") != std::string::npos;
        check(tok, text, &tools, thinking, expect == 1, rng);
        check(tok, text, nullptr, thinking, expect == 1, rng);
    }
    {  // a long string value streams in pieces
        std::string v;
        for (int k = 0; k < 40; ++k) v += "line " + std::to_string(k) + " of a file\n";
        check(tok, "<tool_call>\n<function=write_file>\n<parameter=content>\n" + v + "</parameter>\n</function>\n</tool_call>", &tools,
              false, true, rng, {}, true);
    }
    const int fixed_failures = failures;

    // Random well-formed calls (valid names, no repeated parameter, no tag text in a value): the stream must match
    // exactly.
    {
        const std::vector<std::string> vals = {"a", "Z", " ", "\n", "\n\n", "\t", "\"", "\\", "/", "<", ">", "</", "</par",
                                               "</parameter", "{", "}", "[1, 2]", "3", "-0.5e3", "true", "null", "\xc3\xa9",
                                               "\xe4\xb8\x96", "\xf0\x9f\x98\x80", "x y", "def f():\n    return 1\n"};
        for (int c = 0; c < n_random; ++c) {
            std::string text;
            if (rng() % 3 == 0) text += pick(rng, {"Sure.", "Running it now:", " ", "\n"});
            for (int k = 1 + (int)(rng() % 2); k > 0; --k) {
                text += pick(rng, {"", "\n", "\n\n"}) + "<tool_call>" + pick(rng, {"\n", "", " "}) + "<function=" +
                        pick(rng, {"write_file", "run", "other"}) + ">";
                std::vector<std::string> used;
                for (int j = (int)(rng() % 4); j > 0; --j) {
                    const std::string p = pick(rng, {"path", "content", "mode", "cmd", "timeout", "env", "args", "dry", "free"});
                    bool dup = false;
                    for (const std::string &u : used) dup |= u == p;
                    if (dup) continue;
                    used.push_back(p);
                    std::string v;
                    for (int m = (int)(rng() % 6); m > 0; --m) v += pick(rng, vals);
                    if (v.find("</parameter>") != std::string::npos) continue;
                    text += pick(rng, {"\n", ""}) + "<parameter=" + p + ">" + pick(rng, {"\n", ""}) + v + pick(rng, {"\n", ""}) + "</parameter>";
                }
                text += pick(rng, {"\n", ""}) + "</function>" + pick(rng, {"\n", ""}) + "</tool_call>";
            }
            if (rng() % 4 == 0) text += pick(rng, {" done", "\n\nAnd more.", "  "});
            check(tok, text, rng() % 5 ? &tools : nullptr, false, true, rng);
        }
    }

    // Random calls, then random damage: names, white space, values and schemas a model could produce or mangle.
    const json::Value typed = json::Value::parse(R"([{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{
        "s":{"type":"string"},"sn":{"type":["string","null"]},"i":{"type":"integer"},"o":{"type":"object"},
        "is":{"type":["integer","string"]},"nt":{"description":"no type"},"n":{"type":"number"},"b":{"type":"boolean"},
        "a":{"type":"array"}}}}}])");
    const std::vector<std::string> pieces = {"a", "Z", " ", "\n", "\n\n", "\r", "\r\n", "\t", "\"", "\\", "/", "<", ">", "</",
                                             "</par", "</parameter", "</function", "</function>", "<parameter=", "</tool_call>",
                                             "{", "}", "{\"a\": 1}", "[1, 2]", "3", "-0.5e3", "1e400", "true", "null",
                                             "\xc3\xa9", "\xe4\xb8\x96", "\xf0\x9f\x98\x80", "\xe2\x80\xa8", "\x7f", "\x1f", "x y",
                                             "def f():\n    return 1\n", "END"};
    const std::vector<std::string> bad_bytes = {"\xff", "\xc3", "\x80", "\xed\xa0\x80"};
    const std::vector<std::string> fns = {"write_file", "run", "other", "f", "f ", "f\t", "\xc3\xa9", "a<b", "f\r"};
    const std::vector<std::string> params = {"path", "content", "mode", "cmd", "timeout", "env", "args", "dry", "free",
                                             "s", "sn", "i", "o", "is", "nt", "n", "b", "a", "p q", "\xc3\xa9"};
    const std::vector<std::string> ws = {"", "\n", "\n", " ", "\r\n", "\t", "\n\n"};
    const std::vector<std::vector<std::string>> stops = {{"END"}, {"\n\n"}, {"</"}, {"a"}};
    for (int c = 0; c < n_random; ++c) {
        const bool invalid = rng() % 8 == 0, thinking = rng() % 6 == 0;
        std::string text = thinking ? pick(rng, {"Let me think <tool_call> about it", "ok", "\n"}) + "</think>" : "";
        if (rng() % 3 == 0) text += pick(rng, {"Sure.", "Running it now:", " ", "\n"});
        const int n_calls = 1 + (int)(rng() % 2);
        for (int k = 0; k < n_calls; ++k) {
            text += pick(rng, {"", "\n", "\n\n"}) + "<tool_call>" + pick(rng, ws) + "<function=" + pick(rng, fns) + ">";
            for (int j = (int)(rng() % 4); j > 0; --j) {
                const std::string p = pick(rng, params);
                std::string v;
                for (int m = (int)(rng() % 6); m > 0; --m) v += invalid && rng() % 6 == 0 ? pick(rng, bad_bytes) : pick(rng, pieces);
                text += pick(rng, ws) + "<parameter=" + p + ">" + pick(rng, {"\n", "", "\n"}) + v + pick(rng, {"\n", "", "\n"}) + "</parameter>";
            }
            text += pick(rng, ws) + "</function>" + pick(rng, {"\n", "", " "}) + "</tool_call>";
        }
        if (rng() % 4 == 0) text += pick(rng, {" done", "\n\nAnd more.", "  "});
        const json::Value *t = rng() % 4 == 0 ? nullptr : rng() % 2 ? &tools : &typed;
        const std::vector<std::string> stop = rng() % 5 == 0 ? stops[rng() % stops.size()] : std::vector<std::string>{};
        check(tok, text, t, thinking, false, rng, stop);
        // Damage: cut it off, insert a piece or a bad byte somewhere, or drop a byte.
        std::string bad = text;
        switch (rng() % 3) {
        case 0: bad = bad.substr(0, rng() % (bad.size() + 1)); break;
        case 1: bad.insert(rng() % (bad.size() + 1), rng() % 4 ? pick(rng, pieces) : pick(rng, bad_bytes)); break;
        default: if (!bad.empty()) bad.erase(rng() % bad.size(), 1);
        }
        check(tok, bad, t, thinking, false, rng, stop);
    }
    std::printf("%d fixed cases, %d random well-formed, %d random (x2 with damage), 3 feeds each: %d failures (%d in the "
                "fixed cases)\n",
                (int)fixed.size() * 2, n_random, n_random, failures, fixed_failures);
    return failures ? 1 : 0;
}
