#pragma once

// Generated tokens -> what the API returns, incrementally for streaming.
// The prompt ends inside the think block ("<think>\n"), so generation starts as reasoning until the </think>
// token, then content; a <tool_call> token opens a call that runs to </tool_call> and is parsed from the model's
// XML form (<function=NAME> <parameter=P> value </parameter> ... </function>) into a name and JSON arguments,
// typed from the tool's JSON schema. Markers count only as the single special tokens, never as text the model
// spelled out, and only in their phase (a <tool_call> inside reasoning is reasoning text).
//
// Streamed calls (Config::stream_calls, for streamed chat responses): a call goes out as it is generated instead of
// whole at </tool_call>. ToolCallStart (id, name) as soon as <function=NAME> is complete, then ToolCallArgs pieces
// of the arguments' JSON text: "{", each parameter - a string-typed value piece by piece as its characters arrive,
// any other value whole at its </parameter> (parsed and printed as the buffered path does) - and "}" at the
// </tool_call> of a call the buffered path would parse. Joined, the pieces are its arguments.dump() byte for byte.
// What differs: once the header went out, a call can't be taken back. Where the buffered path shows the call as text
// (a stray tag, text after </function>), the streamed call stays a call whose arguments never get their "}" - they
// don't parse - and malformed_calls() counts it; so too a call that names a parameter twice (the buffered path keeps
// the last value; the stream already sent the first). A call cut off by the end of generation is left the same way
// (dropped_partial_call(), as for a buffered one, and counted malformed).
//
// Text is released in whole UTF-8 characters (a token can end inside one). Like the chat template, which trims
// reasoning and content when the turn is re-rendered, leading and trailing white space of each channel is
// dropped: trailing white space is held back until more text follows. Stop strings apply to content; text that
// could still become a stop string is held back until it can't.

#include "serve/json.hpp"
#include "serve/tokenizer.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace strix {

struct ToolCallOut {
    std::string id, name;
    json::Value arguments;  // an object
};

struct OutputEvent {
    enum class Kind { Reasoning, Content, ToolCall, ToolCallStart, ToolCallArgs } kind;
    std::string text;  // Reasoning / Content / ToolCallArgs (a piece of the arguments' JSON text)
    ToolCallOut call;  // ToolCall; ToolCallStart (id and name; arguments null)
    bool is_text() const { return kind == Kind::Reasoning || kind == Kind::Content; }
};

// Parses the text between <tool_call> and </tool_call>. tools: the request's tools (or nullptr), for typing
// parameter values: a parameter whose schema type is "string" (or that isn't in the schema) stays the raw
// string; any other is parsed as JSON, and kept as the raw string if it isn't JSON. Returns false (out
// untouched) if the text isn't a well-formed call.
bool parse_tool_call(const std::string &body, const json::Value *tools, ToolCallOut &out);

class OutputParser {
public:
    struct Config {
        bool thinking = true;           // the prompt left generation inside the think block
        const json::Value *tools = nullptr;
        std::vector<std::string> stop;  // non-empty strings
        uint64_t id_seed = 0;           // tool call ids
        // Trim each channel's leading / trailing white space (chat, like the template's re-render). Off for raw
        // /v1/completions, where it's part of the text (code indentation, a continuation's leading space).
        bool trim_ws = true;
        // Stream tool calls as they are generated (ToolCallStart / ToolCallArgs) instead of one ToolCall each.
        bool stream_calls = false;
    };
    OutputParser(const Tokenizer &tok, Config cfg);

    // One generated token (never an end-of-sequence token: the caller stops on those). Returns true when a stop
    // string completed - generation should end; this token's text past the stop string is dropped.
    bool feed(int32_t id, std::vector<OutputEvent> &out);
    // Generation over: releases held text (less trailing white space). A tool call still open is dropped - streamed,
    // left unterminated and counted malformed - and dropped_partial_call() says so.
    void finish(std::vector<OutputEvent> &out);

    int64_t reasoning_tokens() const { return reasoning_tokens_; }
    bool in_reasoning() const { return phase_ == Phase::Reasoning; }
    int64_t tool_calls() const { return tool_calls_; }
    bool dropped_partial_call() const { return dropped_call_; }
    int64_t malformed_calls() const { return malformed_calls_; }  // streamed calls that went out but weren't well-formed

private:
    // One text channel: bytes in, whole characters out, leading / trailing white space trimmed, stop strings.
    struct Channel {
        OutputEvent::Kind kind;
        std::string bytes;    // not yet whole characters
        std::string pending;  // whole characters held back (trailing white space, possible stop-string start)
        bool started = false; // a non-white-space character went out
    };
    void add(Channel &c, const std::string &bytes, std::vector<OutputEvent> &out, bool *stopped);
    void release(Channel &c, std::vector<OutputEvent> &out, bool drop_trailing_ws);
    // Streamed calls: the bytes of call_body_ from call_at_ on, as far as they can go out yet.
    void advance_call(std::vector<OutputEvent> &out);
    std::string new_call_id();

    const Tokenizer &tok_;
    Config cfg_;
    int32_t think_end_, call_begin_, call_end_;
    enum class Phase { Reasoning, Content, Call } phase_;
    Channel reasoning_{OutputEvent::Kind::Reasoning, {}, {}, false}, content_{OutputEvent::Kind::Content, {}, {}, false};
    std::string call_body_;
    // A streamed call: where in call_body_ it is, what it waits for (Header: "<function="; FnName: the function's
    // name; Next: a parameter or "</function>"; Name: a parameter's name; Value: its value; Done: nothing but white
    // space; Broken: not well-formed, the rest is dropped), and the parameter being read. NoHeader: not a call it can
    // stream - the buffered path decides at </tool_call>. Header, FnName and NoHeader have sent nothing.
    enum class CallState { Header, FnName, NoHeader, Next, Name, Value, Done, Broken } call_state_ = CallState::Header;
    size_t call_at_ = 0, value_start_ = 0, value_sent_ = 0;
    size_t name_scan_ = 0, value_scan_ = 0;  // no '>' / newline, no "</parameter>" starts before
    bool value_started_ = false, value_string_ = false, first_param_ = true;
    std::string call_name_, param_name_;
    std::unordered_set<std::string> params_seen_;
    int64_t malformed_calls_ = 0;
    int64_t reasoning_tokens_ = 0, tool_calls_ = 0;
    bool dropped_call_ = false, stopped_ = false;
    size_t max_stop_ = 0;
    std::mt19937_64 rng_;
};

}  // namespace strix
