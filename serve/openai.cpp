#include "serve/openai.hpp"

#include "common/check.hpp"

#include <cmath>

namespace strix {

namespace {

[[noreturn]] void bad(const std::string &param, const std::string &msg) { throw ApiError(400, msg, param); }

// Typed field access that turns a wrong type into a 400 naming the field.
const json::Value *field(const json::Value &body, const char *name) {
    const json::Value *v = body.find(name);
    return v && !v->is_null() ? v : nullptr;
}
bool get_bool(const json::Value &v, const char *name) {
    if (!v.is_bool()) bad(name, std::string("'") + name + "' must be a boolean, got " + json::type_name(v.type()));
    return v.as_bool(name);
}
double get_number(const json::Value &v, const char *name) {
    if (!v.is_number()) bad(name, std::string("'") + name + "' must be a number, got " + json::type_name(v.type()));
    return v.as_double(name);
}
int64_t get_int(const json::Value &v, const char *name) {
    if (v.is_number() && !v.is_int()) {
        const double d = v.as_double(name);
        if (d == std::floor(d) && std::fabs(d) < 9e15) return (int64_t)d;  // 100.0 is an integer too
    }
    if (!v.is_int()) bad(name, std::string("'") + name + "' must be an integer, got " + json::type_name(v.type()));
    try {
        return v.as_int(name);
    } catch (const Error &) {
        bad(name, std::string("'") + name + "' is out of range");
    }
}
const std::string &get_string(const json::Value &v, const char *name) {
    if (!v.is_string()) bad(name, std::string("'") + name + "' must be a string, got " + json::type_name(v.type()));
    return v.as_string(name);
}

std::string effort_of(const std::string &e, const char *param) {
    if (e == "xhigh" || e == "high") return "xhigh";
    if (e == "medium") return "medium";
    if (e == "low" || e == "minimal") return "low";
    bad(param, "'" + std::string(param) + "' is '" + e + "'; supported: xhigh, high (= xhigh), medium, low, minimal (= low), none (thinking off)");
}

void check_tools(const json::Value &tools) {
    if (!tools.is_array()) bad("tools", std::string("'tools' must be an array, got ") + json::type_name(tools.type()));
    const auto &ts = tools.as_array("tools");
    for (size_t k = 0; k < ts.size(); ++k) {
        const std::string p = "tools[" + std::to_string(k) + "]";
        const json::Value &t = ts[k];
        if (!t.is_object()) bad(p, "'" + p + "' must be an object");
        const json::Value *type = t.find("type");
        if (!type || !type->is_string() || type->as_string("type") != "function")
            bad(p + ".type", "'" + p + ".type' must be \"function\"");
        const json::Value *fn = t.find("function");
        if (!fn || !fn->is_object()) bad(p + ".function", "'" + p + ".function' must be an object");
        const json::Value *name = fn->find("name");
        if (!name || !name->is_string() || name->as_string("name").empty())
            bad(p + ".function.name", "'" + p + ".function.name' must be a non-empty string");
        const json::Value *params = fn->find("parameters");
        if (params && !params->is_null() && !params->is_object())
            bad(p + ".function.parameters", "'" + p + ".function.parameters' must be an object");
    }
}

// Several leading system messages (opencode sends two) become one, their texts joined by a blank line - the
// template accepts a single system message, first. Content may be a string or text parts.
json::Value merge_leading_system(const json::Value &messages) {
    const auto &ms = messages.as_array("messages");
    size_t n = 0;
    while (n < ms.size() && ms[n].is_object() && ms[n].find("role") && ms[n].find("role")->is_string() &&
           ms[n].find("role")->as_string("role") == "system")
        ++n;
    if (n < 2) return messages;
    std::string text;
    for (size_t k = 0; k < n; ++k) {
        const json::Value *c = ms[k].find("content");
        std::string part;
        if (c && c->is_string()) part = c->as_string("content");
        else if (c && c->is_array())
            for (const json::Value &p : c->as_array("content")) {
                const json::Value *t = p.is_object() ? p.find("text") : nullptr;
                if (!t || !t->is_string()) bad("messages[" + std::to_string(k) + "].content", "system content parts must be text");
                part += t->as_string("text");
            }
        else if (c && !c->is_null()) bad("messages[" + std::to_string(k) + "].content", "system content must be a string or text parts");
        if (part.empty()) continue;
        if (!text.empty()) text += "\n\n";
        text += part;
    }
    json::Value out = json::Value::array();
    json::Value sys = json::Value::object();
    sys.set("role", json::Value::string("system"));
    sys.set("content", json::Value::string(text));
    out.push(std::move(sys));
    for (size_t k = n; k < ms.size(); ++k) out.push(ms[k]);
    return out;
}

// The fields /v1/chat/completions and /v1/completions share: stream, max_tokens, sampling, stop, and the known
// fields that are refused rather than ignored. R has stream, max_tokens, sampling, seeded, stop.
template <typename R> void parse_common(const json::Value &body, R &r, bool legacy_completions) {
    if (const json::Value *v = field(body, "stream")) r.stream = get_bool(*v, "stream");
    const json::Value *mt = field(body, "max_completion_tokens");
    if (!mt) mt = field(body, "max_tokens");
    if (mt) {
        r.max_tokens = get_int(*mt, "max_tokens");
        if (*r.max_tokens < 1) bad("max_tokens", "'max_tokens' must be >= 1, got " + std::to_string(*r.max_tokens));
    }
    if (const json::Value *v = field(body, "temperature")) {
        r.sampling.temperature = get_number(*v, "temperature");
        if (!(r.sampling.temperature >= 0 && r.sampling.temperature <= 2)) bad("temperature", "'temperature' must be in [0, 2]");
    }
    if (const json::Value *v = field(body, "top_p")) {
        r.sampling.top_p = get_number(*v, "top_p");
        if (!(r.sampling.top_p > 0 && r.sampling.top_p <= 1)) bad("top_p", "'top_p' must be in (0, 1]");
    }
    if (const json::Value *v = field(body, "top_k")) {
        r.sampling.top_k = get_int(*v, "top_k");
        if (r.sampling.top_k < 0) bad("top_k", "'top_k' must be >= 0 (0 = off)");
    }
    if (const json::Value *v = field(body, "seed")) r.sampling.seed = (uint64_t)get_int(*v, "seed"), r.seeded = true;
    if (const json::Value *v = field(body, "stop")) {
        if (v->is_string()) r.stop.push_back(v->as_string("stop"));
        else if (v->is_array())
            for (const json::Value &s : v->as_array("stop")) r.stop.push_back(get_string(s, "stop"));
        else bad("stop", "'stop' must be a string or an array of strings");
        if (r.stop.size() > 16) bad("stop", "at most 16 stop strings");
        for (const std::string &s : r.stop)
            if (s.empty()) bad("stop", "stop strings must be non-empty");
    }
    // Known but unsupported: refused rather than silently ignored.
    if (const json::Value *v = field(body, "n"); v && get_int(*v, "n") != 1) bad("n", "only n = 1 is supported");
    if (const json::Value *v = field(body, "logprobs")) {
        // Chat: a boolean; legacy completions: the number of top logprobs per token (0 = none).
        if (legacy_completions ? get_int(*v, "logprobs") != 0 : get_bool(*v, "logprobs")) bad("logprobs", "logprobs are not supported");
    }
    if (field(body, "top_logprobs")) bad("top_logprobs", "logprobs are not supported");
    for (const char *pen : {"presence_penalty", "frequency_penalty", "repetition_penalty"})
        if (const json::Value *v = field(body, pen); v && get_number(*v, pen) != (std::string(pen) == "repetition_penalty" ? 1.0 : 0.0))
            bad(pen, std::string("'") + pen + "' is not supported");
    if (const json::Value *v = field(body, "logit_bias"); v && !(v->is_object() && v->as_object("logit_bias").empty()))
        bad("logit_bias", "'logit_bias' is not supported");
}

}  // namespace

std::string error_body(const std::string &message, const std::string &type, const std::string &param) {
    json::Value e = json::Value::object();
    e.set("message", json::Value::string(message));
    e.set("type", json::Value::string(type));
    e.set("param", param.empty() ? json::Value() : json::Value::string(param));
    e.set("code", json::Value());
    json::Value b = json::Value::object();
    b.set("error", std::move(e));
    return b.dump();
}

ChatRequest parse_chat_request(const json::Value &body, const SamplingParams &defaults) {
    if (!body.is_object()) bad("", std::string("the request body must be a JSON object, got ") + json::type_name(body.type()));
    ChatRequest r;
    r.sampling = defaults;
    const json::Value *messages = field(body, "messages");
    if (!messages || !messages->is_array() || messages->as_array("messages").empty())
        bad("messages", "'messages' must be a non-empty array");
    r.messages = merge_leading_system(*messages);
    for (size_t k = 0; k < r.messages.as_array("messages").size(); ++k) {
        const json::Value &m = r.messages.as_array("messages")[k];
        const json::Value *c = m.is_object() ? m.find("content") : nullptr;
        if (c && c->is_array())
            for (const json::Value &part : c->as_array("content")) {
                const json::Value *type = part.is_object() ? part.find("type") : nullptr;
                if (type && type->is_string() && type->as_string("type") != "text")
                    bad("messages[" + std::to_string(k) + "].content",
                        "content part type '" + type->as_string("type") + "' is not supported (text only; no vision)");
            }
    }
    parse_common(body, r, /*legacy_completions=*/false);
    // Structured output: the shape is checked here, the schema itself where it is
    // compiled (HttpServer: a 400 naming the keyword it can't enforce). An unknown type stays a 400 worded the way
    // OpenAI-compatible clients recognize an unsupported parameter ("... is not supported") - never unconstrained.
    if (const json::Value *v = field(body, "response_format")) {
        if (!v->is_object()) bad("response_format", std::string("'response_format' must be an object, got ") + json::type_name(v->type()));
        const json::Value *type = v->find("type");
        const std::string kind = type && type->is_string() ? type->as_string("type") : std::string("(none)");
        if (kind == "text") {
            r.response_format = ChatRequest::ResponseFormat::Text;
        } else if (kind == "json_object") {
            r.response_format = ChatRequest::ResponseFormat::JsonObject;
        } else if (kind == "json_schema") {
            const json::Value *spec = v->find("json_schema");
            if (!spec || !spec->is_object())
                bad("response_format.json_schema", "response_format type 'json_schema' needs a 'json_schema' object "
                                                   "({\"name\": ..., \"schema\": {...}})");
            const json::Value *schema = spec->find("schema");
            if (!schema || !(schema->is_object() || schema->is_bool()))
                bad("response_format.json_schema.schema", "response_format.json_schema needs a 'schema' (an object)");
            if (const json::Value *strict = spec->find("strict"); strict && !strict->is_null())
                get_bool(*strict, "response_format.json_schema.strict");
            if (const json::Value *name = spec->find("name"); name && !name->is_null())
                r.response_schema_name = get_string(*name, "response_format.json_schema.name");
            r.response_format = ChatRequest::ResponseFormat::JsonSchema;
            r.response_schema = *schema;
        } else {
            bad("response_format", "response_format type '" + kind + "' is not supported (text, json_object, json_schema)");
        }
    }
    // Tools.
    std::string tool_choice = "auto";
    if (const json::Value *v = field(body, "tool_choice")) {
        if (!v->is_string() || (v->as_string("tool_choice") != "auto" && v->as_string("tool_choice") != "none"))
            bad("tool_choice", "tool_choice " + (v->is_string() ? "'" + v->as_string("tool_choice") + "'" : std::string("(a named tool)")) +
                                   " is not supported (no constrained decoding; 'tool_choice' must be \"auto\" or \"none\")");
        tool_choice = v->as_string("tool_choice");
    }
    if (const json::Value *v = field(body, "tools")) {
        check_tools(*v);
        if (tool_choice == "auto" && !v->as_array("tools").empty()) r.tools = *v;
    }
    // A JSON answer and tool calls exclude each other: the grammar holds the whole answer, so the model couldn't call
    // a tool - refused rather than tools silently dropped.
    if (r.response_format != ChatRequest::ResponseFormat::Text && !r.tools.is_null())
        bad("response_format", "response_format json_object / json_schema together with tools is not supported (the "
                               "answer is either a JSON document or tool calls) - send no tools, or tool_choice \"none\"");
    // Thinking budget: any of four spellings; two that disagree are refused, not silently resolved.
    {
        std::vector<std::pair<std::string, int64_t>> budgets;
        const auto take = [&](const json::Value *v, const std::string &name) {
            if (!v || v->is_null()) return;
            const int64_t b = get_int(*v, name.c_str());
            if (b < 0) bad(name, "'" + name + "' must be >= 0, got " + std::to_string(b));
            budgets.emplace_back(name, b);
        };
        take(field(body, "max_thinking_tokens"), "max_thinking_tokens");
        if (const json::Value *o = field(body, "reasoning"); o && o->is_object()) take(o->find("max_tokens"), "reasoning.max_tokens");
        if (const json::Value *o = field(body, "thinking"); o && o->is_object()) take(o->find("budget_tokens"), "thinking.budget_tokens");
        if (const json::Value *o = field(body, "chat_template_kwargs"); o && o->is_object())
            take(o->find("thinking_budget"), "chat_template_kwargs.thinking_budget");
        for (const auto &[name, b] : budgets)
            if (b != budgets[0].second)
                bad(name, "thinking budgets disagree: '" + budgets[0].first + "' = " + std::to_string(budgets[0].second) +
                              ", '" + name + "' = " + std::to_string(b));
        if (!budgets.empty()) r.thinking_budget = budgets[0].second;
    }
    // Thinking on / off: three spellings clients use (smoke test 2026-09-29: the Anthropic- and vLLM-style ones were
    // silently ignored) - chat_template_kwargs.enable_thinking, top-level enable_thinking, thinking.type
    // ("enabled" / "disabled") - plus reasoning_effort "none" (OpenAI's "no reasoning"; Hermes Agent sends it for its
    // auxiliary calls - 7 of them were 400s on 2026-10-03, its title retry among them). Two that disagree are refused,
    // as for the budgets.
    std::vector<std::pair<std::string, bool>> on_off;
    // Reasoning effort: a level, or "none" = thinking off.
    const auto effort = [&](const json::Value &v, const char *param) {
        const std::string e = get_string(v, param);
        if (e == "none") on_off.emplace_back(param, false);
        else r.template_options.reasoning_effort = effort_of(e, param);
    };
    if (const json::Value *v = field(body, "reasoning_effort")) effort(*v, "reasoning_effort");
    if (const json::Value *v = field(body, "enable_thinking"); v && !v->is_null())
        on_off.emplace_back("enable_thinking", get_bool(*v, "enable_thinking"));
    if (const json::Value *o = field(body, "thinking"); o && o->is_object())
        if (const json::Value *t = o->find("type"); t && !t->is_null()) {
            const std::string type = get_string(*t, "thinking.type");
            if (type != "enabled" && type != "disabled")
                bad("thinking.type", "'thinking.type' must be \"enabled\" or \"disabled\", got \"" + type + "\"");
            on_off.emplace_back("thinking.type", type == "enabled");
        }
    if (const json::Value *k = field(body, "chat_template_kwargs")) {
        if (!k->is_object()) bad("chat_template_kwargs", "'chat_template_kwargs' must be an object");
        for (const auto &[key, v] : k->as_object("chat_template_kwargs")) {
            if (v.is_null()) continue;
            if (key == "enable_thinking") on_off.emplace_back("chat_template_kwargs.enable_thinking", get_bool(v, "chat_template_kwargs.enable_thinking"));
            else if (key == "preserve_thinking") r.template_options.preserve_thinking = get_bool(v, "chat_template_kwargs.preserve_thinking");
            else if (key == "thinking_budget") continue;  // read above with the other budget fields
            else if (key == "reasoning_effort") effort(v, "chat_template_kwargs.reasoning_effort");
            else bad("chat_template_kwargs." + key, "chat_template_kwargs '" + key + "' is not supported (enable_thinking, preserve_thinking, reasoning_effort)");
        }
    }
    for (const auto &[name, on] : on_off)
        if (on != on_off[0].second)
            bad(name, "thinking on / off disagree: '" + on_off[0].first + "' says " + (on_off[0].second ? "on" : "off") +
                          ", '" + name + "' says " + (on ? "on" : "off"));
    if (!on_off.empty()) r.template_options.enable_thinking = on_off[0].second;
    if (const json::Value *v = field(body, "model"); v && v->is_string()) r.model = v->as_string("model");
    return r;
}

bool thinking_requested(const ChatRequest &r) {
    return (r.template_options.enable_thinking.has_value() && *r.template_options.enable_thinking) ||
           r.template_options.reasoning_effort.has_value() || r.thinking_budget.has_value();
}

bool thinking_left_out(const ChatRequest &r, int64_t max_tokens) {
    const bool thinking = !(r.template_options.enable_thinking.has_value() && !*r.template_options.enable_thinking);
    return thinking && !thinking_requested(r) && thinking_budget_for(r, max_tokens) == 0;
}

int64_t thinking_budget_for(const ChatRequest &r, int64_t max_tokens) {
    if (r.template_options.enable_thinking.has_value() && !*r.template_options.enable_thinking) return -1;
    if (r.thinking_budget.has_value()) return *r.thinking_budget;
    // The answer room: 1024 tokens by default (halogen 0.11's), but only kExplicitAnswerRoom when the request asked for
    // thinking - then a 600-token cap still thinks (~510) and answers (~90, the ~25-token stop text included).
    const int64_t floor = thinking_requested(r) ? kExplicitAnswerRoom : 1024;
    const int64_t room = std::max<int64_t>(floor, max_tokens * 15 / 100);
    return std::max<int64_t>(0, max_tokens - room);
}

int64_t clamp_max_tokens(const std::optional<int64_t> &asked, int64_t prompt_tokens, int64_t capacity, bool &clamped) {
    const int64_t room = capacity - prompt_tokens;
    if (room < 1)
        throw ApiError(400, "the prompt is " + std::to_string(prompt_tokens) + " tokens; the context holds " +
                                std::to_string(capacity) + " (prompt + at least 1 generated token)",
                       "messages", "context_length_exceeded");
    clamped = !asked.has_value() || *asked > room;
    return asked.has_value() ? std::min(*asked, room) : room;
}

json::Value usage_json(const GenerationResult &r) {
    json::Value u = json::Value::object();
    u.set("prompt_tokens", json::Value::integer(r.prompt_tokens));
    u.set("completion_tokens", json::Value::integer(r.completion_tokens));
    u.set("total_tokens", json::Value::integer(r.prompt_tokens + r.completion_tokens));
    json::Value pd = json::Value::object();
    pd.set("cached_tokens", json::Value::integer(r.cached_tokens));
    u.set("prompt_tokens_details", std::move(pd));
    json::Value cd = json::Value::object();
    cd.set("reasoning_tokens", json::Value::integer(r.reasoning_tokens));
    u.set("completion_tokens_details", std::move(cd));
    return u;
}

json::Value timings_json(const GenerationResult &r) {
    json::Value t = json::Value::object();
    const int64_t prefilled = r.prompt_tokens - r.cached_tokens;
    t.set("prompt_n", json::Value::integer(prefilled));
    t.set("prompt_ms", json::Value::number(r.prompt_ms));
    t.set("prompt_per_second", json::Value::number(r.prompt_ms > 0 ? prefilled / (r.prompt_ms / 1e3) : 0.0));
    t.set("predicted_n", json::Value::integer(r.completion_tokens));
    t.set("predicted_ms", json::Value::number(r.decode_ms));
    t.set("predicted_per_second",
          json::Value::number(r.decode_ms > 0 && r.completion_tokens > 1 ? (r.completion_tokens - 1) / (r.decode_ms / 1e3) : 0.0));
    t.set("cache_n", json::Value::integer(r.cached_tokens));
    t.set("queue_ms", json::Value::number(r.queue_ms));
    return t;
}

json::Value tool_call_start_delta(const ToolCallOut &c, int index) {
    json::Value fn = json::Value::object();
    fn.set("name", json::Value::string(c.name));
    fn.set("arguments", json::Value::string(""));
    json::Value tc = json::Value::object();
    tc.set("index", json::Value::integer(index));
    tc.set("id", json::Value::string(c.id));
    tc.set("type", json::Value::string("function"));
    tc.set("function", std::move(fn));
    return tc;
}

json::Value tool_call_args_delta(const std::string &piece, int index) {
    json::Value fn = json::Value::object();
    fn.set("arguments", json::Value::string(piece));
    json::Value tc = json::Value::object();
    tc.set("index", json::Value::integer(index));
    tc.set("function", std::move(fn));
    return tc;
}

json::Value tool_call_delta(const ToolCallOut &c, int index) {
    json::Value fn = json::Value::object();
    fn.set("name", json::Value::string(c.name));
    fn.set("arguments", json::Value::string(c.arguments.dump()));
    json::Value tc = json::Value::object();
    tc.set("index", json::Value::integer(index));
    tc.set("id", json::Value::string(c.id));
    tc.set("type", json::Value::string("function"));
    tc.set("function", std::move(fn));
    return tc;
}

namespace {
json::Value envelope(const ResponseMeta &m, const char *object) {
    json::Value b = json::Value::object();
    b.set("id", json::Value::string(m.id));
    b.set("object", json::Value::string(object));
    b.set("created", json::Value::integer(m.created));
    b.set("model", json::Value::string(m.model));
    return b;
}
json::Value strix_ext(const GenerationResult &r, bool clamped, const std::string &thinking) {
    json::Value s = json::Value::object();
    s.set("max_tokens_clamped", json::Value::boolean(clamped));
    if (r.dropped_partial_call) s.set("dropped_partial_tool_call", json::Value::boolean(true));
    if (r.malformed_tool_calls > 0) s.set("malformed_tool_calls", json::Value::integer(r.malformed_tool_calls));
    s.set("thinking_budget_hit", json::Value::boolean(r.thinking_budget_hit));
    if (r.think_nudges > 0) s.set("think_nudges", json::Value::integer(r.think_nudges));
    if (!thinking.empty()) s.set("thinking", json::Value::string(thinking));
    if (thinking == "skipped")
        s.set("thinking_skipped_reason",
              json::Value::string("max_tokens <= 1024 leaves no room to think; ask for thinking (enable_thinking / "
                                  "thinking.type / reasoning_effort) or a budget to keep it"));
    if (r.mtp_drafted_tokens > 0) {
        s.set("mtp_drafted_tokens", json::Value::integer(r.mtp_drafted_tokens));
        s.set("mtp_accepted_tokens", json::Value::integer(r.mtp_accepted_tokens));
        s.set("mtp_acceptance_rate", json::Value::number(100.0 * (double)r.mtp_accepted_tokens / (double)r.mtp_drafted_tokens));
    }
    return s;
}
}  // namespace

CompletionRequest parse_completion_request(const json::Value &body, const SamplingParams &defaults) {
    if (!body.is_object()) bad("", std::string("the request body must be a JSON object, got ") + json::type_name(body.type()));
    CompletionRequest r;
    r.sampling = defaults;
    const json::Value *p = field(body, "prompt");
    if (!p) bad("prompt", "'prompt' is required");
    if (p->is_string()) r.prompt = p->as_string("prompt");
    else if (p->is_array() && p->as_array("prompt").size() == 1 && p->as_array("prompt")[0].is_string())
        r.prompt = p->as_array("prompt")[0].as_string("prompt");
    else bad("prompt", "'prompt' must be a string (or an array of one string); token-id prompts and batches are not supported");
    if (r.prompt.empty()) bad("prompt", "'prompt' must not be empty");
    if (const json::Value *v = field(body, "model"); v && v->is_string()) r.model = v->as_string("model");
    parse_common(body, r, /*legacy_completions=*/true);
    if (const json::Value *v = field(body, "echo"); v && get_bool(*v, "echo")) bad("echo", "'echo' is not supported");
    if (const json::Value *v = field(body, "best_of"); v && get_int(*v, "best_of") != 1) bad("best_of", "only best_of = 1 is supported");
    if (const json::Value *v = field(body, "suffix"); v && !(v->is_string() && v->as_string("suffix").empty()))
        bad("suffix", "'suffix' (fill-in-the-middle) is not supported");
    return r;
}

std::string chunk_body(const ResponseMeta &m, const json::Value &delta, const char *finish_reason) {
    json::Value b = envelope(m, "chat.completion.chunk");
    json::Value choice = json::Value::object();
    choice.set("index", json::Value::integer(0));
    choice.set("delta", delta);
    choice.set("finish_reason", finish_reason ? json::Value::string(finish_reason) : json::Value());
    json::Value choices = json::Value::array();
    choices.push(std::move(choice));
    b.set("choices", std::move(choices));
    return b.dump();
}

std::string usage_chunk_body(const ResponseMeta &m, const GenerationResult &r, bool clamped) {
    json::Value b = envelope(m, "chat.completion.chunk");
    b.set("choices", json::Value::array());
    b.set("usage", usage_json(r));
    b.set("timings", timings_json(r));
    b.set("strix", strix_ext(r, clamped, m.thinking));
    return b.dump();
}

std::string completion_body(const ResponseMeta &m, const std::string &reasoning, const std::string &content,
                            const std::vector<ToolCallOut> &calls, const GenerationResult &r, bool clamped) {
    json::Value b = envelope(m, "chat.completion");
    json::Value msg = json::Value::object();
    msg.set("role", json::Value::string("assistant"));
    msg.set("content", content.empty() && !calls.empty() ? json::Value() : json::Value::string(content));
    if (!reasoning.empty()) msg.set("reasoning_content", json::Value::string(reasoning));
    if (!calls.empty()) {
        json::Value tcs = json::Value::array();
        for (size_t k = 0; k < calls.size(); ++k) {
            json::Value tc = tool_call_delta(calls[k], (int)k);
            tcs.push(std::move(tc));
        }
        msg.set("tool_calls", std::move(tcs));
    }
    json::Value choice = json::Value::object();
    choice.set("index", json::Value::integer(0));
    choice.set("message", std::move(msg));
    choice.set("finish_reason", json::Value::string(r.finish_reason));
    json::Value choices = json::Value::array();
    choices.push(std::move(choice));
    b.set("choices", std::move(choices));
    b.set("usage", usage_json(r));
    b.set("timings", timings_json(r));
    b.set("strix", strix_ext(r, clamped, m.thinking));
    return b.dump();
}

namespace {
json::Value text_choice(const std::string &text, const json::Value &finish_reason) {
    json::Value c = json::Value::object();
    c.set("index", json::Value::integer(0));
    c.set("text", json::Value::string(text));
    c.set("logprobs", json::Value());
    c.set("finish_reason", finish_reason);
    return c;
}
}  // namespace

std::string text_completion_body(const ResponseMeta &m, const std::string &text, const GenerationResult &r, bool clamped) {
    json::Value b = envelope(m, "text_completion");
    json::Value choices = json::Value::array();
    choices.push(text_choice(text, json::Value::string(r.finish_reason)));
    b.set("choices", std::move(choices));
    b.set("usage", usage_json(r));
    b.set("timings", timings_json(r));
    b.set("strix", strix_ext(r, clamped, m.thinking));
    return b.dump();
}

std::string text_chunk_body(const ResponseMeta &m, const std::string &text, const char *finish_reason) {
    json::Value b = envelope(m, "text_completion");
    json::Value choices = json::Value::array();
    choices.push(text_choice(text, finish_reason ? json::Value::string(finish_reason) : json::Value()));
    b.set("choices", std::move(choices));
    return b.dump();
}

std::string text_usage_chunk_body(const ResponseMeta &m, const GenerationResult &r, bool clamped) {
    json::Value b = envelope(m, "text_completion");
    b.set("choices", json::Value::array());
    b.set("usage", usage_json(r));
    b.set("timings", timings_json(r));
    b.set("strix", strix_ext(r, clamped, m.thinking));
    return b.dump();
}

}  // namespace strix
