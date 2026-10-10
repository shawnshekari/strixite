#pragma once

// The OpenAI Chat Completions surface: request validation into what the
// engine runs, and the response / SSE chunk bodies. Everything the contract doesn't support is refused with a 400
// naming the field (ApiError); fields it doesn't know are ignored.

#include "serve/chat_template.hpp"
#include "serve/engine.hpp"
#include "serve/json.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix {

// An error the client caused: HTTP status + OpenAI error body.
struct ApiError : std::runtime_error {
    int status;
    std::string type, param;
    ApiError(int status, const std::string &msg, std::string param = {}, std::string type = "invalid_request_error")
        : std::runtime_error(msg), status(status), type(std::move(type)), param(std::move(param)) {}
};
std::string error_body(const std::string &message, const std::string &type, const std::string &param);

struct ChatRequest {
    json::Value messages, tools;  // tools: null unless a non-empty array that will be rendered
    bool stream = false;
    std::string model;  // as asked (any name is served: logged, not refused)
    std::optional<int64_t> max_tokens;
    SamplingParams sampling;
    bool seeded = false;
    std::vector<std::string> stop;
    ChatTemplateOptions template_options;
    std::optional<int64_t> thinking_budget;  // explicit, from any of the four budget fields
    // response_format: text, json_object (any JSON object), or json_schema with
    // response_schema the schema (json_schema.schema; `strict` false is constrained like true, `name` only logged).
    enum class ResponseFormat { Text, JsonObject, JsonSchema };
    ResponseFormat response_format = ResponseFormat::Text;
    json::Value response_schema;
    std::string response_schema_name;
};

// A legacy /v1/completions request (for the Local LLM Benchmarks speed runner, which drives this endpoint): a raw
// prompt, tokenized as is (special-token text included), no chat template, no thinking or tool parsing. The shared
// fields (stream, max_tokens, sampling, stop, refusals) parse as for chat; also refused: echo, best_of != 1, suffix,
// logprobs > 0, token-id / batched prompts.
struct CompletionRequest {
    std::string prompt;
    bool stream = false;
    std::string model;
    std::optional<int64_t> max_tokens;
    SamplingParams sampling;
    bool seeded = false;
    std::vector<std::string> stop;
};
CompletionRequest parse_completion_request(const json::Value &body, const SamplingParams &defaults);

// The reasoning tokens to allow (GenerationRequest::thinking_budget): -1 with thinking off; the explicit budget if
// the request set one; else the answer room - leave max(1024, 15% of max_tokens) for the answer.
int64_t thinking_budget_for(const ChatRequest &r, int64_t max_tokens);
// True when thinking is on (not switched off), no budget was given, and the answer room leaves it none (max_tokens
// <= 1024): the prompt is then rendered with thinking off. Opening a think block only to force it
// shut spent ~25 completion tokens on Qwen's early-stop text - max_tokens <= 25 came back empty ("length").
bool thinking_left_out(const ChatRequest &r, int64_t max_tokens);
// True when the request asked for thinking in any way (enable_thinking true in any spelling, reasoning_effort, a
// budget): then it is never left out, and the answer room without a budget is kExplicitAnswerRoom, not 1024
// (second smoke test, 2026-09-29: thinking.type "enabled" at max_tokens 600 thought 0 tokens).
bool thinking_requested(const ChatRequest &r);
constexpr int64_t kExplicitAnswerRoom = 64;

// Validates body (already parsed JSON) against the contract. defaults: generation_config's sampling values.
ChatRequest parse_chat_request(const json::Value &body, const SamplingParams &defaults);

// max_tokens against the room the prompt leaves: returns the tokens to allow and sets clamped when the request
// asked for more (or for nothing, meaning "all the room"). Throws ApiError 400 when the prompt alone fills the
// context.
int64_t clamp_max_tokens(const std::optional<int64_t> &asked, int64_t prompt_tokens, int64_t capacity, bool &clamped);

// Response bodies. model: the id to report; created: unix seconds.
struct ResponseMeta {
    std::string id, model;
    int64_t created = 0;
    std::string thinking;  // strix.thinking: "on" | "off" | "skipped" (thinking_left_out); empty = not reported
};
// A streamed call (OutputParser::Config::stream_calls) as OpenAI streams one: the first delta with id, type, name
// and empty arguments, then deltas with only the index and the next piece of the arguments.
json::Value tool_call_start_delta(const ToolCallOut &c, int index);
json::Value tool_call_args_delta(const std::string &piece, int index);
std::string chunk_body(const ResponseMeta &m, const json::Value &delta, const char *finish_reason);
std::string usage_chunk_body(const ResponseMeta &m, const GenerationResult &r, bool clamped);
std::string completion_body(const ResponseMeta &m, const std::string &reasoning, const std::string &content,
                            const std::vector<ToolCallOut> &calls, const GenerationResult &r, bool clamped);
json::Value tool_call_delta(const ToolCallOut &c, int index);
// Legacy completions ("text_completion"): the whole response, a streamed text chunk (finish_reason null until the
// last), and the final usage chunk (empty choices, as chat's).
std::string text_completion_body(const ResponseMeta &m, const std::string &text, const GenerationResult &r, bool clamped);
std::string text_chunk_body(const ResponseMeta &m, const std::string &text, const char *finish_reason);
std::string text_usage_chunk_body(const ResponseMeta &m, const GenerationResult &r, bool clamped);
json::Value usage_json(const GenerationResult &r);
json::Value timings_json(const GenerationResult &r);

}  // namespace strix
