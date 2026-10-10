#pragma once

// Qwen3.8-Flash-Next's chat template (the checkpoint's chat_template.jinja), rendered by hand in C++ - no Jinja
// engine. Output is byte-identical to transformers' apply_chat_template
// for the same messages / tools / options (tests/test_chat_template.cpp against
// reference/golden_chat_template_qwen4exp.py), including its `trim` (Python str.strip) and `tojson`
// (json.dumps, ensure_ascii=False) semantics. What the template would reject (no messages, a system message not
// first, no user query, an unknown role or content type, a bad reasoning effort) throws strix::Error with the
// template's reason; so do the shapes it would render nonsensically from (non-string role, non-object parts).
//
// Messages are OpenAI-shaped: {role, content (string | null | [{type: "text", text}]), reasoning_content?,
// tool_calls?: [{function: {name, arguments}}]}; `arguments` may be an object or a JSON string holding one (as
// OpenAI clients send it) - an empty string renders no parameters. Image / video parts are refused (vision is
// cut from v1).

#include "serve/json.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strix {

struct ChatTemplateOptions {
    bool add_generation_prompt = true;
    std::optional<bool> enable_thinking;          // unset = the template's default (thinking on)
    std::optional<bool> preserve_thinking;        // unset = on (earlier assistant turns keep their reasoning)
    std::optional<std::string> reasoning_effort;  // "xhigh" | "medium" | "low"; unset = xhigh
};

// tools: nullptr or the request's `tools` array (rendered only when it's a non-empty array).
// plain_spans (optional): set to the byte ranges of the rendered text that came from the request rather than the
// template - system / user / tool message content, tool-call argument values, the tool schemas - for
// Tokenizer::encode(text, spans): a special token's text there is the user's or a file's text, not a control token
// (an opencode tool result holding "<|im_end|>" otherwise ended up as a real end of turn in the prompt). Assistant
// content and reasoning are left out: the model wrote those, and a special token it emitted there must re-encode
// as that token (the session / prompt cache resume on the exact ids).
std::string render_chat(const json::Value &messages, const json::Value *tools, const ChatTemplateOptions &opt,
                        std::vector<std::pair<size_t, size_t>> *plain_spans = nullptr);

}  // namespace strix
