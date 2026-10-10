#include "serve/chat_template.hpp"

#include "common/check.hpp"
#include "serve/unicode.hpp"

namespace strix {

namespace {

constexpr const char *kXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
    "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
constexpr const char *kLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
    "without unnecessary elaboration.";
constexpr const char *kToolsIntro = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr const char *kToolsFormat =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
    "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
    "the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

bool starts_with(const std::string &s, const char *p) { return s.rfind(p, 0) == 0; }
bool ends_with(const std::string &s, const std::string &p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// The template's render_content macro, then |trim.
std::string content_of(const json::Value &msg, const std::string &path, bool is_system) {
    const json::Value *c = msg.find("content");
    if (!c || c->is_null()) return "";
    if (c->is_string()) return unicode::python_strip(c->as_string(path));
    STRIX_CHECK(c->is_array(), "chat template: '", path, "' must be a string, null or an array of parts, got ",
                json::type_name(c->type()), " (Unexpected content type.)");
    std::string out;
    const auto &parts = c->as_array(path);
    for (size_t k = 0; k < parts.size(); ++k) {
        const std::string pp = path + "[" + std::to_string(k) + "]";
        const json::Value &part = parts[k];
        STRIX_CHECK(part.is_object(), "chat template: '", pp, "' must be an object, got ", json::type_name(part.type()));
        const json::Value *type = part.find("type");
        const bool image = part.find("image") || part.find("image_url") || (type && type->is_string() && type->as_string(pp) == "image");
        const bool video = part.find("video") || (type && type->is_string() && type->as_string(pp) == "video");
        STRIX_CHECK(!(is_system && (image || video)), "chat template: '", pp, "': System message cannot contain images or videos.");
        STRIX_CHECK(!image && !video, "chat template: '", pp, "' is an image / video part; vision is not served (v1)");
        const json::Value *text = part.find("text");
        STRIX_CHECK(text != nullptr, "chat template: '", pp, "' has no 'text' (Unexpected item type in content.)");
        out += text->as_string(pp + ".text");
    }
    return unicode::python_strip(out);
}

const std::string &role_of(const json::Value &msg, const std::string &path) {
    STRIX_CHECK(msg.is_object(), "chat template: '", path, "' must be an object, got ", json::type_name(msg.type()));
    const json::Value *r = msg.find("role");
    STRIX_CHECK(r != nullptr, "chat template: '", path, "' has no 'role'");
    return r->as_string(path + ".role");
}

// A tool call's arguments as the object the template iterates: an object as is, a JSON string parsed (it must
// hold an object). nullptr = render no parameters: absent, the empty string, or an object's JSON left open the way
// a streamed call that was cut off or malformed leaves it ("}" or "\"}" would close it), sent back as received.
std::optional<json::Value> arguments_of(const json::Value &call, const std::string &path) {
    const json::Value *a = call.find("arguments");
    if (!a) return std::nullopt;
    if (a->is_string()) {
        const std::string &s = a->as_string(path);
        if (s.empty()) return std::nullopt;
        json::Value v;
        try {
            v = json::Value::parse(s);
        } catch (const Error &e) {
            for (const char *close : {"}", "\"}"}) {
                try {
                    if (json::Value::parse(s + close).is_object()) return std::nullopt;
                } catch (const Error &) {
                }
            }
            STRIX_FAIL("chat template: '", path, ".arguments' is a string but not JSON: ", e.what());
        }
        STRIX_CHECK(v.is_object(), "chat template: '", path, ".arguments' must hold a JSON object, got ",
                    json::type_name(v.type()));
        return v;
    }
    STRIX_CHECK(a->is_object(), "chat template: '", path, ".arguments' must be an object or a JSON string, got ",
                json::type_name(a->type()));
    return *a;
}

}  // namespace

std::string render_chat(const json::Value &messages, const json::Value *tools, const ChatTemplateOptions &opt) {
    const auto &msgs = messages.as_array("messages");
    STRIX_CHECK(!msgs.empty(), "chat template: No messages provided.");
    std::vector<std::string> roles;
    for (size_t k = 0; k < msgs.size(); ++k) roles.push_back(role_of(msgs[k], "messages[" + std::to_string(k) + "]"));

    std::string reasoning;
    if (!opt.enable_thinking.has_value() || *opt.enable_thinking) {
        const std::string effort = opt.reasoning_effort.value_or("xhigh");
        STRIX_CHECK(effort == "xhigh" || effort == "medium" || effort == "low", "chat template: Unexpected reasoning effort ",
                    effort, ". Supported types are xhigh (default), medium, and low.");
        if (effort == "xhigh") reasoning = kXhigh;
        else if (effort == "low") reasoning = kLow;
    }

    std::string out;
    const bool have_tools = tools && tools->is_array() && !tools->as_array("tools").empty();
    if (have_tools) {
        out += "<|im_start|>system\n";
        if (!reasoning.empty()) out += reasoning + "\n\n";
        out += kToolsIntro;
        for (const json::Value &tool : tools->as_array("tools")) out += "\n" + tool.dump_python();
        out += "\n</tools>";
        out += kToolsFormat;
        if (roles[0] == "system") {
            const std::string c = content_of(msgs[0], "messages[0].content", true);
            if (!c.empty()) out += "\n\n" + c;
        }
        out += "<|im_end|>\n";
    } else if (roles[0] == "system") {
        const std::string c = content_of(msgs[0], "messages[0].content", true);
        if (!c.empty()) out += "<|im_start|>system\n" + (reasoning.empty() ? "" : reasoning + "\n\n") + c + "<|im_end|>\n";
        else if (!reasoning.empty()) out += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
    } else if (!reasoning.empty()) {
        out += "<|im_start|>system\n" + reasoning + "<|im_end|>\n";
    }

    // The last real user query (a user message that isn't just wrapped tool responses).
    long last_query = -1;
    for (size_t k = msgs.size(); k-- > 0;) {
        if (roles[k] != "user") continue;
        const std::string c = content_of(msgs[k], "messages[" + std::to_string(k) + "].content", false);
        if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) {
            last_query = (long)k;
            break;
        }
    }
    STRIX_CHECK(last_query >= 0, "chat template: No user query found in messages.");

    for (size_t k = 0; k < msgs.size(); ++k) {
        const std::string path = "messages[" + std::to_string(k) + "]";
        const json::Value &m = msgs[k];
        const std::string &role = roles[k];
        const std::string content = content_of(m, path + ".content", role == "system");
        if (role == "system") {
            STRIX_CHECK(k == 0, "chat template: '", path, "': System message must be at the beginning.");
        } else if (role == "user") {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string reasoning_content;
            if (const json::Value *r = m.find("reasoning_content"); r && r->is_string())
                reasoning_content = unicode::python_strip(r->as_string(path));
            if (!opt.preserve_thinking.has_value() || *opt.preserve_thinking || (long)k > last_query)
                out += "<|im_start|>assistant\n<think>\n" + reasoning_content + "\n</think>\n\n" + content;
            else
                out += "<|im_start|>assistant\n" + content;
            const json::Value *calls = m.find("tool_calls");
            if (calls && calls->is_array()) {
                const auto &cs = calls->as_array(path + ".tool_calls");
                for (size_t j = 0; j < cs.size(); ++j) {
                    const std::string cp = path + ".tool_calls[" + std::to_string(j) + "]";
                    STRIX_CHECK(cs[j].is_object(), "chat template: '", cp, "' must be an object");
                    const json::Value *fn = cs[j].find("function");
                    const json::Value &call = fn ? *fn : cs[j];
                    const std::string fp = fn ? cp + ".function" : cp;
                    STRIX_CHECK(call.is_object(), "chat template: '", fp, "' must be an object");
                    const json::Value *name = call.find("name");
                    STRIX_CHECK(name != nullptr, "chat template: '", fp, "' has no 'name'");
                    const std::string &nm = name->as_string(fp + ".name");
                    if (j == 0) out += (content.empty() ? "" : "\n\n");
                    else out += "\n";
                    out += "<tool_call>\n<function=" + nm + ">\n";
                    if (const auto args = arguments_of(call, fp))
                        for (const auto &[an, av] : args->as_object(fp + ".arguments"))
                            out += "<parameter=" + an + ">\n" + (av.is_string() ? av.as_string(an) : av.dump_python()) +
                                   "\n</parameter>\n";
                    out += "</function>\n</tool_call>";
                }
            } else {
                STRIX_CHECK(!calls || calls->is_null(), "chat template: '", path, ".tool_calls' must be an array, got ",
                            json::type_name(calls->type()));
            }
            out += "<|im_end|>\n";
        } else if (role == "tool") {
            if (k > 0 && roles[k - 1] != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (k + 1 == msgs.size() || roles[k + 1] != "tool") out += "<|im_end|>\n";
        } else {
            STRIX_FAIL("chat template: '", path, ".role' is '", role, "': Unexpected message role.");
        }
    }
    if (opt.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        out += opt.enable_thinking.has_value() && !*opt.enable_thinking ? "<think>\n\n</think>\n\n" : "<think>\n";
    }
    return out;
}

}  // namespace strix
