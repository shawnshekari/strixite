#include "serve/http_server.hpp"

#include "common/check.hpp"
#include "serve/chat_template.hpp"
#include "serve/log.hpp"
#include "serve/openai.hpp"
#include <httplib.h>  // third_party/cpp-httplib (vendored, SYSTEM include)

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <optional>
#include <random>

namespace strix {

namespace {

int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string random_id(const char *prefix) {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    static const char *kAlnum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string s = prefix;
    for (int k = 0; k < 24; ++k) s += kAlnum[rng() % 62];
    return s;
}

// The engine's events for one request, handed to the HTTP thread through a queue.
class QueueSink : public GenerationSink {
public:
    struct Item {
        enum class Kind { Start, Progress, Events, Done } kind;
        PrefillProgress progress;
        std::vector<OutputEvent> events;
        GenerationResult result;
    };
    void on_start() override { push({Item::Kind::Start, {}, {}, {}}); }
    void on_prefill(const PrefillProgress &p) override { push({Item::Kind::Progress, p, {}, {}}); }
    void on_events(std::vector<OutputEvent> &events) override {
        push({Item::Kind::Events, {}, std::move(events), {}});
        events.clear();
    }
    void on_done(const GenerationResult &r) override { push({Item::Kind::Done, {}, {}, r}); }
    bool cancelled() override { return cancelled_.load(); }
    void cancel() { cancelled_ = true; }

    // Waits up to `timeout` for the next item; false on timeout.
    bool pop(Item &out, double timeout_s) {
        std::unique_lock<std::mutex> lock(mu_);
        if (!cv_.wait_for(lock, std::chrono::duration<double>(timeout_s), [&] { return !q_.empty(); })) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

private:
    void push(Item it) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            q_.push_back(std::move(it));
        }
        cv_.notify_one();
    }
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> q_;
    std::atomic<bool> cancelled_{false};
};

void send_error(httplib::Response &res, int status, const std::string &msg, const std::string &type = "invalid_request_error",
                const std::string &param = {}) {
    res.status = status;
    res.set_content(error_body(msg, type, param), "application/json");
}

// A request answered with an error before reaching the engine (no block - one line): 4xx the client's (warning),
// 5xx the server's (error).
void log_error(int64_t id, const httplib::Request &req, int status, const std::string &msg) {
    slog(status >= 500 ? LogLevel::Error : LogLevel::Warning, "Request %lld failed: %s %s from %s: %d - %s", (long long)id,
         req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), status, msg.c_str());
    slog(LogLevel::Info, "\xc2\xa0");  // a blank line after each request
}

// The top rows of a request's log block (GenerationRequest::log_header): where it came from and what it asked.
std::vector<std::pair<std::string, std::string>> header_rows(const httplib::Request &req, bool stream,
                                                             const std::string &model) {
    char from[256];
    std::snprintf(from, sizeof from, "%s, %s %s, %s KB, %s", req.remote_addr.c_str(), req.method.c_str(),
                  req.path.c_str(), fmt_rate(req.body.size() / 1e3, 1).c_str(), stream ? "streamed" : "not streamed");
    // The model asked for is only logged: any name is served, and responses carry the real one.
    return {{"from", from}, {"model", model.empty() ? "none requested" : "\"" + model + "\" requested"}};
}

// The max_tokens row: what the request gets, and why when that isn't what it asked for.
std::string max_tokens_text(const std::optional<int64_t> &asked, int64_t granted) {
    if (!asked) return fmt_n(granted) + " (none asked: the room the context leaves)";
    if (*asked > granted)
        return fmt_n(granted) + " (asked " + fmt_n(*asked) + ", clamped to the room the context leaves)";
    return fmt_n(granted);
}

// The sampling row: the values the request decodes with, and which of them it set itself (the rest are the server's
// defaults: generation_config.json or the conf) - so a report of odd output says what the client sent.
std::string sampling_text(const SamplingParams &s, bool seeded, const std::vector<std::string> &asked) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "temperature %g, top_p %g, top_k %lld", s.temperature, s.top_p, (long long)s.top_k);
    std::string out = buf;
    if (seeded) out += ", seed " + std::to_string(s.seed);
    if (asked.empty()) return out + " (none asked: server defaults)";
    std::string names;
    for (const std::string &a : asked) names += (names.empty() ? "" : ", ") + a;
    return out + " (asked: " + names + ")";
}

}  // namespace

HttpServer::HttpServer(Engine &engine, const Tokenizer &tok, ServerConfig cfg)
    : engine_(engine), tok_(tok), cfg_(std::move(cfg)), started_(unix_now()), svr_(std::make_unique<httplib::Server>()) {
    STRIX_CHECK(cfg_.port >= 0 && cfg_.port <= 65535, "HttpServer: port ", cfg_.port);
    STRIX_CHECK(cfg_.keepalive_s > 0, "HttpServer: keepalive_s ", cfg_.keepalive_s);
    STRIX_CHECK(cfg_.request_read_timeout_s >= 1 && cfg_.request_read_timeout_s <= 3600,
                "HttpServer: request_read_timeout_s ", cfg_.request_read_timeout_s, ", expected 1..3600");
    // The sampling defaults (generation_config.json) seed every request that doesn't set them: a bad one fails
    // startup here, naming it, instead of every such request in the Sampler.
    try {
        Sampler probe(cfg_.defaults, 1);
    } catch (const std::exception &e) {
        STRIX_CHECK(false, "HttpServer: the sampling defaults (temperature ", cfg_.defaults.temperature, ", top_k ",
                    cfg_.defaults.top_k, ", top_p ", cfg_.defaults.top_p, ") are invalid: ", e.what());
    }
    svr_->set_payload_max_length(cfg_.max_body_bytes);
    // The default 5 s is httplib's, sized for local APIs: a multi-MB history over a weak Wi-Fi uplink can pause
    // longer than that mid-body (strixite #15).
    svr_->set_read_timeout(cfg_.request_read_timeout_s, 0);
    routes();
}

HttpServer::~HttpServer() { stop(); }

int HttpServer::bind() {
    const int port = cfg_.port == 0 ? svr_->bind_to_any_port(cfg_.host) : (svr_->bind_to_port(cfg_.host, cfg_.port) ? cfg_.port : -1);
    STRIX_CHECK(port > 0, "HttpServer: can't bind ", cfg_.host, ":", cfg_.port);
    return port;
}

void HttpServer::serve() { svr_->listen_after_bind(); }
void HttpServer::stop() {
    if (svr_) svr_->stop();
}

void HttpServer::routes() {
    svr_->Post("/v1/chat/completions", [this](const httplib::Request &req, httplib::Response &res) {
        const int64_t id = ++next_id_;
        ChatRequest cr;
        GenerationRequest gen;
        bool thinking_skipped = false, thinking_defaulted_off = false;
        bool clamped = false;
        std::string format_row;  // the response_format row of the log block, if one was asked
        std::string vision_note;  // the "vision" row of the log block, if the request carried images
        try {
            json::Value body;
            try {
                body = json::Value::parse(req.body);
            } catch (const Error &e) {
                throw ApiError(400, std::string("request body: ") + e.what());
            }
            cr = parse_chat_request(body, cfg_.defaults, /*vision_enabled=*/cfg_.vision != nullptr);
            // The server's thinking default (config `thinking`) for a request that doesn't say either way.
            if (!cfg_.thinking_default && !cr.template_options.enable_thinking.has_value() && !thinking_requested(cr)) {
                cr.template_options.enable_thinking = false;
                thinking_defaulted_off = true;
            }
            // Structured output: the response format compiled (or found compiled) here, on the HTTP thread; a schema
            // the compiler can't enforce is a 400 naming the keyword and where it is.
            if (cr.response_format != ChatRequest::ResponseFormat::Text) {
                const int64_t misses = grammars_.misses();
                try {
                    gen.grammar = cr.response_format == ChatRequest::ResponseFormat::JsonObject
                                      ? grammars_.any_object()
                                      : grammars_.schema(cr.response_schema, "response_format.json_schema.schema");
                } catch (const Error &e) {
                    // The check's own prefix (function, file:line, condition) goes to the log; the client gets the reason.
                    const std::string full = e.what();
                    const size_t at = full.find("failed: ");
                    throw ApiError(400, at == std::string::npos ? full : full.substr(at + 8), "response_format");
                }
                format_row = std::string(cr.response_format == ChatRequest::ResponseFormat::JsonObject ? "json_object"
                                                                                                        : "json_schema") +
                             (cr.response_schema_name.empty() ? "" : " '" + cr.response_schema_name + "'") + " (grammar " +
                             (grammars_.misses() > misses ? "compiled" : "cached") + ", " + fmt_n(grammars_.size()) +
                             " cached)";
            }
            std::string prompt;
            std::vector<std::pair<size_t, size_t>> plain_spans;  // what the request wrote: never control tokens
            const bool vision = !cr.images.empty();  // image parts (parsed only when the server has a tower)
            try {
                prompt = render_chat(cr.messages, cr.tools.is_null() ? nullptr : &cr.tools, cr.template_options,
                                     &plain_spans, vision);
            } catch (const Error &e) {
                throw ApiError(400, e.what(), "messages");
            }
            gen.prompt = tok_.encode(prompt, plain_spans);
            // A user message last (not a tool result): the disk cache keeps this turn as a checkpoint.
            if (cr.messages.is_array() && !cr.messages.as_array("messages").empty()) {
                const json::Value &last = cr.messages.as_array("messages").back();
                const json::Value *role = last.find("role"), *content = last.find("content");
                const bool tool_results = content && content->is_string() &&
                                          content->as_string("content").rfind("<tool_response>", 0) == 0;
                gen.user_turn = role && role->is_string() && role->as_string("role") == "user" && !tool_results;
                // One message, no tools: nothing a later turn extends - its cache entries stay in RAM only.
                gen.one_shot = cr.messages.as_array("messages").size() == 1 &&
                               (cr.tools.is_null() || (cr.tools.is_array() && cr.tools.as_array("tools").empty()));
            }
            gen.max_tokens = clamp_max_tokens(cr.max_tokens, (int64_t)gen.prompt.size(), engine_.capacity(), clamped);
            // No room to think (max_tokens <= the answer room, no budget asked): render without the think block.
            if (thinking_left_out(cr, gen.max_tokens)) {
                cr.template_options.enable_thinking = false;
                thinking_skipped = true;
                try {
                    prompt = render_chat(cr.messages, cr.tools.is_null() ? nullptr : &cr.tools, cr.template_options,
                                         &plain_spans, vision);
                } catch (const Error &e) {
                    throw ApiError(400, e.what(), "messages");
                }
                gen.prompt = tok_.encode(prompt, plain_spans);
                gen.max_tokens = clamp_max_tokens(cr.max_tokens, (int64_t)gen.prompt.size(), engine_.capacity(), clamped);
            }
            if (vision) {
                // The image pipeline: decode + resize + pack on this thread for every image the server hasn't
                // decoded before (a repeat takes its grid from vision_memo_ and keeps its bytes - the engine
                // decodes it only if its prefill reaches the block), the template's single pad expanded to each
                // image's merged rows, and the mRoPE walk + the cache-key stream.
                // Every failure here is the client's image (a 400 naming what was expected).
                if (cfg_.vision == nullptr)
                    throw ApiError(400, "image parts need the server's vision tower (config vision)", "messages");
                // The transient-RAM bound first, before any bytes decode (serve/vision_request.hpp
                // PreprocessGate): all slots held is a server-load refusal (503, retryable), not the
                // client's error. The slot releases when the vision block below unwinds.
                vision::PreprocessGate::Slot preprocess_slot;
                try {
                    preprocess_slot = vision_gate_.acquire(cfg_.vision_preprocess_limit);
                } catch (const Error &e) {
                    log_error(id, req, 503, e.what());
                    return send_error(res, 503, e.what(), "server_error", "messages");
                }
                try {
                    std::vector<vision::ServedImage> images = vision::prepare_images(std::move(cr.images), *cfg_.vision,
                                                                                      &vision_memo_);
                    int64_t decoded = 0;
                    for (const vision::ServedImage &im : images) decoded += !im.prefill.pixel_values.empty();
                    std::vector<int64_t> counts;
                    counts.reserve(images.size());
                    for (const vision::ServedImage &im : images) counts.push_back(im.merged_tokens);
                    const int64_t pre_tokens = (int64_t)gen.prompt.size();
                    gen.prompt = vision::expand_image_pads(gen.prompt, cfg_.vision->image_token_id, counts);
                    vision::VisionPromptPlan plan =
                        vision::plan_vision_prompt(gen.prompt, images, cfg_.vision->cfg,
                                                   cfg_.vision->image_token_id, engine_.backend().max_positions());
                    gen.image_pos3 = std::move(plan.pos3);
                    gen.cache_key = std::move(plan.cache_prompt);
                    gen.prefill_images.clear();
                    for (vision::ServedImage &im : images)
                        gen.prefill_images.push_back(PrefillImage{std::move(im.prefill), im.row_start, im.rows});
                    // An image block is forwarded whole in one forward (the session's forward limit is
                    // the chunk size): a server configured with a tiny chunk cannot serve big images.
                    const int64_t max_block = engine_.backend().max_chunk();
                    for (const PrefillImage &im : gen.prefill_images)
                        if (im.rows > max_block)
                            throw ApiError(400,
                                           cat("the image expands to ", im.rows, " rows, past this server's "
                                               "forward chunk of ", max_block, " (config chunk) - it cannot "
                                               "be forwarded in one piece"),
                                           "messages");
                    if (pre_tokens != gen.prompt.size())
                        vision_note = std::to_string((long long)images.size()) +
                                      (images.size() == 1 ? " image (" : " images (") +
                                      std::to_string((long long)decoded) + " decoded, " +
                                      std::to_string((long long)(images.size() - decoded)) + " known), " +
                                      std::to_string((long long)plan.merged_rows_total) + " expanded rows (prompt " +
                                      std::to_string((long long)pre_tokens) + " -> " +
                                      std::to_string((long long)gen.prompt.size()) + " tokens, rope positions to " +
                                      std::to_string((long long)plan.rope_positions_reached) + ")";
                    gen.max_tokens = clamp_max_tokens(cr.max_tokens, (int64_t)gen.prompt.size(), engine_.capacity(),
                                                      clamped);
                } catch (const Error &e) {
                    throw ApiError(400, e.what(), "messages");
                } catch (const std::exception &e) {
                    throw ApiError(400, e.what(), "messages");
                }
            }
        } catch (const ApiError &e) {
            log_error(id, req, e.status, e.what());
            return send_error(res, e.status, e.what(), e.type, e.param);
        } catch (const std::exception &e) {
            log_error(id, req, 500, e.what());
            return send_error(res, 500, e.what(), "server_error");
        }
        {
            const size_t msgs = cr.messages.is_array() ? cr.messages.as_array("messages").size() : 0;
            const size_t tools = cr.tools.is_array() ? cr.tools.as_array("tools").size() : 0;
            const bool thinking = !(cr.template_options.enable_thinking.has_value() && !*cr.template_options.enable_thinking);
            gen.log_header = header_rows(req, cr.stream, cr.model);
            gen.log_header.emplace_back("prompt", fmt_n((long long)gen.prompt.size()) + " tokens, " + fmt_n((long long)msgs) +
                                                      (msgs == 1 ? " message, " : " messages, ") + fmt_n((long long)tools) +
                                                      (tools == 1 ? " tool, thinking " : " tools, thinking ") +
                                                      (thinking_skipped      ? "skipped (no room)"
                                                       : thinking             ? "on"
                                                       : thinking_defaulted_off ? "off (server default)"
                                                                              : "off"));
            gen.log_header.emplace_back("max_tokens", max_tokens_text(cr.max_tokens, gen.max_tokens));
            gen.log_header.emplace_back("sampling", sampling_text(cr.sampling, cr.seeded, cr.sampling_asked));
            if (!format_row.empty()) gen.log_header.emplace_back("format", format_row);
            if (!vision_note.empty()) gen.log_header.emplace_back("vision", vision_note);
        }
        gen.id = id, gen.stream = cr.stream;
        if (!cr.seeded) cr.sampling.seed = std::random_device{}();
        gen.sampling = cr.sampling;
        auto tools = std::make_shared<json::Value>(cr.tools);  // outlives the request inside the engine
        gen.parser.thinking = !(cr.template_options.enable_thinking.has_value() && !*cr.template_options.enable_thinking);
        gen.parser.tools = tools->is_null() ? nullptr : tools.get();
        gen.parser.stop = cr.stop;
        gen.parser.id_seed = std::random_device{}();
        gen.parser.stream_calls = cr.stream;
        gen.thinking_budget = thinking_budget_for(cr, gen.max_tokens);
        if (cfg_.thinking_cap > 0 && !cr.thinking_budget.has_value() && gen.thinking_budget > cfg_.thinking_cap) {
            gen.thinking_budget = cfg_.thinking_cap;
            gen.thinking_budget_grace = cfg_.thinking_cap_grace;
            gen.log_header.emplace_back("thinking", cat("capped to ", fmt_n(cfg_.thinking_cap), " tokens (the answer's "
                                        "room was larger; +", fmt_n(cfg_.thinking_cap_grace), " grace to a sentence end)"));
        }
        auto sink = std::make_shared<QueueSink>();
        const bool thinking_on = !(cr.template_options.enable_thinking.has_value() && !*cr.template_options.enable_thinking);
        const ResponseMeta meta{random_id("chatcmpl-"), cfg_.model_id, unix_now(),
                                thinking_skipped ? "skipped" : thinking_on ? "on" : "off"};
        try {
            engine_.submit(std::move(gen), sink);
        } catch (const std::exception &e) {
            log_error(id, req, 503, e.what());
            return send_error(res, 503, e.what(), "server_error");
        }
        const double keepalive = cfg_.keepalive_s;

        if (!cr.stream) {
            std::string reasoning, content;
            std::vector<ToolCallOut> calls;
            QueueSink::Item it;
            for (;;) {
                if (!sink->pop(it, 0.25)) {
                    if (req.is_connection_closed()) sink->cancel();
                    continue;
                }
                if (it.kind == QueueSink::Item::Kind::Events)
                    for (OutputEvent &e : it.events) {
                        if (e.kind == OutputEvent::Kind::Reasoning) reasoning += e.text;
                        else if (e.kind == OutputEvent::Kind::Content) content += e.text;
                        else if (e.kind == OutputEvent::Kind::ToolCall) calls.push_back(std::move(e.call));
                    }
                if (it.kind == QueueSink::Item::Kind::Done) break;
            }
            if (it.result.finish_reason == "error") return send_error(res, 500, it.result.error, "server_error");
            res.set_content(completion_body(meta, reasoning, content, calls, it.result, clamped), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [sink, meta, clamped, keepalive, tools, rp = cr.return_progress](size_t, httplib::DataSink &out) {
            const auto send = [&](const std::string &data) {
                const std::string frame = "data: " + data + "\n\n";
                if (!out.write(frame.data(), frame.size())) {
                    sink->cancel();
                    return false;
                }
                return true;
            };
            const auto comment = [&](const std::string &text) {
                const std::string frame = ": " + text + "\n\n";
                if (!out.write(frame.data(), frame.size())) {
                    sink->cancel();
                    return false;
                }
                return true;
            };
            json::Value first = json::Value::object();
            first.set("role", json::Value::string("assistant"));
            first.set("content", json::Value::string(""));
            if (!send(chunk_body(meta, first, nullptr))) return false;
            int tool_index = 0;
            QueueSink::Item it;
            std::string progress;
            for (;;) {
                if (!sink->pop(it, keepalive)) {
                    if (!comment(progress.empty() ? "keep-alive" : progress)) return false;
                    continue;
                }
                switch (it.kind) {
                case QueueSink::Item::Kind::Start: progress = "started"; break;
                case QueueSink::Item::Kind::Progress:
                    progress = "prefill " + std::to_string(it.progress.processed) + "/" + std::to_string(it.progress.total);
                    if (rp) {
                        if (!send(prompt_progress_chunk_body(meta, false, it.progress))) return false;
                    } else if (!comment(progress)) {
                        return false;
                    }
                    break;
                case QueueSink::Item::Kind::Events:
                    for (size_t k = 0; k < it.events.size(); ++k) {
                        const OutputEvent &e = it.events[k];
                        json::Value d = json::Value::object();
                        if (e.kind == OutputEvent::Kind::Reasoning) d.set("reasoning_content", json::Value::string(e.text));
                        else if (e.kind == OutputEvent::Kind::Content) d.set("content", json::Value::string(e.text));
                        else {
                            json::Value tcs = json::Value::array();
                            if (e.kind == OutputEvent::Kind::ToolCall) {
                                tcs.push(tool_call_delta(e.call, tool_index++));
                            } else if (e.kind == OutputEvent::Kind::ToolCallStart) {
                                tcs.push(tool_call_start_delta(e.call, tool_index++));
                            } else {  // ToolCallArgs: consecutive pieces (one token can complete several), as one delta
                                std::string piece = e.text;
                                while (k + 1 < it.events.size() && it.events[k + 1].kind == OutputEvent::Kind::ToolCallArgs)
                                    piece += it.events[++k].text;
                                tcs.push(tool_call_args_delta(piece, tool_index - 1));
                            }
                            d.set("tool_calls", std::move(tcs));
                        }
                        if (!send(chunk_body(meta, d, nullptr))) return false;
                    }
                    break;
                case QueueSink::Item::Kind::Done: {
                    const GenerationResult &r = it.result;
                    if (r.finish_reason == "error") {
                        send(error_body(r.error, "server_error", ""));
                    } else if (r.finish_reason != "cancelled") {
                        send(chunk_body(meta, json::Value::object(), r.finish_reason.c_str()));
                        send(usage_chunk_body(meta, r, clamped));
                    }
                    const std::string done = "data: [DONE]\n\n";
                    out.write(done.data(), done.size());
                    out.done();
                    return true;
                }
                }
            }
        });
    });

    // Legacy completions: the raw prompt as is - no chat template, no
    // thinking or tool parsing; the same engine, queue, prefix reuse and prompt cache as chat.
    svr_->Post("/v1/completions", [this](const httplib::Request &req, httplib::Response &res) {
        const int64_t id = ++next_id_;
        CompletionRequest cr;
        GenerationRequest gen;
        bool clamped = false;
        try {
            json::Value body;
            try {
                body = json::Value::parse(req.body);
            } catch (const Error &e) {
                throw ApiError(400, std::string("request body: ") + e.what());
            }
            cr = parse_completion_request(body, cfg_.defaults);
            try {
                gen.prompt = tok_.encode(cr.prompt);
            } catch (const Error &e) {
                throw ApiError(400, e.what(), "prompt");
            }
            gen.max_tokens = clamp_max_tokens(cr.max_tokens, (int64_t)gen.prompt.size(), engine_.capacity(), clamped);
        } catch (const ApiError &e) {
            log_error(id, req, e.status, e.what());
            return send_error(res, e.status, e.what(), e.type, e.param);
        } catch (const std::exception &e) {
            log_error(id, req, 500, e.what());
            return send_error(res, 500, e.what(), "server_error");
        }
        gen.log_header = header_rows(req, cr.stream, cr.model);
        gen.log_header.emplace_back("prompt", fmt_n((long long)gen.prompt.size()) + " tokens, raw completion (no chat template)");
        gen.log_header.emplace_back("max_tokens", max_tokens_text(cr.max_tokens, gen.max_tokens));
        gen.log_header.emplace_back("sampling", sampling_text(cr.sampling, cr.seeded, cr.sampling_asked));
        gen.id = id, gen.stream = cr.stream;
        gen.one_shot = true;  // a raw prompt: its cache entries stay in RAM (benchmarks send many unique ones)
        if (!cr.seeded) cr.sampling.seed = std::random_device{}();
        gen.sampling = cr.sampling;
        gen.parser.thinking = false;  // no think block: every token is text
        gen.parser.trim_ws = false;   // white space is part of a raw continuation
        gen.parser.tools = nullptr;
        gen.parser.stop = cr.stop;
        gen.parser.id_seed = 0;
        gen.thinking_budget = -1;
        auto sink = std::make_shared<QueueSink>();
        const ResponseMeta meta{random_id("cmpl-"), cfg_.model_id, unix_now(), ""};
        try {
            engine_.submit(std::move(gen), sink);
        } catch (const std::exception &e) {
            log_error(id, req, 503, e.what());
            return send_error(res, 503, e.what(), "server_error");
        }
        const double keepalive = cfg_.keepalive_s;

        if (!cr.stream) {
            std::string text;
            QueueSink::Item it;
            for (;;) {
                if (!sink->pop(it, 0.25)) {
                    if (req.is_connection_closed()) sink->cancel();
                    continue;
                }
                if (it.kind == QueueSink::Item::Kind::Events)
                    for (const OutputEvent &e : it.events)
                        if (e.is_text()) text += e.text;
                if (it.kind == QueueSink::Item::Kind::Done) break;
            }
            if (it.result.finish_reason == "error") return send_error(res, 500, it.result.error, "server_error");
            res.set_content(text_completion_body(meta, text, it.result, clamped), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [sink, meta, clamped, keepalive, rp = cr.return_progress](size_t, httplib::DataSink &out) {
            const auto frame = [&](const std::string &f) {
                if (!out.write(f.data(), f.size())) {
                    sink->cancel();
                    return false;
                }
                return true;
            };
            QueueSink::Item it;
            std::string progress;
            for (;;) {
                if (!sink->pop(it, keepalive)) {
                    if (!frame(": " + (progress.empty() ? std::string("keep-alive") : progress) + "\n\n")) return false;
                    continue;
                }
                switch (it.kind) {
                case QueueSink::Item::Kind::Start: progress = "started"; break;
                case QueueSink::Item::Kind::Progress:
                    progress = "prefill " + std::to_string(it.progress.processed) + "/" + std::to_string(it.progress.total);
                    if (!frame(rp ? "data: " + prompt_progress_chunk_body(meta, true, it.progress) + "\n\n" : ": " + progress + "\n\n"))
                        return false;
                    break;
                case QueueSink::Item::Kind::Events:
                    for (const OutputEvent &e : it.events)
                        if (e.is_text() && !e.text.empty())
                            if (!frame("data: " + text_chunk_body(meta, e.text, nullptr) + "\n\n")) return false;
                    break;
                case QueueSink::Item::Kind::Done: {
                    const GenerationResult &r = it.result;
                    if (r.finish_reason == "error") {
                        frame("data: " + error_body(r.error, "server_error", "") + "\n\n");
                    } else if (r.finish_reason != "cancelled") {
                        frame("data: " + text_chunk_body(meta, "", r.finish_reason.c_str()) + "\n\n");
                        frame("data: " + text_usage_chunk_body(meta, r, clamped) + "\n\n");
                    }
                    frame("data: [DONE]\n\n");
                    out.done();
                    return true;
                }
                }
            }
        });
    });

    svr_->Get("/v1/models", [this](const httplib::Request &, httplib::Response &res) {
        json::Value m = json::Value::object();
        m.set("id", json::Value::string(cfg_.model_id));
        m.set("object", json::Value::string("model"));
        m.set("created", json::Value::integer(started_));
        m.set("owned_by", json::Value::string("strix-infer"));
        m.set("context_length", json::Value::integer(engine_.capacity()));
        json::Value data = json::Value::array();
        data.push(std::move(m));
        json::Value b = json::Value::object();
        b.set("object", json::Value::string("list"));
        b.set("data", std::move(data));
        res.set_content(b.dump(), "application/json");
    });

    svr_->Get("/health", [this](const httplib::Request &, httplib::Response &res) {
        const EngineStats s = engine_.stats();
        const std::string fatal = engine_.fatal_error();
        json::Value b = json::Value::object();
        // After a GpuFatalError: 503 and the error, until the process exits (strix_server) or forever (no on_fatal).
        if (!fatal.empty()) res.status = 503;
        b.set("status", json::Value::string(fatal.empty() ? "ok" : "error"));
        if (!fatal.empty()) b.set("error", json::Value::string(fatal));
        if (!engine_.degraded().empty()) b.set("degraded", json::Value::string(engine_.degraded()));
        b.set("model", json::Value::string(cfg_.model_id));
        b.set("backend", json::Value::string(engine_.backend().describe()));
        // Vision (Phase D): what the served engine can answer, not what the config asked for.
        json::Value vis = json::Value::object();
        vis.set("enabled", json::Value::boolean(engine_.backend().has_vision()));
        b.set("vision", std::move(vis));
        b.set("context_length", json::Value::integer(engine_.capacity()));
        b.set("busy", json::Value::boolean(s.busy != 0));
        b.set("queued", json::Value::integer(s.queued));
        b.set("idle_seconds", json::Value::number(s.idle_seconds));
        b.set("live_tokens", json::Value::integer(s.live_tokens));
        // Slots (total-slots): each one's running request (0: idle) and the tokens its state holds.
        b.set("slots", json::Value::integer(s.slots));
        b.set("busy_slots", json::Value::integer(s.busy));
        json::Value per = json::Value::array();
        for (size_t k = 0; k < s.slot_request.size(); ++k) {
            json::Value o = json::Value::object();
            o.set("request", json::Value::integer(s.slot_request[k]));
            o.set("live_tokens", json::Value::integer(k < s.slot_live_tokens.size() ? s.slot_live_tokens[k] : 0));
            o.set("snapshot_pos", json::Value::integer(k < s.slot_snapshot_pos.size() ? s.slot_snapshot_pos[k] : -1));
            per.push(std::move(o));
        }
        b.set("slot_states", std::move(per));
        res.set_content(b.dump(), "application/json");
    });

    svr_->Get("/cache", [this](const httplib::Request &, httplib::Response &res) {
        const EngineStats s = engine_.stats();
        json::Value b = json::Value::object();
        b.set("hits", json::Value::integer(s.cache_hits));
        b.set("misses", json::Value::integer(s.cache_misses));
        b.set("prompt_tokens_saved", json::Value::integer(s.cached_tokens));
        b.set("live_tokens", json::Value::integer(s.live_tokens));
        json::Value snaps = json::Value::array();  // per slot (-1: none)
        for (int64_t v : s.slot_snapshot_pos) snaps.push(json::Value::integer(v));
        b.set("snapshot_pos", std::move(snaps));
        b.set("disk_hits", json::Value::integer(s.disk_hits));
        b.set("disk_tokens_saved", json::Value::integer(s.disk_tokens));
        if (const PromptCache *pc = engine_.prompt_cache()) {
            const PromptCacheStats d = pc->stats();
            json::Value disk = json::Value::object();
            disk.set("dir", json::Value::string(pc->dir()));
            disk.set("entries", json::Value::integer(d.entries));
            disk.set("ram_entries", json::Value::integer(d.ram_entries));
            disk.set("ram_bytes", json::Value::integer(d.ram_bytes));
            disk.set("spare_bytes", json::Value::integer(d.spare_bytes));
            disk.set("ram_hits", json::Value::integer(d.ram_hits));
            disk.set("ram_evicted", json::Value::integer(d.ram_evicted));
            disk.set("ram_only_dropped", json::Value::integer(d.ram_only_dropped));
            disk.set("room_refused", json::Value::integer(d.room_refused));
            disk.set("room_waits", json::Value::integer(d.room_waits));
            disk.set("turn_buffers_reused", json::Value::integer(d.turn_buffers_reused));
            disk.set("disk_entries", json::Value::integer(d.disk_entries));
            disk.set("bytes", json::Value::integer(d.bytes));
            disk.set("max_bytes", json::Value::integer(d.max_bytes));
            disk.set("loads", json::Value::integer(d.loads));
            disk.set("writes", json::Value::integer(d.writes));
            disk.set("pending", json::Value::integer(d.pending));
            disk.set("evicted", json::Value::integer(d.evicted));
            disk.set("replaced", json::Value::integer(d.replaced));
            disk.set("bytes_written", json::Value::integer(d.bytes_written));
            disk.set("writes_evict", json::Value::integer(d.writes_evict));
            disk.set("writes_idle", json::Value::integer(d.writes_idle));
            disk.set("writes_shutdown", json::Value::integer(d.writes_shutdown));
            disk.set("rejected_space", json::Value::integer(d.rejected_space));
            disk.set("rejected_budget", json::Value::integer(d.rejected_budget));
            disk.set("rejected_queue", json::Value::integer(d.rejected_queue));
            disk.set("invalid", json::Value::integer(d.invalid));
            disk.set("write_failures", json::Value::integer(d.write_failures));
            b.set("disk", std::move(disk));
        }
        res.set_content(b.dump(), "application/json");
    });

    svr_->Get("/metrics", [this](const httplib::Request &, httplib::Response &res) {
        const EngineStats s = engine_.stats();
        std::string m;
        const auto metric = [&](const char *name, const char *type, const char *help, double v, const char *labels = "") {
            char buf[512];
            std::snprintf(buf, sizeof buf, "# HELP %s %s\n# TYPE %s %s\n%s%s %.17g\n", name, help, name, type, name, labels, v);
            m += buf;
        };
        const auto sample = [&](const char *name, const char *labels, double v) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "%s%s %.17g\n", name, labels, v);
            m += buf;
        };
        metric("strix_requests_total", "counter", "Chat completion requests by outcome.", (double)s.requests_ok, "{outcome=\"ok\"}");
        sample("strix_requests_total", "{outcome=\"error\"}", (double)s.requests_error);
        sample("strix_requests_total", "{outcome=\"cancelled\"}", (double)s.requests_cancelled);
        metric("strix_prompt_tokens_total", "counter", "Prompt tokens of finished prefills.", (double)s.prompt_tokens);
        metric("strix_prompt_tokens_cached_total", "counter", "Prompt tokens reused from the session.", (double)s.cached_tokens);
        metric("strix_prefill_tokens_total", "counter", "Prompt tokens prefilled.", (double)s.prefill_tokens);
        metric("strix_generated_tokens_total", "counter", "Tokens generated.", (double)s.generated_tokens);
        metric("strix_mtp_drafted_tokens_total", "counter", "MTP candidate tokens drafted.", (double)s.mtp_drafted_tokens);
        metric("strix_think_nudges_total", "counter", "Thinking nudges fed into a think block going in circles.",
               (double)s.think_nudges);
        metric("strix_think_end_blocked_total", "counter",
               "Ends of turn the guard took out inside the think block or a tool call while they were the top token.", (double)s.think_end_blocked);
        metric("strix_think_closed_for_call_total", "counter",
               "Think blocks the engine closed ahead of a tool call the model began inside them.", (double)s.think_closed_for_call);
        metric("strix_mtp_accepted_tokens_total", "counter", "MTP candidate tokens accepted.", (double)s.mtp_accepted_tokens);
        metric("strix_mtp_rollbacks_total", "counter", "MTP speculation rollbacks on rejection.", (double)s.mtp_rollbacks);
        metric("strix_mtp_rejections_total", "counter",
               "MTP rollbacks by the number of drafts accepted before the rejected one.", (double)s.mtp_reject_at[0],
               "{accepted=\"0\"}");
        for (int64_t i = 1; i < kMtpMaxDraft; ++i) {
            const std::string l = "{accepted=\"" + std::to_string(i) + (i == kMtpMaxDraft - 1 ? "+" : "") + "\"}";
            sample("strix_mtp_rejections_total", l.c_str(), (double)s.mtp_reject_at[i]);
        }
        const BackendStats bs = engine_.backend_stats();
        metric("strix_ngram_rows_requested_total", "counter", "PLE n-gram rows requested by forwards.",
               (double)bs.ngram_rows_requested);
        metric("strix_ngram_rows_unique_total", "counter", "Distinct PLE n-gram rows per gather, summed.",
               (double)bs.ngram_rows_unique);
        metric("strix_ngram_rows_cached_total", "counter", "Distinct rows served from the row cache.",
               (double)bs.ngram_rows_cached);
        metric("strix_ngram_rows_read_total", "counter", "Distinct rows read from the table file (SSD / page cache).",
               (double)bs.ngram_rows_read);
        metric("strix_ngram_rows_prefetched_total", "counter",
               "Rows read from the table file into the row cache ahead of their gather (gathers count them as cached).",
               (double)bs.ngram_rows_prefetched);
        metric("strix_ple_prefetches_total", "counter", "PLE row prefetch jobs run.", (double)bs.ple_prefetches);
        metric("strix_ple_prefetch_seconds_total", "counter", "Time spent in PLE row prefetches.", bs.ple_prefetch_seconds);
        metric("strix_ple_prefetch_skipped_total", "counter", "PLE row prefetches skipped (queue full).",
               (double)bs.ple_prefetch_skipped);
        metric("strix_ple_gathers_total", "counter", "PLE row gathers.", (double)bs.ple_gathers);
        metric("strix_ple_gather_seconds_total", "counter", "Time spent gathering PLE rows.", bs.ple_gather_seconds);
        metric("strix_ple_wait_seconds_total", "counter", "Time forwards waited for PLE rows (the GPU idle).",
               bs.ple_wait_seconds);
        metric("strix_ple_wait_max_seconds", "gauge", "Longest single wait for PLE rows.", bs.ple_wait_max_seconds);
        metric("strix_vision_images_total", "counter", "Images through the vision tower (one forward each).",
               (double)bs.vision_images);
        metric("strix_vision_patches_total", "counter", "Patches the tower forwards ran over.",
               (double)bs.vision_patches);
        metric("strix_vision_merged_tokens_total", "counter", "Merged image tokens scattered into prompts by the tower.",
               (double)bs.vision_merged_tokens);
        metric("strix_vision_tower_seconds_total", "counter",
               "Time in vision tower runs (upload, tower, scatter; the journal line's span).", bs.vision_tower_seconds);
        metric("strix_vision_tower_max_seconds", "gauge", "Longest single vision tower run.", bs.vision_tower_max_seconds);
        metric("strix_mtp_acceptance_rate", "gauge", "MTP acceptance rate [0, 1].",
               s.mtp_drafted_tokens > 0 ? (double)s.mtp_accepted_tokens / (double)s.mtp_drafted_tokens : 0.0);
        metric("strix_prefill_seconds_total", "counter", "Time spent prefilling.", s.prefill_seconds);
        metric("strix_decode_seconds_total", "counter", "Time spent generating.", s.decode_seconds);
        metric("strix_cache_hits_total", "counter", "Requests that reused part of the session.", (double)s.cache_hits);
        metric("strix_cache_misses_total", "counter", "Requests prefilled from scratch.", (double)s.cache_misses);
        metric("strix_last_prefill_tokens_per_second", "gauge", "Prefill rate of the last request.", s.last_prefill_tps);
        metric("strix_last_decode_tokens_per_second", "gauge", "Decode rate of the last request.", s.last_decode_tps);
        metric("strix_queue_depth", "gauge", "Requests waiting.", (double)s.queued);
        metric("strix_busy", "gauge", "Requests running (0..strix_slots).", (double)s.busy);
        metric("strix_slots", "gauge", "Conversations served at once (total-slots).", (double)s.slots);
        metric("strix_verify_batches_total", "counter", "Batched MTP verifies across slots.", (double)s.verify_batches);
        metric("strix_verify_batch_runs_total", "counter",
               "Slots' runs in those batched verifies (/ batches = runs per batched forward).", (double)s.verify_batch_runs);
        metric("strix_verify_solo_mrope_total", "counter",
               "Verifies that ran alone because the conversation carries mRoPE position state (it ran an image, "
               "or resumed one): a multi-run forward takes scalar positions only, so that conversation never "
               "shares a batch (issue #29).",
               (double)s.verify_solo_mrope);
        metric("strix_idle_seconds", "gauge",
               "Seconds since the last request finished; 0 while one runs or waits. Batch clients: wait for a gap here.",
               s.idle_seconds);
        metric("strix_live_tokens", "gauge", "Tokens held across all slots.", (double)s.live_tokens);
        metric("strix_context_length", "gauge", "Positions the session holds.", (double)engine_.capacity());
        metric("strix_prompt_cache_disk_hits_total", "counter", "Requests resumed from the prompt cache (RAM or disk).", (double)s.disk_hits);
        metric("strix_prompt_cache_disk_tokens_total", "counter", "Prompt tokens restored from the prompt cache.", (double)s.disk_tokens);
        metric("strix_prompt_cache_export_seconds_total", "counter", "Time copying states out for the prompt cache.", s.export_seconds);
        if (const PromptCache *pc = engine_.prompt_cache()) {
            const PromptCacheStats d = pc->stats();
            metric("strix_prompt_cache_entries", "gauge", "Entries on disk.", (double)d.disk_entries);
            metric("strix_prompt_cache_bytes", "gauge", "Bytes on disk.", (double)d.bytes);
            metric("strix_prompt_cache_ram_entries", "gauge", "Entries in RAM.", (double)d.ram_entries);
            metric("strix_prompt_cache_ram_bytes", "gauge", "Host memory held by RAM entries.", (double)d.ram_bytes);
            metric("strix_prompt_cache_spare_bytes", "gauge", "Host memory held by spare buffers.", (double)d.spare_bytes);
            metric("strix_prompt_cache_ram_hits_total", "counter", "Entries resumed from RAM.", (double)d.ram_hits);
            metric("strix_prompt_cache_ram_evicted_total", "counter", "Entries that left RAM for lack of memory.", (double)d.ram_evicted);
            metric("strix_prompt_cache_ram_only_dropped_total", "counter",
                   "One-shot requests' entries that left RAM (never written to disk, by design).", (double)d.ram_only_dropped);
            metric("strix_prompt_cache_room_refused_total", "counter",
                   "Saves skipped or disk loads refused: no room in RAM above the margin.", (double)d.room_refused);
            metric("strix_prompt_cache_room_waits_total", "counter",
                   "Polls make_room_for spent waiting for evicted entries' writes to give their RAM back.", (double)d.room_waits);
            metric("strix_prompt_cache_buffers_reused_total", "counter",
                   "Exports / loads given an on-disk entry's RAM buffer instead of a fresh mapping.", (double)d.buffers_reused);
            metric("strix_prompt_cache_prefaulted_total", "counter", "Spare buffers populated ahead of an export.", (double)d.prefaulted);
            metric("strix_prompt_cache_writes_total", "counter", "Entries written to disk.", (double)d.writes);
            metric("strix_prompt_cache_writes_by_rule_total", "counter", "Entries written to disk, by write rule.",
                   (double)d.writes_evict, "{rule=\"evict\"}");
            sample("strix_prompt_cache_writes_by_rule_total", "{rule=\"idle\"}", (double)d.writes_idle);
            sample("strix_prompt_cache_writes_by_rule_total", "{rule=\"shutdown\"}", (double)d.writes_shutdown);
            metric("strix_prompt_cache_bytes_written_total", "counter", "Bytes written to disk.", (double)d.bytes_written);
            metric("strix_prompt_cache_rejected_total", "counter", "Entries leaving RAM that weren't written, by reason.",
                   (double)d.rejected_space, "{reason=\"space\"}");
            sample("strix_prompt_cache_rejected_total", "{reason=\"budget\"}", (double)d.rejected_budget);
            sample("strix_prompt_cache_rejected_total", "{reason=\"queue\"}", (double)d.rejected_queue);
            metric("strix_prompt_cache_loads_total", "counter", "Entries loaded.", (double)d.loads);
            metric("strix_prompt_cache_load_seconds_total", "counter", "Time loading entries.", d.load_seconds);
            metric("strix_prompt_cache_write_seconds_total", "counter", "Time writing entries.", d.write_seconds);
            metric("strix_prompt_cache_evicted_total", "counter", "Entries evicted for room.", (double)d.evicted);
            metric("strix_prompt_cache_invalid_total", "counter", "Entries dropped as damaged or foreign.", (double)d.invalid);
        }
        res.set_content(m, "text/plain; version=0.0.4");
    });

    // One handler for every error status. 404 gets its JSON body; a 400 whose body never fully arrived gets turned
    // into the retryable 408 the client can act on (strixite #15: httplib answers the abandoned read itself,
    // before any route ran - an empty, unlogged 400, which OpenCode took for its own fault and stopped the
    // session; the OpenAI SDKs retry 408 and x-should-retry, not 400). Errors raised inside a route carry their own JSON and pass
    // through untouched, as do parse failures (which never get a POST's framing).
    svr_->set_error_handler([this](const httplib::Request &req, httplib::Response &res) {
        if (res.status == 404) {
            send_error(res, 404, "no route " + req.method + " " + req.path, "not_found_error");
            return;
        }
        if (res.status != 400 || !res.body.empty() || req.method != "POST" || req.target.empty()) return;
        const std::string content_length = req.get_header_value("Content-Length");
        if (!content_length.empty() && content_length.find_first_not_of("0123456789") != std::string::npos)
            return;  // a malformed Content-Length also reads as this empty 400 - that's the client's framing to fix, not a slow upload
        const std::string arrived = content_length.empty()
            ? std::to_string(req.body.size()) + " bytes arrived"
            : fmt_n((int64_t)req.body.size()) + " of " + content_length + " bytes arrived";
        slog(LogLevel::Warning, "Request failed: POST %s from %s: the body stopped arriving (read timeout %d s, %s) - "
             "answered 408 (retryable)", req.path.c_str(), req.remote_addr.c_str(), cfg_.request_read_timeout_s,
             arrived.c_str());
        send_error(res, 408, "the request body stopped arriving (read timeout " + std::to_string(cfg_.request_read_timeout_s) +
                             " s of silence on the connection" + (content_length.empty() ? "" : ", " + arrived) +
                             ") - the upload may be slow or the connection was lost; safe to retry", "api_error");
        res.set_header("x-should-retry", "true");
    });
}

}  // namespace strix
