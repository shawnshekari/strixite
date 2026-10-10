# What the server accepts

strixite's server speaks the OpenAI API on port 5300. This page covers the parts where it does something you might
not expect: how thinking is switched and limited, the thinking nudge, and exactly what structured output enforces -
and what it refuses.

| endpoint | what it is |
|---|---|
| `POST /v1/chat/completions` | chat, with streaming, tool calls, reasoning and structured output |
| `POST /v1/completions` | a raw prompt, no chat template (same engine, queue and prompt cache) |
| `GET /v1/models` | the one served model |
| `GET /health`, `GET /cache`, `GET /metrics` | liveness, the prompt cache's state, Prometheus counters |

Every response carries a `strix` object next to `usage` with what the server did: whether thinking was `on`, `off`
or `skipped`, a thinking budget that was hit, MTP acceptance, and the number of thinking nudges.

## Thinking

The model thinks before it answers, in a think block returned as `reasoning_content`. Thinking is **on** unless the
request turns it off. Any of these spellings works - clients differ, so the server accepts them all:

| to... | send any of |
|---|---|
| turn thinking off | `"enable_thinking": false`, `"chat_template_kwargs": {"enable_thinking": false}`, `"thinking": {"type": "disabled"}`, `"reasoning_effort": "none"` |
| turn it on explicitly | the same with `true` / `"enabled"`, or any other `reasoning_effort` |
| cap it | `"max_thinking_tokens": N`, `"reasoning": {"max_tokens": N}`, `"thinking": {"budget_tokens": N}`, `"chat_template_kwargs": {"thinking_budget": N}` |

Two spellings that disagree (on and off, or two different budgets) are a 400, never a guess.

When a budget runs out, the server closes the think block itself and the model goes on to answer. Without a budget,
thinking may use whatever `max_tokens` leaves after room for the answer (1024 tokens, or 15% of `max_tokens` if
that's more). A request that didn't ask for thinking and whose `max_tokens` leaves no room at all is answered with
thinking skipped, and `strix.thinking` says so.

Server setting: `thinking = off` in the config makes requests that don't say either way run without thinking. I
measured it on terminal-bench: the model took 4-7 times as many steps and was no faster, so the shipped default is on.

## The thinking nudge

Sometimes the model's thinking goes in circles - the same calculation, the same doubt, again and again. When the
server sees that, it writes one sentence into the think block in the model's own voice, suggesting it commit to an
approach. It never closes the think block itself; the model decides whether to wrap up.

- **When:** at least 25% of the last 1024 thinking tokens repeat 8-token sequences seen earlier in the same thinking,
  and the thinking is at least 3072 tokens long. It waits for a paragraph break, so it never lands mid-sentence.
- **At most twice** per turn: a firmer second nudge if the circling goes on well past the first.
- **How often:** rarely. On a full terminal-bench run it fired in 14 of ~550 requests, and most of those turns
  finished thinking within ~1.4k tokens. On turns forked at the exact point it fires, no nudge meant no turn stopped
  within 2000 more tokens; with the nudge, a quarter did.
- **What I tried and dropped:** nudging earlier (10% repetition from 1024 tokens) - mostly ignored; and a nudge on
  length alone - ignored, or it made the model write its answer inside the think block.
- **Settings:** `think-nudge = on|off`, `think-nudge-rate` (0.25), `think-nudge-min-tokens` (3072). Turn it off for
  like-for-like comparisons with engines that don't have it. `strix.think_nudges` in a response counts the nudges.

## Structured output

`response_format` with `json_object` or `json_schema` (`{name, strict, schema}`) constrains the answer token by token:
the model can only produce text the schema allows. Thinking is unaffected - the constraint starts at the first token
after the think block (or the first token, with thinking off).

**Enforced:** `type`, `properties`, `required`, `additionalProperties: false`, `items`, `enum`, `const`, `anyOf` /
`oneOf`, `$ref` with `$defs`, `minLength` / `maxLength` (in code points) and `minItems` / `maxItems` (bounds up to
4096).

**Stricter than the schema in two ways, on purpose:**
- `strict: false` is constrained exactly like `strict: true`.
- Whitespace between JSON tokens is bounded, so the model can't run away with spaces.

**Ignored** (they describe, they don't constrain): `description`, `title`, `default`, `examples`, `$comment`.

**Refused with a 400 naming the keyword and where it is** - never accepted and silently left unenforced: numeric
bounds (`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`), `pattern`, `format`,
`uniqueItems`, `allOf`, `not`, `if` / `then` / `else`, `prefixItems`, `patternProperties`, `propertyNames`,
`dependentRequired` / `dependentSchemas`, and a schema for `additionalProperties`. Also refused: `json_object` /
`json_schema` together with tools (the answer is one or the other), and `tool_choice` `required` or a named tool.

These wait for a real client that needs them. If one of these 400s gets in your way, please open an issue with the
schema - that's what moves a keyword up the list.

A schema is compiled once and cached (the 16 most recent), so repeating one costs nothing the second time. Long
`maxLength` bounds cost a little on a schema's first use, since every position in the string is its own state.

## Streamed tool calls

In a streamed response (`"stream": true`) a tool call goes out as the model writes it, the way OpenAI streams one:
the call's `id`, `type` and function `name` as soon as the model has written the name, then its `arguments` piece by
piece - a string parameter (a file's content, a command) as its characters come, a number, boolean, object or array
whole once the model closes it, and the closing `}` when the call ends. Joined, the pieces are exactly the
`arguments` a non-streamed response would have.

A streamed call can't be taken back. A call that turns out malformed - a tag out of place, text after
`</function>`, a parameter named twice - would come back as plain `content` without streaming (or, named twice,
with the last value); streamed, it stays a call whose `arguments` never get their closing `}`, so they don't parse
as JSON. A call cut off - by `max_tokens`, or by the model ending its turn inside it - is left the same way. Either
way `finish_reason` is `tool_calls` (`length` for a `max_tokens` cut), and `strix.malformed_tool_calls` in the last
chunk counts such calls (`strix.dropped_partial_tool_call` says one was cut off).

What a client does with unparseable arguments is up to it: most report an error back to the model, which then tries
again; openai-python's parsing stream helpers (strict tools) raise; a client that repairs partial JSON (LangChain)
can run the repaired call - complete when the call broke between parameters, wrong when it was cut off inside a
value. Sent back in the conversation, a call whose arguments were left open this way is shown to the model with no
parameters, rather than failing the request.
