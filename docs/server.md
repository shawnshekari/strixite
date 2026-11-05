# What the server accepts

strixite's server speaks the OpenAI API on port 5300. This page covers the parts where it does something you might
not expect: how thinking is switched and limited, the thinking nudge, prefill progress in a stream, conversations
served together, the K/V budget behind long uptimes, and exactly what structured output enforces - and what it
refuses.

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

Server setting: `thinking-cap = N` caps the thinking of every request that sets no budget of its own: past N tokens
(and the end of the sentence it's in) the server closes the think block and the model answers. The shipped default
is `0`, no cap. v0.3 shipped 2,048, and people told me the model seemed less smart with it, so from v0.4.0 the cap is
yours to set. A request's own budget always wins.

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

## Streaming

A streamed response (`"stream": true`) sends server-sent events: one `data:` chunk per piece of reasoning, content
or tool call, then a chunk with the `finish_reason`, a last chunk with `usage` and `strix`, and `data: [DONE]`.

**Prefill progress.** A long prompt can take minutes to read before the first token. By default the stream says how
far it got only in SSE comment lines (`: prefill 24576/65536`), which keep the connection alive but which clients
don't show. A request that sends `"return_progress": true` gets it as `data:` chunks instead - one when the prefill
begins, one after each prefill chunk - with a top-level `prompt_progress` object and an empty delta:

```json
{"id": "chatcmpl-...", "object": "chat.completion.chunk", "created": 1760000000, "model": "...",
 "choices": [{"index": 0, "delta": {}, "finish_reason": null}],
 "prompt_progress": {"total": 65536, "cache": 8192, "processed": 24576, "time_ms": 9120}}
```

`total` is the prompt's tokens, `cache` the ones reused from the prompt cache rather than read again, `processed`
the ones in the model's state so far (cached ones included; it reaches `total` when the prefill is done), `time_ms`
the time since the prefill began. llama.cpp's server streams the same field with the same keys for the same flag,
so a harness that asks llama.cpp for it (llama.cpp's own web UI does, when it streams) can ask strixite too.
`/v1/completions` takes the flag as well (its chunks carry empty `text`); without streaming it does nothing.

## Conversations at once

The server holds `total-slots` conversations at the same time (2 by default, up to 8; [how many fit a given
memory](memory.md#more-conversations-at-once)). Each slot is one conversation's
state at the full context, so a coding agent and its subagents, or two clients, don't evict one another - and their
decode steps share one forward through the model, so the weights are read once for all of them. Two conversations
cost nearly nothing extra; more of them share the step but read more expert weights each, so deep conversations
stay best in pairs and shallow ones run in crowds. Requests past the slots wait
in arrival order; `strix_busy` out of `strix_slots` in `/metrics` shows how many are taken, and while one slot reads
a long prompt the others keep decoding, because the prefill runs in small chunks when a second slot is busy. A slot's
memory grows with its conversation: [memory and the slot recipes](memory.md#slots-conversations-at-once).

A conversation that has held an image takes its decode steps on its own instead of sharing them: its positions
aren't the plain kind a shared step takes. The other slots still share with each other, and
`strix_verify_alone_total` in `/metrics` counts the steps that ran alone.

## Long uptimes: the K/V budget

On this GPU, memory released and mapped again at the same virtual address can read the wrong data without any
error. So whenever a slot switches to a different conversation, the new conversation's K/V goes at addresses the
process has never used. That costs no RAM, only address space, and address space is large but not endless: each
slot gets `kv-generations` switches. The default, 0, means auto - the most, up to 1024, that fit your slots and
capacity: 1024 at one or two slots of 512k, 1001 at three, 801 at four. The startup log prints the count. A
conversation that continues in its slot spends nothing; only a switch does.

You don't have to restart to get the budget back. When a slot comes within `kv-renew-margin` (16) switches of its
last, the server renews itself: it holds new requests, lets the running ones finish, writes the conversations it
kept in memory to disk and rebuilds its working state on the loaded model. The weights stay loaded and the prompt
cache stays; it takes seconds, and every conversation resumes from the cache. Set an explicit `kv-generations` only
if you want a different number - startup refuses one that doesn't fit and names the most that would.

What to watch:

- `GET /health` carries `kv_budget`: `generations_left` per slot, `generations_capacity`, `parked_left` (the
  shared budget for conversations kept in memory at a switch) and `renewals` since startup. `status` turns
  `degraded` - still HTTP 200, the server keeps serving - with the reasons and an `action` when a slot is at its last
  switch, the shared budget is spent, a request was refused, or a renewal failed.
- With `kv-renew-margin = 0` (renewal off), a switch past the last is refused with HTTP 503 and the error type
  `kv_budget_exhausted`, and only a restart gets the budget back.
- `/metrics`: `strix_kv_generation_used` / `strix_kv_generation_capacity` per slot,
  `strix_parked_generation_used` / `strix_parked_generation_capacity`, `strix_kv_renewals_total` and
  `strix_kv_renew_failures_total`.

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
