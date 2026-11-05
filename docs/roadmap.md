# Roadmap

**Priorities, in order: stability, quality, performance.** A server that stays up comes first, then answers you can
trust, then speed.

What I'm working on, what comes next, and what I'm deliberately not doing. No dates: strixite is built by one person
around real use, and priorities move with what I measure. I update this page with each release.
The limit is rarely ideas: every change is tested and measured on the one machine I have, and that takes most of the
time. With fifteen Strix Halos, this page would be a lot shorter.

The focus stays the same throughout: a Strix Halo dedicated to serving a coding agent - fast at long context, stable
through long sessions.

Nothing here is set in stone. If there's a feature you need, or something on this page matters more to you than
where it sits, [open an issue](https://github.com/shawnshekari/strixite/issues/new) and say what you'd use it for -
real use cases are what move items up.

## Working on now

- **Less wasted thinking.** On agent benchmarks, most of the generated tokens are thinking, and much of it goes in
  circles. I'm measuring where it adds nothing to the answer, then tuning the thinking nudge and the think-block guard
  to cut it - the biggest speed lever left for agent work.
- **Faster prefill at long context and on short prompts.** v0.5.0 rewrote the grouped-expert kernels and moved whole
  passes onto the matrix units; still to do: the sparse-attention scoring pass at depth, and the first chunk of a
  prompt waiting on n-gram table reads from the SSD.
- **Sharing more work when conversations run together.** Each conversation's expert routing picks its own experts, so
  a step with more than one conversation reads more of the weights; making bigger batches share more of that is the
  next step for agents that run several at once.

## Next

- **Hand an idle slot's memory back**: a slot keeps the K/V memory its deepest conversation used; giving it to the
  prompt cache while the slot waits would lend it back.
- **Say why a conversation was read again**: when a client changes the start of the prompt (a tool switched on, a
  date in the system prompt), the log and the response will name the cause, so it's clear the server didn't lose it
  ([why it happens](memory.md#when-the-client-changes-the-start-of-the-prompt)).
- **`tool_choice: required` and a named tool**, enforced the way structured output is.

## Considering

Ideas I want, but that wait for evidence that they're worth their cost, or for the items above.

- **More conversations at once**: two slots by default; other counts fit (three and four of 512k start since
  v0.6.0, six shorter ones measured fine). Raising the default is my call, not a limit - say in an issue what you'd
  run.
- **Changing settings without a restart**: an endpoint to change the server's settings while it runs - no 30-second
  reload of the weights, no emptied prompt cache.
- **Sleep mode**: after a configurable idle time, free the GPU memory and reload on the next request - for machines
  that do other things between sessions.
- **Smarter low-memory behaviour**: when memory is short, keep the prompt cache's RAM tier at whatever size fits
  instead of turning it off.
- **Smaller state**: the 8-bit KV cache shipped in v0.4.0; a BF16 recurrent state is next - more room for the
  prompt cache, if quality holds.
- **Responses API over WebSocket** ([#9](https://github.com/shawnshekari/strixite/issues/9)): a client sends each
  step's new items instead of the whole history, so long agent sessions upload and parse far less.
- **Calibrated 4-bit weights**: same size, possibly better quality - a new weights release if it measures better.
- **Easier setup**: a `models-dir` setting next to the `STRIX_MODELS_DIR` variable, and a one-command build per
  distribution.

## Not planned

- **Other GPUs, other operating systems, a CPU fallback.** strixite is written for one chip (gfx1151) and tested on
  one machine; a fallback would hide the bugs it should surface. Forks are welcome.
- **Other models**, for now - every kernel is written for this model's architecture.
- **Prebuilt binaries** beyond the container image.

## Helping

The most useful contributions are bug reports with the server's startup log (the `startup: memory:` line especially)
and changes that come with a measurement - a before / after on the same machine. See
[CONTRIBUTING.md](../CONTRIBUTING.md).
