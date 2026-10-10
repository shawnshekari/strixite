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
- **Faster prefill at long context and on short prompts.** The sparse-attention scoring pass at depth, and the first
  chunk of a prompt waiting on n-gram table reads from the SSD.

## Next

- **`tool_choice: required` and a named tool**, enforced the way structured output is.

## Considering

Ideas I want, but that wait for evidence that they're worth their cost, or for the items above.

- **Vision (image input).** Planned, not yet. It will be opt-in, with the vision encoder as a separate download
  (~0.9 GB) so existing weights stay as they are. First a quality check of this model's encoder on real screenshots
  and diagrams.
- **Two conversations at once.** Measured ~1.3x combined throughput with two agents, but in my use two requests are
  rarely in flight together, and a second slot's memory would cost every user.
- **Prefill progress for clients**, as a stream chunk rather than a comment most clients don't show - there's an open
  pull request for it (#4, @TheBeaninator).
- **Changing settings without a restart**: an endpoint to change the server's settings while it runs - no 30-second
  reload of the weights, no emptied prompt cache.
- **Sleep mode**: after a configurable idle time, free the GPU memory and reload on the next request - for machines
  that do other things between sessions.
- **Smarter low-memory behaviour**: when memory is short, keep the prompt cache's RAM tier at whatever size fits
  instead of turning it off.
- **Smaller state**: an 8-bit KV cache and a BF16 recurrent state - more room for the prompt cache, if quality holds.
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
