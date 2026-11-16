# Contributing to strixite

Thank you for wanting to help. I take ideas, bug reports and measurements as issues. I don't accept pull requests.

## Why issues and not pull requests

strixite is published from the repository I work in day to day, and every change has to pass my full test suite
and benchmarks on my own Strix Halo before it ships. That means I write each change myself, even when someone sends
working code. What helps me most is the idea behind a change and the evidence for it, and an issue carries both.

When an issue leads to a change, I credit you in the release notes, for example "@you (suggested in #12)".

Please don't paste code copied from other projects into an issue. strixite is written from scratch, and I'd like to
keep it that way. Ideas from elsewhere are fine; say where they came from.

## What makes an issue useful

- **For a bug:** the request, the settings and the output that show it, and the strixite version you ran. The
  smallest request that still shows the bug is the most useful.
- **For a speed idea:** what you'd change and why you expect it to be faster. A before/after measurement from a
  Strix Halo helps a lot, with how you took it:
  - the command and the settings;
  - the old and the new build on the same machine, alternated (old, new, old, new, ...), at least four rounds each;
  - every round's number, or the spread, next to the difference. One run per side can't tell a gain of a few percent
    from run-to-run noise.
- **For a kernel idea:** where the time goes. Give the kernel's own time before and after (a micro-benchmark, or a
  trace with the `strix-roctx` preset), and roughly what share of a forward it takes. A kernel with a tiny share
  can't move end-to-end speed much, however much faster it gets.
- **Output identity:** if you tried a speed change, run a few prompts at `temperature` 0 on both builds and compare
  the tokens. Identical output means the change only moves speed. If the output differs, say where. Decode speed with
  the draft head depends on the text being written, so comparing the speed of two different outputs mostly measures
  the difference in text.
- **One idea per issue.** Separate ideas are much easier for me to test and to credit.

## Issues written with an LLM

Fine with me; I use them too. Before you send one, please check it against the code: the functions it names should
exist, the mechanism it describes should be what the code does, and every number should come from a run you did. A
wrong explanation costs us both time, even when the idea itself is good.

## Scope

strixite targets one chip and one model: AMD Strix Halo (gfx1151) on Linux, and Qwen3.8-Flash-Next. I can't take on
ports to other platforms or models here, but forks are very welcome (see [Platform](README.md#platform)).
