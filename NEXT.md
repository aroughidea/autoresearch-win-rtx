# Next

Updated: 2026-11-22

## Where it stands

This is a fork of [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx), the Windows consumer-GPU port of Karpathy's autoresearch. This copy adds the experiment record (`record.py`, `lab.py`, `session.py`), `chat.py` and `generate.py`, a library of trained models, and the hosted demo at https://autoresearch-demo.fly.dev/, which answers. The last work (23 and 24 Oct) put the eight models from the two 10-hour full-attention sessions (Folktales and TinyStories) on the hosted demo, in How a Model Gets Made's look.

## Next step

Give `generate.py` the same model lookup `chat.py` has, so it builds each model with the code that trained it (the version kept in `sessions/<name>/versions/`, or the attention windows in its run file) instead of today's `train.py`. Until then, `DEMO.md`'s Tier 3 fallback can build a session's best model with the wrong code (`docs/TODO.md`, "Next").

## After that

- Show each run's step count in `lab.py history`, so the speed check needs no `Select-String` (`docs/TODO.md`).
- Run one session with a smaller agent model than the one the sessions used, to see what participants without it would get; decided on 13 October as "not now".

## Blocked or waiting

- New training sessions are started by TJ, when the laptop is free.

## Open questions

- Is anything still owed to the course starter kit, or is this repository now the finished worked example?
- This plan was drafted by Claude on 2026-11-22 from the code and its history. Is the next step right?
