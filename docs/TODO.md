# To do

Open work, in rough order. Each item says what it waits for.

## Done: the first sessions with the record (12-13 October)

Two 10-hour sessions of 10-minute runs, one launch each, the record consistent (`lab.py check`),
every run 600 s, no git: `tinystories-10min-2026-10-12` (44 runs; 0.4767 to 0.4281, 10.2%, about
22 times the noise, mostly from attention settings that train 2.5 times as many steps) and
`folktales-10min-2026-10-12` (43 runs; 1.9158 to 1.2429, 35%, mostly from a matrix learning rate of
3.0, sixty times the starter's). The agent was Claude Fable 5 (`claude-fable-5`). Their logs are in
TJ's local `ar-session-logs`; their records in the benches `ar-bench-<name>/sessions/<name>/`.
#29 merged after them, with `session.py`'s stop, resume, continue and model recording.

## Next

- **Starter kit (aroughidea/autoresearch-starter#9):** it has the record and the same `session.py`
  (da37bf7). Rehearse from a fresh "Use this template" copy, following its README literally
  (quickstart, `session.py --check`, a short session), then merge.
- **Explorer (tj60647/llm-explorables#36):** checked on 13 October with both session folders: all
  runs load, the staircase starts at each baseline's average with the session's own noise, and the
  totals match the record. TJ merges.
- **`chat.py` builds each model with the code that trained it** (fixed 13 October): the version the
  record kept in `sessions/<name>/versions/`, else the attention windows its run file recorded. Built
  with today's `train.py` instead, the TinyStories session's best model scored 0.4798 on held-out text,
  worse than the session's starting point; with its own code, 0.4303.
- **The starter recipe's attention setting.** Its single shared key/value head, explicit window mask and
  logit softcap force a slow attention kernel on this laptop's GPU; the TinyStories agent found and
  removed them (2.5 times the steps), the Folktales agent did not. Decided 13 October: keep the recipe
  for now, so every session can make the discovery and the existing baselines still compare; revisit
  after the starter rehearsal (fixing it would start new sessions faster but break comparison with
  every baseline and session so far).
- **The agent's model for participants:** both sessions used Fable; participants may not have it or
  may have tighter limits. Run one session with `--model claude-opus-5-5` or `claude-sonnet-5-5`
  and compare. Decided 13 October: not now.
- **`generate.py` builds every model with today's `train.py`.** `chat.py` now uses the code that
  trained each model; `generate.py` does not, so DEMO.md's Tier 3 fallback (the reveal in the
  terminal) would build a best version whose code changed with the wrong code. Give it the same
  lookup as `chat.py`.

## Done: the hosted demo serves the build sessions' models (13 October)

<https://autoresearch-demo.fly.dev/> serves four models from the 12-13 October sessions: TinyStories'
and Folktales' starting recipe (the last of three baseline runs) and best version, opening on
TinyStories' pair (decided 13 October). Their files, the parts of the session records `chat.py`
reads, and both vocabularies are in `deploy/`; `deploy/README.md` has the steps for changing them.
Simulating the image first found a page bug: the two sessions' baselines share a version, and the
page matched scoreboard rows to models by version alone, so it opened on Folktales' start beside
TinyStories' best. Each row now names the model its own run saved. DEMO.md's reveal and fallbacks
name the four models.

## Then

- **DEMO.md dry run** on the laptop, to time the segment.
- **Explorer** (llm-explorables `/training/`, production; TJ merges): the diff between a version and
  its parent (needs `versions/`), and the library format and `scripts/sync-training-library.mjs`
  for record sessions, so the 12 October sessions can join the library. The redesign ("How a Model
  Gets Made") is a draft, tj60647/llm-explorables#38.
- **Statistics from data, not prose** (decided 10 October). Every "read in five minutes" kind of
  figure is specific to one machine and one run length, and the defaults are moving to 10 minutes.
  1. `capture.py` records the GPU's name in every run file (`torch.cuda.get_device_name`), in the
     worked example and the starter.
  2. The explorer takes the run length from `time_budget_s` (older files: `training_s` to the
     minute) and the machine from the run file or a collection field, and builds every sentence
     that says "five minutes" from them. It shows one conditions line per tab.
  3. When the baselines are remade at 10 minutes, the 5-minute set stays as a dated collection;
     the "5 or 10 minutes" comparison needs both.
- **chat.py's chart:** the best line should start at the baseline's average, as in the explorer.
- **The session map as an explorables page** ("How a session runs"), linked from the Evolution
  view, the explorer's intro and the workshops' Going Further quest 5. Keep
  `docs/session-map.html` in step with it.
- **`library/report.py`:** the manifest's `size` (8192) counts 4 control tokens; the registry
  wants 8188. Fix before vendoring the vocabularies (Phase A2).
- **Workshops "Pretraining on a Laptop"** (tj60647/aroughidea-workshops#2): TJ merges.
