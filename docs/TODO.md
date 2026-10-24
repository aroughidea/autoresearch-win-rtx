# To do

Open work, in rough order. Each item says what it waits for.

## Done: the full-attention studies (20-22 October)

The starter kit's recipe is full attention since 20 October (aroughidea/autoresearch-starter#13). TJ, 21
October: one recipe on the learner's page. So the library now has the baselines and both sessions
remade from it, all in `library/README.md`, "Full attention":
- six ten-minute baselines (`baselines-b-10min/`) and a five-minute pair (`baselines-b-5min/`)
- `tinystories-b-10min-2026-10-20`: 49 runs, 0.4366 to 0.4251, 2.6%, 21 times the noise
- `folktales-b-10min-2026-10-21`: 46 runs, 3.2870 to 1.2074, 63%, 56 times the noise. The faster
  recipe overfits Folktales, and the agent cured it with dropout.

Every run trained 600 s (300 for the pair) with power throttling opted out, and every start passed
`speed_gate.py`. The agent was `claude-fable-5` with the long-lived token, except TinyStories' first
launch, which ran on the stored sign-in until it expired at 03:19. The earlier collections are
labelled "windowed attention". Next: llm-explorables syncs this library into the new `/training/` (its
build plan), which shows full attention only.

## Done: the first sessions with the record (12-13 October)

Two 10-hour sessions of 10-minute runs, one launch each, the record consistent (`lab.py check`),
every run 600 s, no git: `tinystories-10min-2026-10-12` (44 runs; 0.4767 to 0.4281, 10.2%, about
22 times the noise, mostly from attention settings that train 2.5 times as many steps) and
`folktales-10min-2026-10-12` (43 runs; 1.9158 to 1.2429, 35%, mostly from a matrix learning rate of
3.0, sixty times the starter's). The agent was Claude Fable 5 (`claude-fable-5`). Their logs are in
TJ's local `ar-session-logs`; their records in the benches `ar-bench-<name>/sessions/<name>/`.
#29 merged after them, with `session.py`'s stop, resume, continue and model recording.

## Next

- **Starter kit rehearsal (13 October, done):** a fresh "Use this template" copy, its README followed
  literally. The quickstart, `session.py --check`, `chat.py`, `lab.py check` and `--stop` all worked; the
  agent ran no git and its model was recorded. A 1.5-hour session got through its three baseline runs and
  one experiment (cut off). The finding: the laptop slowed between baseline runs (1,193 steps, then 744
  and 744), the session's noise came out 13 times the overnight one, and the agent took
  the normal-speed run for the outlier. Both READMEs now say how to keep the machine at one speed and how
  to check the step counts. The cause, measured that afternoon: Windows power throttling (the same recipe
  ran at 790, 479 and 810 ms a step, opted out only the second time); TJ also found the disk full. Since 20
  October `prepare.py` opts every training process out, in both repos, and `library/speed_gate.py` checks
  a run's steps against a floor. Possible follow-up: show each run's steps in `lab.py history`, so the check
  needs no `Select-String`. The rehearsal's local copy and cache were removed afterwards; its private
  GitHub repo (tj60647/autoresearch-rehearsal-2026-10-13) is to be deleted from GitHub's settings page.
- **Explorer (tj60647/llm-explorables#36):** merged 13 October.
- **`chat.py` builds each model with the code that trained it** (fixed 13 October): the version the
  record kept in `sessions/<name>/versions/`, else the attention windows its run file recorded. Built
  with today's `train.py` instead, the TinyStories session's best model scored 0.4798 on held-out text,
  worse than the session's starting point; with its own code, 0.4303.
- **The starter recipe's attention setting.** Its single shared key/value head, explicit window mask and
  logit softcap force a slow attention kernel on this laptop's GPU; the TinyStories agent found and
  removed them (2.5 times the steps), the Folktales agent did not. Decided 13 October: keep the recipe
  for now, so every session can make the discovery and the existing baselines still compare; revisit
  after the starter rehearsal (fixing it would start new sessions faster but break comparison with
  every baseline and session so far). The rehearsal did not reach it: its one experiment halved the batch
  for more steps, which the TinyStories night had tried and discarded. Decided 20 October (TJ: "it looks
  like recipe b is a good way to start"): the starter kit switches to recipe B, the starter recipe with
  only those three changes (`library/recipes/recipe-b.py`, version `c9ba08e4f442`;
  aroughidea/autoresearch-starter#13, merged before the recipe B sessions). Every session before it
  started from `91ffb4c37015`. The baselines and both sessions are remade from recipe B:
  `docs/superpowers/plans/2026-10-20-recipe-b-studies.md`.
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

## Doing: the hosted demo serves the featured sessions' models, in the explorables' look (23 October)

How a Model Gets Made's "Use a model" step is to link to <https://autoresearch-demo.fly.dev/>, so the
demo serves the sessions that page features: Folktales' 21 October and TinyStories' 20 to 21 October,
each one's start and best (decided 23 October, TJ: "the fly demo needs an overhaul to better align with
the latest work and visual language"). `chat.py`'s page now wears that page's look, light and dark,
in four tabs (Continue text, Watch it learn, The session, The vocabulary), and opens on the run files'
own openings and sampling settings. `deploy/sessions/` holds the two sessions' records.

The same server is also the backend of LLM Explorables' model demo (its contract:
`docs/superpowers/specs/2026-10-23-model-demo-contract.md` there), so it serves eight models, four per
session (the start, two kept along the way, the best), each named by its run id, loaded on request with
four kept in memory (decided 23 October). Within a run (the contract's later section, the same day): each also
answers untrained (`<run id>@0s`), and the two bests at three moments of a re-run made with `deploy/rerun.py`
and `capture.py`'s moment switch. Left to do on the training machine: the eight model files into
`deploy/checkpoints/` (named in `deploy/README.md`; `tests/test_deploy_bundle.py` fails until they are the
right ones), the two bests re-run with `deploy/rerun.py`, the vocabularies checked, the image simulated, and `fly deploy` (the steps are in PR #39).

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
