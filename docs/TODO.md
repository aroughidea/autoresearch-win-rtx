# To do

Open work, in rough order. Each item says what it waits for.

## Waits for the first overnight session with the record

- **Merge #29** if the session's first hour checks out: `run.log` prints `Time budget: 600s`;
  after the three baseline runs, `uv run lab.py status` shows the noise; `uv run lab.py check`
  is clean; the agent's log has no git commands.
- **Starter kit (aroughidea/autoresearch-starter#9):** rehearse from a fresh "Use this
  template" copy, following its README literally (quickstart, `session.py --check`, a short
  session), then merge after #29.

## Waits for a few build sessions

- **Hosted demo: serve the build sessions' models.** <https://autoresearch-demo.fly.dev/> still
  serves the May session's two models (5-minute runs, from before the record). Once a few build
  sessions have run (10-minute runs, the record), choose what it serves. Proposal (9 October):
  one session's baseline and best, for the agent's night, plus the ten-minute study's Folktales
  model, for a decision you can read in the writing (the dataset). Steps:
  1. Pick the checkpoints. A baseline's model file is the last of its three runs: `lab.py keep`
     saves only the last run's model.
  2. Copy their run files (this also switches on the demo's "Watch it learn" card) and the
     session's `results.tsv` into the image.
  3. Vendor each extra vocabulary into `deploy/` (today only TinyStories' own is there).
  4. Update `deploy/Dockerfile` and `deploy/README.md`.
  5. Deploy to Fly. It is a public site: ask TJ first.
- **DEMO.md's reveal and fallbacks** then name those models (they are marked pending now).

## Then

- **DEMO.md dry run** on the laptop, to time the segment.
- **Explorer** (llm-explorables `/training/`, production; TJ merges):
  - **In review: tj60647/llm-explorables#36** reads session folders, matches runs to rows by version
    and start time, starts the staircase at the baseline's average with the session's own noise,
    marks crashes, and names a run's version and parent. Check it with tonight's real session
    folder (drag and drop), then merge after #29.
  - Later: the diff between a version and its parent (needs `versions/`), and the library format
    and `scripts/sync-training-library.mjs` for record sessions.
- **chat.py's chart:** the best line should start at the baseline's average, as in the explorer.
- **The session map as an explorables page** ("How a session runs"), linked from the Evolution
  view, the explorer's intro and the workshops' Going Further quest 5. Keep
  `docs/session-map.html` in step with it.
- **`library/report.py`:** the manifest's `size` (8192) counts 4 control tokens; the registry
  wants 8188. Fix before vendoring the vocabularies (Phase A2).
- **Workshops "Pretraining on a Laptop"** (tj60647/aroughidea-workshops#2): TJ merges.
