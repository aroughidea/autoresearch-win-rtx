# deploy/ — the public demo

This folder is everything needed to put `chat.py` on the public internet, so
anyone can talk to the models this repository's research session produced
without installing a thing.

**Live:** <https://autoresearch-demo.fly.dev/>

## What it serves

The same `chat.py` UI you get locally, in public-demo mode, with eight models
from the two sessions LLM Explorables' [How a Model Gets Made](https://llmexplorables.aroughidea.com/training/)
features, whose "Use a model" step links here: `folktales-b-10min-2026-10-21` and
`tinystories-b-10min-2026-10-20` (10-minute runs, the full-attention starter
script). For each, four kept versions: the start (the last of its three start
runs, the one that kept its model), two kept along the way, and the best version the
agent found. It opens on Folktales' start and best side by side; its tabs are
Continue text, Watch it learn (all eight), The session (that session's chart) and
The vocabulary. The page wears How a Model Gets Made's look, light and dark.

LLM Explorables' model demo calls this same server (`/models`, `/generate`) through
its own route, naming each model by its run id, the file's stem; those eight are
listed in its `shared/DemoModels.js`. `tests/test_model_server.py` pins what it
depends on.

| Model | Experiment | File (`checkpoints/`) | val_bpb |
|-------|-----------|-----------------------|---------|
| Folktales, start | 3 | `20261021T202736-0700_c9ba08e4f442.pt` | 3.292523 (the three start runs average 3.287000) |
| Folktales, along the way | 4 | `20261021T203936-0700_ae5c68b35cb6.pt` | 2.203520 |
| Folktales, along the way | 9 | `20261021T213750-0700_7d99f54b4193.pt` | 1.299880 |
| Folktales, best | 46 | `20261022T052816-0700_6c874e790ab4.pt` | 1.207438 |
| TinyStories, start | 3 | `20261021T000138-0700_c9ba08e4f442.pt` | 0.436369 (the three average 0.436605) |
| TinyStories, along the way | 6 | `20261021T003646-0700_6db69b64b97b.pt` | 0.431590 |
| TinyStories, along the way | 17 | `20261021T024603-0700_b351add2f9da.pt` | 0.427558 |
| TinyStories, best | 49 | `20261021T130839-0700_05602a6a5706.pt` | 0.425055 |

The two along the way are the first kept versions a third and two thirds of the
way from start to best, except Folktales' second, moved from experiment 5 to 9,
where its steep fall levels off. Together the eight files are 0.69 GB (76.5 MiB each, 120 MiB for Folktales'
best).

**Within a run** (the contract's section of that name). Every one of the eight
also answers untrained, as `<run id>@0s`: `chat.py` builds the version's model
from its own code with `train.py`'s seed (42) and never stores it. It is a model
of the same shape as the one the run began from, not that exact one, which the
run drew on the training machine's GPU. And the two bests answer at three moments
of a re-run, `<run id>@10s`, `@60s` and `@300s`, each listed with `moment_s` and
`rerun_of`; their end is the session's own best, the plain `<run id>`, so each
dataset has one best everywhere. A session saves only its end, so
`deploy/rerun.py` trains a best again on the training machine, outside any
session, with `capture.py`'s moment switch on (`AR_SAVE_MOMENTS`, which a session
refuses). It files the re-run's run file in `sessions/<session>/reruns/` and its
three models in `checkpoints/` as `<re-run id>@<seconds>s.pt`; the re-run's own
end model is not kept. A moment is the model as it trained at that moment, the
weights its samples there came from (both bests' saved end is the average of
their weights, EMA, which only the end gets). A re-run isn't identical to the
original (the laptop fits a slightly different number of steps into ten
minutes), so a moment's growth card shows the re-run's own samples. The re-runs'
six files add 0.62 GB, 1.31 GB in all: 22 models, 14 of them files.

The models have 26 to 34 million parameters, so it runs on a plain CPU machine;
that fact is part of what the demo teaches. A model loads the first time it is
asked for (about a second) and at most four stay in memory, the least recently
used dropped first (`MODELS_IN_MEMORY` in `chat.py`, which says why four). On
CPU, Python with torch and `chat.py` takes about 650 MB, each 26.3 M model about
100 MB more and the 33.6 M one about 150 MB; a model is read through `mmap`, so
loading one never holds a second copy. Cycling through 20 of them, the 22 less
one re-run's moments, stays near 1.5 GB, and four visitors generating at once from
random models peaked at 1.6 GB, inside the machine's 2 GB.

`tests/test_deploy_bundle.py` checks the bundle: `deploy/checkpoints/` holds exactly
the models the run files in `deploy/sessions/` name, each with the SHA-256 its run
recorded (Git LFS names a file by that hash, so the check works on an unfetched
clone), every model has its version's code, and the image opens on one of the
active session's models.

## How it is wired

`chat.py` is the *same file* the repo uses locally. Setting `SPACE_MODE=1`
(done in the Dockerfile) switches on the public-demo behavior:

- binds `0.0.0.0:$PORT` instead of scanning for a free localhost port
- clamps requests: max 500 tokens, `top_k` ≤ 200, prompt ≤ 2000 characters —
  silently, never an error
- adds a paragraph under the title that says what the eight models are, linking
  How a Model Gets Made and the README's non-technical introduction

With the variable unset — every local run — none of that applies.

**The image builds from the repo root, not from this folder.** That is
deliberate: the site is built out of the committed `chat.py` and this folder's
models and session records, so whatever it serves corresponds to a state that
exists in git history. `../.dockerignore` keeps the context small (`.git` alone
is ~1 GB).

| File | Role |
|------|------|
| `Dockerfile` | CPU-only image; build context is the repo root |
| `fly.toml` | Fly.io app config (region, machine size, idle suspend) |
| `checkpoints/` | The eight models and the two re-runs' three moments each (Git LFS). Kept out of the root `checkpoints/`, so a local `chat.py` never lists them without their run files |
| `sessions/` | What `chat.py` reads from each session's record: `session.json`, `results.tsv`, the eight models' run files and the version code each is rebuilt with, and `reruns/`, the two bests' re-runs' run files; `active.txt` names the session whose chart the page opens on |
| `tokenizer/tokenizer.pkl` | TinyStories' vocabulary, vendored because `prepare.py` generates it into a user cache outside the repo |
| `tokenizer-folktales/tokenizer.pkl` | Folktales' vocabulary, the same way |

## Deploying

Requires the [Fly CLI](https://fly.io/docs/flyctl/install/) and `fly auth login`.
Run from the **repo root**, not from this folder:

```powershell
fly deploy --config deploy/fly.toml
```

First build takes a few minutes (the CPU torch wheel dominates); later builds
reuse cached layers. Check on it with `fly logs -a autoresearch-demo` or
`fly status -a autoresearch-demo`.

## Serving a newer model

The demo shows whatever is in `checkpoints/` and `sessions/` here. To serve a
model from another session:

1. Copy its model file from the session's machine (`checkpoints/<run id>.pt`)
   into `checkpoints/` here. The page lists models in file-time order, so set
   the file's time to the run's (`touch -d`, or in PowerShell
   `(Get-Item $f).LastWriteTime = "2026-10-21 20:27:36"`); git does not keep it.
2. Copy into `sessions/<name>/` what `chat.py` reads: `session.json`,
   `results.tsv`, `runs/<run id>.json` for the model, and
   `versions/<version>.py` for its version. Without the run file the model is
   decoded with TinyStories' vocabulary; without the version it is rebuilt
   with today's `train.py`, which scored the TinyStories best 0.4798 instead
   of 0.4303. Set `sessions/active.txt` to the session the page should open on.
3. A dataset with a new vocabulary needs its `tokenizer.pkl` vendored here and
   a `COPY` line in `Dockerfile` to that dataset's cache folder.
4. Simulate the image before deploying: a folder holding exactly what the
   `COPY` lines copy, with `SPACE_MODE=1`, `CUDA_VISIBLE_DEVICES=-1` and
   `AUTORESEARCH_CACHE_DIR` pointing at the tokenizers. Check `/models` lists
   every model as available and each one continues a prompt.
5. `fly deploy --config deploy/fly.toml` from the repo root.

The two panes start on the active session's first `keep` row (its starting
recipe) and its lowest-scoring `keep` row from the same dataset (its best).
Each row leads to the model its own run saved; a baseline row whose run kept
no model leads to its session's model of the same version. Either picker can
then switch to any model in the image.

## Cost and cold starts

The machine suspends when idle and wakes on the next request, which takes
**about 12 seconds**; Fly holds the request during the wake, so a visitor sees
a slow first load rather than an error. Warm loads are ~0.07s. At demo traffic
this runs to a couple of dollars a month.

For a lecture, open the link a minute beforehand so the machine is awake — see
[DEMO.md](../DEMO.md).

## Hosting it somewhere else

Nothing here is Fly-specific except `fly.toml`. The image runs anywhere that
takes a container: Cloud Run, Render, a VPS. It also runs as a **Hugging Face
Docker Space** — add Space front-matter (`sdk: docker`, `app_port: 7860`) to a
README at the repo root and push. Note that as of August 2026 Hugging Face
requires a PRO subscription for Docker Spaces even on free CPU hardware, which
is why this deployment lives on Fly.
