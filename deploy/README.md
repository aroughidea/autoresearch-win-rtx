# deploy/ — the public demo

This folder is everything needed to put `chat.py` on the public internet, so
anyone can talk to the models this repository's research session produced
without installing a thing.

**Live:** <https://autoresearch-demo.fly.dev/>

## What it serves

The same `chat.py` UI you get locally, in public-demo mode, with four models
from the two sessions LLM Explorables' [How a Model Gets Made](https://llmexplorables.aroughidea.com/training/)
features, whose "Use a model" step links here: `folktales-b-10min-2026-10-21` and
`tinystories-b-10min-2026-10-20` (10-minute runs, the full-attention starter
script). For each, the start (the last of its three start runs, the one that kept
its model) and the best version the agent found. It opens on Folktales' start and
best side by side; its tabs are Continue text, Watch it learn (all four), The
session (that session's chart) and The vocabulary. The page wears How a Model Gets
Made's look, light and dark.

| Model | File (`checkpoints/`) | val_bpb |
|-------|-----------------------|---------|
| Folktales, start | `20261021T202736-0700_c9ba08e4f442.pt` | 3.292523 (the three start runs average 3.287000) |
| Folktales, best | `20261022T052816-0700_6c874e790ab4.pt` | 1.207438 |
| TinyStories, start | `20261021T000138-0700_c9ba08e4f442.pt` | 0.436369 (the three average 0.436605) |
| TinyStories, best | `20261021T130839-0700_05602a6a5706.pt` | 0.425055 |

The models have 26 to 34 million parameters, so it runs on a plain CPU machine;
that fact is part of what the demo teaches.

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
- adds a paragraph under the title that says what the four models are, linking
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
| `checkpoints/` | The four models (Git LFS). Kept out of the root `checkpoints/`, so a local `chat.py` never lists them without their run files |
| `sessions/` | What `chat.py` reads from each session's record: `session.json`, `results.tsv`, the four models' run files and the version code each is rebuilt with; `active.txt` names the session whose chart the page opens on |
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
