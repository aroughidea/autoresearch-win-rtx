# deploy/ — the public demo

This folder is everything needed to put `chat.py` on the public internet, so
anyone can talk to the models this repository's research session produced
without installing a thing.

**Live:** <https://autoresearch-demo.fly.dev/>

## What it serves

The same `chat.py` UI you get locally, in public-demo mode, with four models
from the two overnight sessions of 12–13 October (10-minute runs): for each of
`tinystories-10min-2026-10-12` and `folktales-10min-2026-10-12`, the starting
recipe (the last of its three baseline runs) and the best version the agent
found. It opens on TinyStories' start and best side by side, with that night's
progress chart, the "Watch it learn" card for all four, and the vocabulary
browser.

| Model | File (`checkpoints/`) | val_bpb |
|-------|-----------------------|---------|
| TinyStories, start | `20261012T102252-0700_91ffb4c37015.pt` | 0.476278 |
| TinyStories, best | `20261012T182536-0700_dfb66193b616.pt` | 0.428101 |
| Folktales, start | `20261012T203326-0700_91ffb4c37015.pt` | 1.913044 |
| Folktales, best | `20261013T033410-0700_f02876c49522.pt` | 1.242914 |

The models have 19 to 26 million parameters, so it runs on a plain CPU machine;
that fact is part of what the demo teaches.

## How it is wired

`chat.py` is the *same file* the repo uses locally. Setting `SPACE_MODE=1`
(done in the Dockerfile) switches on the public-demo behavior:

- binds `0.0.0.0:$PORT` instead of scanning for a free localhost port
- clamps requests: max 500 tokens, `top_k` ≤ 200, prompt ≤ 2000 characters —
  silently, never an error
- retitles the page (it isn't "local" any more) and adds a banner that says
  what the four models are, linking [TRAINING-DECISIONS.md](../TRAINING-DECISIONS.md)
  and the README's non-technical introduction

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
   the file's time to the run's (`touch -d`); git does not keep it.
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
