# deploy/ — the public demo

This folder is everything needed to put `chat.py` on the public internet, so
anyone can talk to the models this repository's research session produced
without installing a thing.

**Live:** <https://autoresearch-demo.fly.dev/>

## What it serves

The same `chat.py` UI you get locally, in public-demo mode: the session's
Baseline (`75027e8`, val_bpb 0.520096) and Best (`e9fffd9`, val_bpb 0.518708)
side by side, the 16-experiment progress chart, and the vocabulary browser.
The models are ~19M parameters, so it runs on a plain CPU machine — that fact
is part of what the demo teaches.

## How it is wired

`chat.py` is the *same file* the repo uses locally. Setting `SPACE_MODE=1`
(done in the Dockerfile) switches on the public-demo behavior:

- binds `0.0.0.0:$PORT` instead of scanning for a free localhost port
- clamps requests: max 500 tokens, `top_k` ≤ 200, prompt ≤ 2000 characters —
  silently, never an error
- retitles the page (it isn't "local" any more) and adds a banner linking
  [WALKTHROUGH.md](../WALKTHROUGH.md)

With the variable unset — every local run — none of that applies.

**The image builds from the repo root, not from this folder.** That is
deliberate: the site is built out of the committed `chat.py`, `results.tsv`,
and checkpoints, so whatever it serves corresponds to a state that exists in
git history. `../.dockerignore` keeps the context small (`.git` alone is ~1 GB).

| File | Role |
|------|------|
| `Dockerfile` | CPU-only image; build context is the repo root |
| `fly.toml` | Fly.io app config (region, machine size, idle suspend) |
| `tokenizer/tokenizer.pkl` | Vendored because `prepare.py` generates it into a user cache outside the repo |

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

The demo shows whichever checkpoints the Dockerfile copies. To point it at a
model from a later session:

1. Commit the new checkpoint under `checkpoints/` and its `results.tsv` row.
2. Update the two `COPY` paths in `Dockerfile` (and the `--checkpoint` in
   `CMD`), plus the un-ignore lines at the bottom of `../.dockerignore`.
3. `fly deploy --config deploy/fly.toml` from the repo root.

The Baseline/Best labels come from `results.tsv` — the UI picks the first
`keep` row as Baseline and the lowest-scoring `keep` row as Best — so the row
and the checkpoint file must both be present for a model to appear.

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
