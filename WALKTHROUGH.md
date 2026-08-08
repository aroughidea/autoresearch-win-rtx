# Walkthrough: the may23-pm research session

## What you're reading

On the afternoon of 2026-05-23, an AI agent ran a roughly two-hour hyperparameter research session in this repository. It edited `train.py`, committed the change, trained a small language model on TinyStories for a fixed 5-minute budget on a single laptop RTX GPU, read off the validation score, and then made a call: keep the change if the score improved, `git reset` it away if it didn't. Sixteen logged runs later, the model's validation bits-per-byte (`val_bpb`, lower is better) had gone from 0.520096 to 0.518708. No human touched the keyboard during the loop.

The agent used git as its lab notebook. Every experiment is a commit; every keep/discard decision is a state transition; the scoreboard (`results.tsv`) is committed alongside the code it describes. That model is explained in the README's [How does git fit in?](README.md#how-does-git-fit-in) section, and one convention makes the whole thing navigable: **the 7-character commit hash is the join key**. A hash links a `results.tsv` row to the exact code that produced it (`git show <hash>`) and to the checkpoint file it saved (`checkpoints/<timestamp>_<commit>.pt`).

This document materializes that git history into one readable file: every diff the agent tried, every score it got, and what each result meant — including the ten failures whose commits were reset away and survive only because they were rescued from the local reflog (more on that in [War stories](#war-stories)). You can read this instead of running anything, and check any claim against the repo with `git show`.

## The session at a glance

Every run trains from random initialization for 5 wall-clock minutes, then evaluates. "Keep" means the run beat the current best and its commit stayed on `master`; "discard" means it didn't and the change was removed from `master` (by rollback — or, for the late-evaluated experiment 16, by revert).

| # | Commit | Change tried | val_bpb | Verdict | Takeaway |
|---|--------|--------------|---------|---------|----------|
| 1 | `75027e8` | — (baseline, inherited config) | 0.520096 | keep | Anchor for the afternoon; every change must beat this. |
| 2 | `b6b0ec3` | `DEPTH` 6 → 7 | 0.537554 | discard | Bigger model, fewer steps in 5 min: badly undertrained. |
| 3 | `48c6a02` | `WINDOW_PATTERN` `"SSSL"` → `"SSSS"` | 0.519684 | keep | Dropping full-context attention is a free throughput win here. |
| 4 | `35b184c` | `TOTAL_BATCH_SIZE` 2¹⁵ → 2¹⁴ | 0.519890 | discard | More steps, noisier gradients: a wash. |
| 5 | `b240c00` | `MATRIX_LR` 0.05 → 0.06 | 0.520477 | discard | Hotter matrix LR hurts — probe the other direction. |
| 6 | `d07a0d2` | `MATRIX_LR` 0.05 → 0.04 | 0.519105 | keep | Cooler is better; the optimum is below 0.05. |
| 7 | `1e3f3e1` | `MATRIX_LR` 0.04 → 0.035 | 0.520977 | discard | Too cold. Optimum now bracketed in (0.035, 0.05). |
| 8 | `780adb0` | `MATRIX_LR` 0.04 → 0.045 | 0.518993 | keep | Midpoint of the bracket wins. |
| 9 | `89a8842` | `MATRIX_LR` 0.045 → 0.0475 | 0.520235 | discard | Confirming probe; sweep closed at 0.045. |
| 10 | `07992b2` | `EMBEDDING_LR` 1.0 → 0.8 | 0.519878 | discard | Slower embeddings hurt. |
| 11 | `cb8fe91` | `EMBEDDING_LR` 1.0 → 1.2 | 0.521757 | discard | Faster embeddings hurt more. 1.0 confirmed. |
| 12 | `407918e` | `UNEMBEDDING_LR` 0.004 → 0.003 | 0.520669 | discard | Output head needs its learning speed. |
| 13 | `58af6ef` | `UNEMBEDDING_LR` 0.004 → 0.005 | 0.519294 | discard | Beat baseline, not the best — strict rule says discard. |
| 14 | `742986f` | `WARMDOWN_RATIO` 0.45 → 0.35 | 0.520582 | discard | Shorter anneal is worse. |
| **15** | **`e9fffd9`** | **`WARMDOWN_RATIO` 0.45 → 0.55** | **0.518708** | **keep** | **Session best. Longer anneal wins.** |
| 16 | `f588ed5` | `WARMDOWN_RATIO` 0.55 → 0.65 | 0.523293 | discard | Evaluated 75 days late after an interruption; too much anneal. |

> **Footnote on honesty:** one historical commit just before the session, `cf23e99` ("Implement feature X to enhance user experience and fix bug Y in module Z"), carries a meaningless placeholder message left by editor autocomplete. It was kept as-is. The history here is honest, not curated.

## The tour

Chronological, one experiment per subsection. Each shows the exact `train.py` diff the agent committed, the score against the then-best, and the decision.

### 1. Baseline — `75027e8` (15:57) · 0.520096 · keep

No `train.py` change — the baseline run was made at a docs commit (`75027e82`, README wording changes only). The config it measured was the state inherited from prior sessions:

```
DEPTH = 6, WINDOW_PATTERN = "SSSL", TOTAL_BATCH_SIZE = 2**15,
EMBEDDING_LR = 1.0, UNEMBEDDING_LR = 0.004, MATRIX_LR = 0.05, SCALAR_LR = 0.5,
WEIGHT_DECAY = 0.1, ADAM_BETAS = (0.8, 0.95),
WARMUP_RATIO = 0.02, WARMDOWN_RATIO = 0.45, FINAL_LR_FRAC = 0.1
```

A reference run to anchor the noise floor for the afternoon: the existing Muon+AdamW config — the trainer uses two optimizers, Muon for the weight matrices and AdamW (a standard adaptive optimizer) for the embeddings and scalars, each group with its own learning rate — run once, scoring **0.520096** at 6.6 GB of VRAM. This became the bar every subsequent one-variable change had to beat. Running the baseline first — rather than trusting an old number — is cheap insurance against silent environment drift.

### 2. Depth 7 — `b6b0ec3` (16:09) · 0.537554 vs 0.520096 · discard

```diff
diff --git a/train.py b/train.py
index ac214c5..7798652 100644
--- a/train.py
+++ b/train.py
@@ -816,7 +816,7 @@ WARMDOWN_RATIO = 0.45
 FINAL_LR_FRAC = 0.1
 
 # Model size + memory defaults
-DEPTH = 6
+DEPTH = 7
 N_KV_HEAD = 1  # MQA: all query heads share 1 KV head. None = full MHA (n_kv_head=n_head)
 DEVICE_BATCH_SIZE = int(os.environ.get("AUTORESEARCH_DEVICE_BATCH_SIZE", "16"))
 EVAL_BATCH_SIZE = 8
```

Hypothesis: a deeper (and, via `ASPECT_RATIO = 64`, also wider) model learns more per token. But under a fixed 5-minute wall clock, the larger model (VRAM jumped 6.6 → 9.2 GB) runs fewer, slower steps and sees far less data, so it finished badly undertrained — the worst result of the session by a wide margin (+0.0175 over baseline). Lesson: under a time budget, "bigger model" is not one change, it's two — more capacity *and* less training.

### 3. Mostly short attention — `48c6a02` (16:19) · 0.519684 vs 0.520096 · **keep**

```diff
diff --git a/train.py b/train.py
index ac214c5..af7e743 100644
--- a/train.py
+++ b/train.py
@@ -801,7 +801,7 @@ class MuonAdamW(torch.optim.Optimizer):
 # Model architecture
 ASPECT_RATIO = 64         # model_dim = depth * ASPECT_RATIO
 HEAD_DIM = 128            # target head dimension for attention
-WINDOW_PATTERN = "SSSL"   # sliding window pattern: L=full, S=half context
+WINDOW_PATTERN = "SSSS"   # sliding window pattern: L=full, S=half context
 
 # Optimization
 TOTAL_BATCH_SIZE = 2 ** 15
```

Hypothesis: TinyStories' short, local narratives don't need any full-context attention layer, so replacing the last full-context `L` layer with a half-context `S` window buys throughput for free. It did: the compute savings let more tokens through in 5 minutes with no modeling penalty, a small but real win (0.519684 vs 0.520096). This became the new base config for the rest of the session.

### 4. Global batch 16384 — `35b184c` (16:27) · 0.519890 vs 0.519684 · discard

```diff
diff --git a/train.py b/train.py
index af7e743..d4ca59f 100644
--- a/train.py
+++ b/train.py
@@ -804,7 +804,7 @@ HEAD_DIM = 128            # target head dimension for attention
 WINDOW_PATTERN = "SSSS"   # sliding window pattern: L=full, S=half context
 
 # Optimization
-TOTAL_BATCH_SIZE = 2 ** 15
+TOTAL_BATCH_SIZE = 2 ** 14
 EMBEDDING_LR = 1.0
 UNEMBEDDING_LR = 0.004
 MATRIX_LR = 0.05
```

Hypothesis: halving the global batch (32768 → 16384 tokens) doubles the optimizer-step count within the time budget and speeds convergence on a small model. The result beat the *original* baseline but landed just above the new SSSS base (0.519890 vs 0.519684) — the extra steps were offset by noisier gradients at unchanged learning rates. Discarded as a wash. Note the moving target: an idea that would have "won" an hour earlier now loses, because keeps compound.

### 5–9. The matrix-LR bracketing search

The next five experiments are a textbook **bracketing search** on a single scalar, `MATRIX_LR` — the learning rate for the Muon-managed weight matrices. Watch the pattern: probe one direction, probe the other, bracket the optimum between a too-low and a too-high point, then bisect.

#### 5. `MATRIX_LR` 0.06 — `b240c00` (16:35) · 0.520477 vs 0.519684 · discard

```diff
diff --git a/train.py b/train.py
index af7e743..de51bb8 100644
--- a/train.py
+++ b/train.py
@@ -807,7 +807,7 @@ WINDOW_PATTERN = "SSSS"   # sliding window pattern: L=full, S=half context
 TOTAL_BATCH_SIZE = 2 ** 15
 EMBEDDING_LR = 1.0
 UNEMBEDDING_LR = 0.004
-MATRIX_LR = 0.05
+MATRIX_LR = 0.06
 SCALAR_LR = 0.5
 WEIGHT_DECAY = 0.1
 ADAM_BETAS = (0.8, 0.95)
```

First probe: is the short run LR-limited — can it tolerate 20% hotter? No. 0.520477 is worse than both the baseline and the SSSS base, so 0.05 was already at or past the sweet spot on the high side. One number, but it orients the whole search: look *down*.

#### 6. `MATRIX_LR` 0.04 — `d07a0d2` (16:43) · 0.519105 vs 0.519684 · **keep**

```diff
diff --git a/train.py b/train.py
index af7e743..3fa75d3 100644
--- a/train.py
+++ b/train.py
@@ -807,7 +807,7 @@ WINDOW_PATTERN = "SSSS"   # sliding window pattern: L=full, S=half context
 TOTAL_BATCH_SIZE = 2 ** 15
 EMBEDDING_LR = 1.0
 UNEMBEDDING_LR = 0.004
-MATRIX_LR = 0.05
+MATRIX_LR = 0.04
 SCALAR_LR = 0.5
 WEIGHT_DECAY = 0.1
 ADAM_BETAS = (0.8, 0.95)
```

The opposite direction: 20% cooler. Best result of the session so far (0.519105, −0.0006 vs the SSSS base), confirming the optimum sits below the inherited 0.05 and opening a finer search.

#### 7. `MATRIX_LR` 0.035 — `1e3f3e1` (16:51) · 0.520977 vs 0.519105 · discard

```diff
diff --git a/train.py b/train.py
index 3fa75d3..c99517f 100644
--- a/train.py
+++ b/train.py
@@ -807,7 +807,7 @@ WINDOW_PATTERN = "SSSS"   # sliding window pattern: L=full, S=half context
 TOTAL_BATCH_SIZE = 2 ** 15
 EMBEDDING_LR = 1.0
 UNEMBEDDING_LR = 0.004
-MATRIX_LR = 0.04
+MATRIX_LR = 0.035
 SCALAR_LR = 0.5
 WEIGHT_DECAY = 0.1
 ADAM_BETAS = (0.8, 0.95)
```

Keep pushing down — does the trend hold? It doesn't: at 0.035 the model is undertrained within the fixed step budget, worse than even the original 0.05. This "failure" is the most informative run of the sweep: the optimum is now cleanly **bracketed between 0.035 and 0.05**.

#### 8. `MATRIX_LR` 0.045 — `780adb0` (17:00) · 0.518993 vs 0.519105 · **keep**

```diff
diff --git a/train.py b/train.py
index 3fa75d3..b1ea9f8 100644
--- a/train.py
+++ b/train.py
@@ -807,7 +807,7 @@ WINDOW_PATTERN = "SSSS"   # sliding window pattern: L=full, S=half context
 TOTAL_BATCH_SIZE = 2 ** 15
 EMBEDDING_LR = 1.0
 UNEMBEDDING_LR = 0.004
-MATRIX_LR = 0.04
+MATRIX_LR = 0.045
 SCALAR_LR = 0.5
 WEIGHT_DECAY = 0.1
 ADAM_BETAS = (0.8, 0.95)
```

The bisection step: with 0.04 good, 0.035 bad, and 0.05 mediocre, try the midpoint of the good side. 0.518993 was the best matrix-LR value evaluated, and it became the base for every remaining experiment — though honestly, the margin over 0.04 (~0.0001) is within run-to-run noise. Single 5-minute runs cannot distinguish 0.04 from 0.045; the sweep found a plateau, not a point.

#### 9. `MATRIX_LR` 0.0475 — `89a8842` (17:08) · 0.520235 vs 0.518993 · discard

```diff
diff --git a/train.py b/train.py
index b1ea9f8..508db8c 100644
--- a/train.py
+++ b/train.py
@@ -807,7 +807,7 @@ WINDOW_PATTERN = "SSSS"   # sliding window pattern: L=full, S=half context
 TOTAL_BATCH_SIZE = 2 ** 15
 EMBEDDING_LR = 1.0
 UNEMBEDDING_LR = 0.004
-MATRIX_LR = 0.045
+MATRIX_LR = 0.0475
 SCALAR_LR = 0.5
 WEIGHT_DECAY = 0.1
 ADAM_BETAS = (0.8, 0.95)
```

One last fine-grained step, midway between the winning 0.045 and the old 0.05. The regression closed out the search: `MATRIX_LR = 0.045` stands as the local optimum (to within measurement noise), and the agent moved on to the other learning-rate groups. Five runs, ~40 minutes, one hyperparameter settled — that's what a disciplined line search costs.

### 10–11. Embedding LR: a two-sided probe that confirms the incumbent

#### 10. `EMBEDDING_LR` 0.8 — `07992b2` (17:16) · 0.519878 vs 0.518993 · discard

```diff
diff --git a/train.py b/train.py
index b1ea9f8..db9969f 100644
--- a/train.py
+++ b/train.py
@@ -805,7 +805,7 @@ WINDOW_PATTERN = "SSSS"   # sliding window pattern: L=full, S=half context
 
 # Optimization
 TOTAL_BATCH_SIZE = 2 ** 15
-EMBEDDING_LR = 1.0
+EMBEDDING_LR = 0.8
 UNEMBEDDING_LR = 0.004
 MATRIX_LR = 0.045
 SCALAR_LR = 0.5
```

Now the AdamW-managed embedding table: is its aggressive LR of 1.0 overshooting? Lowering it 20% cost ~0.0009, suggesting the fast-moving embeddings genuinely help in a short run.

#### 11. `EMBEDDING_LR` 1.2 — `cb8fe91` (17:24) · 0.521757 vs 0.518993 · discard

```diff
diff --git a/train.py b/train.py
index b1ea9f8..fd80dba 100644
--- a/train.py
+++ b/train.py
@@ -805,7 +805,7 @@ WINDOW_PATTERN = "SSSS"   # sliding window pattern: L=full, S=half context
 
 # Optimization
 TOTAL_BATCH_SIZE = 2 ** 15
-EMBEDDING_LR = 1.0
+EMBEDDING_LR = 1.2
 UNEMBEDDING_LR = 0.004
 MATRIX_LR = 0.045
 SCALAR_LR = 0.5
```

The mirror test: if lower hurt, maybe higher helps. It was the worst learning-rate result of the session — 1.2 destabilizes embedding training. With *both* directions worse, `EMBEDDING_LR = 1.0` is confirmed near-optimal, and that pair of failures is a positive result: this knob is done.

### 12–13. Unembedding LR: same probe, one borderline call

#### 12. `UNEMBEDDING_LR` 0.003 — `407918e` (17:33) · 0.520669 vs 0.518993 · discard

```diff
diff --git a/train.py b/train.py
index b1ea9f8..1d11d29 100644
--- a/train.py
+++ b/train.py
@@ -806,7 +806,7 @@ WINDOW_PATTERN = "SSSS"   # sliding window pattern: L=full, S=half context
 # Optimization
 TOTAL_BATCH_SIZE = 2 ** 15
 EMBEDDING_LR = 1.0
-UNEMBEDDING_LR = 0.004
+UNEMBEDDING_LR = 0.003
 MATRIX_LR = 0.045
 SCALAR_LR = 0.5
 WEIGHT_DECAY = 0.1
```

Same two-sided probe on the output projection, first 25% lower. The ~0.0017 regression says the output head needs its learning speed — slow it down and the layer that turns the model's internal state into next-token predictions hasn't finished calibrating within the 5 minutes.

#### 13. `UNEMBEDDING_LR` 0.005 — `58af6ef` (17:41) · 0.519294 vs 0.518993 · discard

```diff
diff --git a/train.py b/train.py
index b1ea9f8..7413829 100644
--- a/train.py
+++ b/train.py
@@ -806,7 +806,7 @@ WINDOW_PATTERN = "SSSS"   # sliding window pattern: L=full, S=half context
 # Optimization
 TOTAL_BATCH_SIZE = 2 ** 15
 EMBEDDING_LR = 1.0
-UNEMBEDDING_LR = 0.004
+UNEMBEDDING_LR = 0.005
 MATRIX_LR = 0.045
 SCALAR_LR = 0.5
 WEIGHT_DECAY = 0.1
```

The upward direction handily beat the original baseline but fell ~0.0003 short of the running best — so under the strict beat-the-best rule, it was discarded. This is one of the session's most borderline calls (alongside experiment 4): 0.0003 is arguably within noise of the incumbent, and a protocol with repeated runs might have kept it. Simple decision rules buy consistency at the price of occasionally discarding a tie.

### 14–16. The warmdown dose-response curve

The last three experiments (plus the 0.45 baseline they pivot around) sweep `WARMDOWN_RATIO` — the fraction of training spent linearly annealing the learning rate down. Together they trace a clean dose-response curve:

| `WARMDOWN_RATIO` | 0.35 | 0.45 (base) | 0.55 | 0.65 |
|---|---|---|---|---|
| val_bpb | 0.520582 | 0.518993 | **0.518708** | 0.523293 |

#### 14. Warmdown 0.35 — `742986f` (17:50) · 0.520582 vs 0.518993 · discard

```diff
diff --git a/train.py b/train.py
index b1ea9f8..8242e3c 100644
--- a/train.py
+++ b/train.py
@@ -812,7 +812,7 @@ SCALAR_LR = 0.5
 WEIGHT_DECAY = 0.1
 ADAM_BETAS = (0.8, 0.95)
 WARMUP_RATIO = 0.02
-WARMDOWN_RATIO = 0.45
+WARMDOWN_RATIO = 0.35
 FINAL_LR_FRAC = 0.1
 
 # Model size + memory defaults
```

Hypothesis: a shorter warmdown means more time at peak LR, covering more ground. The regression showed the opposite — this model benefits from a long, gentle anneal that lets the weights settle before evaluation.

#### 15. Warmdown 0.55 — `e9fffd9` (17:58) · 0.518708 vs 0.518993 · **keep — session best**

```diff
diff --git a/train.py b/train.py
index b1ea9f8..506284e 100644
--- a/train.py
+++ b/train.py
@@ -812,7 +812,7 @@ SCALAR_LR = 0.5
 WEIGHT_DECAY = 0.1
 ADAM_BETAS = (0.8, 0.95)
 WARMUP_RATIO = 0.02
-WARMDOWN_RATIO = 0.45
+WARMDOWN_RATIO = 0.55
 FINAL_LR_FRAC = 0.1
 
 # Model size + memory defaults
```

The complementary test: extend the warmdown to 55% of training. Consistent with the 0.35 failure, more anneal time helped — **0.518708**, the session's best result and its final kept configuration: `SSSS` attention + `MATRIX_LR 0.045` + `WARMDOWN_RATIO 0.55`. Net improvement over the session's opening baseline: 0.001388 bits/byte from four kept one-line changes.

#### 16. Warmdown 0.65 — `f588ed5` (committed 18:07, evaluated 75 days later) · 0.523293 vs 0.518708 · discard

```diff
diff --git a/train.py b/train.py
index 506284e..2e12195 100644
--- a/train.py
+++ b/train.py
@@ -812,7 +812,7 @@ SCALAR_LR = 0.5
 WEIGHT_DECAY = 0.1
 ADAM_BETAS = (0.8, 0.95)
 WARMUP_RATIO = 0.02
-WARMDOWN_RATIO = 0.55
+WARMDOWN_RATIO = 0.65
 FINAL_LR_FRAC = 0.1
 
 # Model size + memory defaults
```

The natural continuation of the line search: with 0.35 worse and 0.55 the new best, push to 0.65. (Note this diff builds on the 0.55 state, index `506284e`, unlike experiments 9–15, which all diffed against the 0.045 base `b1ea9f8`.) The agent committed it at 18:07:11 on 2026-05-23 — and then the session was interrupted before the run ever executed. The commit sat on `master`, queued but unevaluated, for **75 days**. On 2026-08-06 it was finally run: 0.523293, the worst warmdown arm by far, completing the dose-response curve with a steep upturn past 0.55. Honesty requires two caveats: a 75-day gap means the software environment may have drifted, so this number is less comparable to May's than May's runs are to each other; and because later commits had by then landed on top of `f588ed5`, the discard could not be a `git reset` — it was a forward revert (`e1e15e4`, warmdown 0.65 → 0.55) followed by the log commit (`95fe21a`). Same decision, different git mechanics — see War story 3.

## War stories

Three incidents from this project that teach more than the clean results do.

### 1. The TSV-wipe bug: don't write your log inside the transaction you're rolling back

The loop's original program recorded the result in `results.tsv` *and then*, on a discard, ran `git reset --hard` back to the pre-experiment commit. See the bug? The reset rolled back the scoreboard row along with the experiment — the discard path erased its own log entry. Failed experiments kept vanishing from the record as if they had never run. A later cleanup pass reconstructed what it could from timestamps and commit messages, but the scores were gone for good: 13 rows of the pre-fix scoreboard survive only as score-less placeholders with status `missing` (see for yourself with `git show 0aa0e07:results.tsv`).

The fix (`0aa0e07`, 2026-05-22 — the day before this session) reordered the loop: reset first if discarding, *then* write and commit the TSV row, "AFTER any git reset so the TSV update is never undone." It's the classic transactional mistake — writing the audit log inside the transaction you might roll back — wearing a git costume. The reason all 16 rows of the may23-pm session survive, discards included, is that this session was the first full one run under the fixed program.

### 2. The ghost commits: the scoreboard referenced commits that didn't exist

The discard mechanic (`git reset --hard`) has a side effect: the discarded experiment commit becomes unreachable — no branch or tag points at it, and it was never pushed. So the public repo's `results.tsv` faithfully listed rows like `b240c00 0.520477 discard` while `b240c00` itself existed *nowhere on GitHub*. Anyone auditing the scoreboard would find ten hashes that resolve to nothing: a lab notebook citing specimens that had been thrown away.

The commits weren't gone yet, though — git keeps a machine-local journal of recent history called the reflog, and unreachable commits linger there until git's garbage collector destroys them. Before that could happen, all ten were pinned with local tags (`rescue/<hash>`, one per discard; run `git tag -l 'rescue/*'`). Every failed-experiment diff quoted in this document — the depth-7 disaster, the LR probes, all of it — was recovered that way. Without the rescue, each failure would be a one-line TSV row with no recoverable code, and this walkthrough could only have told you the success story. The general lesson: a keep-only history is a biased history. If your process deletes failures, your process deletes most of what happened.

### 3. The interrupted experiment: the repo *was* the resume file

Experiment 16 (`f588ed5`, warmdown 0.65) was committed at 18:07 on 2026-05-23 and then the session stopped — the run never started. The repo sat for 75 days with an unevaluated experiment at the top of `master`.

Here's the payoff of the "git is the state machine" design: resuming required no reconstruction at all. The repo at HEAD *was* the exact experiment state — `train.py` already contained `WARMDOWN_RATIO = 0.65`, the commit already named the hypothesis, and `results.tsv` already showed what the score had to beat (0.518708). On 2026-08-06 the resume was simply: run `train.py`, read the score (0.523293), decide (worse → discard), log it. The only wrinkle was mechanical: with later commits stacked on top, the discard used a revert commit instead of a reset — which is why `f588ed5`, uniquely among the session's failures, remains visible in `master`'s history rather than surviving only as a rescue tag. A loop whose entire state lives in the repo is a loop you can abandon mid-stride, for two and a half months, and pick up with one command.

## Where to go next

- **See the models, not just the scores — nothing to install.** The session's anchor and its best run are both up right now at **https://autoresearch-demo.fly.dev/**: **Baseline** (`75027e8`, 0.520096) and **Best** (`e9fffd9`, 0.518708), side by side in the browser. Give both the same prompt and read what those 0.001388 bits per byte bought — the scores in the session table are the models you are typing at. The idle machine takes ~12 seconds to wake on the first load, then it is instant; generation there is capped (max 500 tokens, top-k ≤ 200), and the progress chart on the page is the same 16 rows.
- **Or run the same UI locally.** If you cloned the repo, `uv run chat.py` opens it at `http://localhost:8000` with no caps and every archived checkpoint available, not just those two. The hash in each checkpoint's filename is the same join key used throughout this document.
- **Read [`program.md`](program.md).** It is the *entire* program the agent runs — the loop you just watched execute 16 times, including the exact keep/discard rules and the post-bugfix TSV ordering from War story 1. It's shorter than this walkthrough.
- **Get hands-on.** The starter kit at [aroughidea/autoresearch-starter](https://github.com/aroughidea/autoresearch-starter) packages this same loop to set up from scratch — baseline configuration, empty scoreboard, a no-GPU Colab path, and "Use this template" so your copy starts clean. The improvements this session found are yours to rediscover.
