# Training Decisions: a learning journey for autoresearch

Status: in progress · components 1 (capture) and 2 (dataset and tokenizer options) built · updated 2026-10-07 · Thomas J McLeish with Claude

## Learning objectives

**Core objective:** participants understand that *changing a design decision changes the final
experience in a specific way*. The goal is understanding: they are not asked to find a best
recipe or to hand in explanations.

After the demo and the journey, participants can:

1. **Follow how a model gets made:** get and understand a dataset, set the tokens, build a
   model, use it, and iterate on the build at the algorithmic level.
2. **Speak to how a dataset's qualities show up in the experience,** having trained on one
   dataset and later on another. The score cannot compare datasets; the writing can.
3. **Speak to how the tokenizer shapes the model:** vocabulary size and origin change the
   model's size, how far it trains in a fixed time, and how it writes.
4. **Experience consequences and compare decision sets:** watch a model grow over training
   time, and set two recipes side by side.
5. **Compare models through use,** in the chat page, alongside the score.
6. **Understand AI-driven research:** the agent changes the algorithm, so no deep expertise is
   needed; it evolves the recipe run by run and keeps or discards each change by a metric
   (`val_bpb`), because many of its gains are too small to read.
7. **Know where training happens and how it is recorded:** their own hardware, rented or cloud
   training services, and GitHub as the lab record.

## Purpose

Workshop participants understand that **changing a design decision changes the final
experience in a specific way**. Workshops 1 and 2 already describe how a model gets made,
in the abstract. This journey makes it concrete: pick a dataset, set the tokens, build a
model, use it, and let an AI agent iterate on the algorithm, so no deep expertise is needed.

The goal is understanding. Learners are not asked to choose a "best" recipe, and they do not
hand in written explanations.

## Who and where

- **Audience:** participants in A Rough Idea workshops (designers and other
  non-specialists), reached through workshops.aroughidea.com and llmexplorables.aroughidea.com.
- **In class: explore.** No training, no GPU, nothing to install. A browser explorer over
  models trained ahead of time.
- **Out of class: a run.** Autoresearch overnight on the learner's own PC or on a rented
  GPU (a pod, or a Hugging Face Job once tested; see "Remote training"). The agent does the
  research. Colab was tested and dropped.
- **Back in the explorer:** the learner's own run loads next to the library, so they see what
  their decisions and their agent's changes did.

## The decisions

One dataset per run. Datasets are compared across runs, never mixed inside one.

| Decision | Options | Who makes it |
|---|---|---|
| Dataset | TinyStories (children's stories, GPT-4 written) · Folktales (folk and myth tales) | Learner, per run |
| Tokenizer | Built from the dataset (8,192) · Phi-3 / Llama 2 (32,011) · GPT-2 (50,257) | Learner, per run |
| Training time | Snapshots at 0 s, 10 s, 30 s, 1 min, 2 min, 5 min | Fixed; shown as growth |
| Algorithm | Baseline recipe, then the agent's changes (learning rates, attention pattern, schedule, model shape) | The agent |
| Hand-set recipe | Learner edits settings and trains once | Stretch goal only |

Stretch tokenizers: the frontier vocabularies the explorables already show (128k to 248k),
only if the check shows they fit on a consumer GPU.

## The consequences we show

Every run is described the same way, so any two can be compared:

1. **Writing.** The model's continuation of the same fixed prompts, with the same
   decoding settings (fixed seed, temperature, top-k) at every snapshot and in every run.
   Only the model differs.
2. **Score** (`val_bpb`, bits per byte, lower is better). Comparable across tokenizers and
   recipes on the **same dataset**. Not comparable across datasets; the page says so, and
   the writing is the evidence there.
3. **Cost.** Training time, peak GPU memory, model size, share of the model spent on the
   vocabulary table.
4. **Tokens.** How each tokenizer splits the prompts, and how much of its vocabulary the
   dataset never uses.

## The journey

1. **Workshop 2, "How a model gets made"** (existing, abstract) links to the explorer.
2. **In class: Training Decisions explorer** (new llm-explorables sketch).
   - *Growth:* one model from noise to stories, scrubbed over training time.
   - *Compare:* two runs on the same dataset, decisions diffed, writing side by side,
     growth in sync.
   - *Tokens:* the same sentence through each tokenizer, plus each untrained model's first
     words (a portrait of its vocabulary).
   - *Evolution:* an agent session in order. A staircase of scores: the best-so-far line
     steps down at each kept change, discarded experiments sit above it as hollow dots, each
     labelled with the agent's change in plain words. Click one to see its writing next to the
     best before it. The view says plainly that these gains are too small to read (the May
     session improved 0.27% in all), which is why the agent needs a score.
   - Live chat with a chosen model stays on the hosted demo (autoresearch-demo.fly.dev).
3. **Out of class: the starter kit** (autoresearch-starter, simplified). Pick a dataset and
   tokenizer, start the agent, sleep. Own PC or a rented GPU.
4. **Back in the explorer:** load your run from your GitHub repo URL or by dropping the
   folder. Your runs appear next to the library.
5. **Later:** a second night on the other dataset. Now the dataset itself can be compared.

## Components

Each is its own design, plan and build cycle, in this order.

1. **Capture** (autoresearch-win-rtx, then starter). Logic lives in a new `capture.py` that
   the agent may not modify, like `prepare.py`; `train.py` only calls it at three points
   (each step before the clock starts, end of training, after the score). Samples the fixed
   prompts in-process at 0, 10, 30, 60 and 120 s of training time and at the end, with the
   clock paused and with its own random generator, so training and scores are identical to an
   uncaptured run. Writes one small run file per experiment (`runs/<timestamp>_<commit>.json`),
   which the agent commits with the scoreboard row in its log commit, so it survives the
   revert of a discarded experiment. No extra weights saved. Keep or discard status and the
   agent's description stay in `results.tsv`; the explorer joins the two on the commit.
2. **Dataset and tokenizer options** (`prepare.py`). Add Folktales. Add a tokenizer choice:
   built from data, Phi-3/Llama 2, GPT-2. The evaluation stays fixed and read-only to the
   agent.
3. **Library.** Overnight on TJ's GPU: both datasets × three tokenizers, baseline recipe,
   plus one agent session per dataset. Published as run files.
4. **Explorer** (llm-explorables, `/training/`). Static p5/DOM sketch in the house style.
   Reads run files and `results.tsv` from its own folder, a GitHub repo, or a dropped folder.
   Growth, Compare, Tokens and Evolution views. Presentation mode for the projector.
5. **Tokens in the explorables.** Add the home-made vocabularies, Phi-3/Llama 2 and GPT-2
   to `/tokens/` and the Tokenizer Map, next to the eight already there.
6. **Starter kit and docs.** Agent-first README: try the demo, explore, run your own.
   Dataset and tokenizer flags. Hand-set training moves to "going further". The worked
   example's README shrinks to one screen pointing at the three paths.
7. **Workshops site.** A page that sits after "How a model gets made"; Going Further quest 5
   rewritten to link the explorer, the starter kit and the demo.
8. **Hosted chat (optional).** The Fly demo serves any library model in its two panes.

## Run file (shared contract between capture, library, starter kit and explorer)

```json
{
  "schema": 1,
  "run_id": "20261007T013000-0700_75027e8",
  "commit": "75027e8",
  "created": "2026-10-07T01:40:12-07:00",
  "dataset": "tinystories",
  "tokenizer": { "name": "own", "vocab_size": 8192 },
  "recipe": { "DEPTH": 6, "WINDOW_PATTERN": "SSSL", "MATRIX_LR": 0.05, "WARMDOWN_RATIO": 0.45,
              "n_layer": 6, "n_embd": 384, "vocab_size": 8192 },
  "decoding": { "seed": 1234, "temperature": 0.8, "top_k": 40, "min_chars": 320, "max_tokens": 300 },
  "prompts": ["Once upon a time", "The old king said", "In the dark forest", "The little girl found a"],
  "snapshot_times": [0, 10, 30, 60, 120],
  "snapshots": [
    { "t_s": 0, "step": 0, "train_loss": null, "samples": ["", "", "", ""] }
  ],
  "final": { "val_bpb": 0, "training_s": 300, "peak_vram_mb": 0, "num_steps": 0,
             "params_m": 0, "sampling_s": 0 }
}
```

Zeros and empty strings above mark fields, not values. Keep/discard and the agent's description stay in `results.tsv`, joined on `commit`. The share of the model spent on the vocabulary is computed by the explorer from `recipe`. Component 2 adds a tokenizer `source`.

## Risks and open questions

- **Are the differences visible?** Unknown until the check. If writing looks alike between
  tokenizers or snapshots, the explorer has nothing to show and the decisions must change.
- **Memory.** Measured in the check: Phi-3 (32k) peaks at 6.8 GB and GPT-2 (50k) at 9.3 GB,
  both at half batch size. GPT-2 does not fit on 8 GB cards. Frontier vocabularies do not fit.
- **Folktales is small.** The model may memorize it within 5 minutes. That is itself a
  dataset consequence worth showing, but needs checking.
- **Licenses.** Confirm the Folktales dataset license and the Phi-3 tokenizer license
  before redistributing run files and vocabularies.
- **Sampling cost.** Generating samples six times per run must not eat the training budget;
  the clock pauses, but total wall time grows. Measure in the check.

## The check (first, throwaway)

Scratch worktree, nothing pushed, about an hour on TJ's GPU. Four runs with growth samples:

1. TinyStories, built from data (8,192)
2. TinyStories, Phi-3 / Llama 2 (32,011)
3. TinyStories, GPT-2 (50,257)
4. Folktales, built from data

Pass: a person reading the samples can see growth over time, and can see a difference
between at least two of the tokenizer runs and between the two datasets. Report peak memory
and wall time per run.

## Check results (2026-10-06, RTX 4000 Ada laptop GPU, 12 GB)

Same recipe, 5 minutes of training each, snapshots with the clock paused. The recipe was the
session best in `train.py` (May score 0.518708), not the baseline, so the built-from-data
TinyStories run's 0.520082 is 0.0014 above May on this machine, not a match with the baseline's
0.520096 as first reported. A later A/B with in-loop capture (capture plan, Task 6) scored
0.521174 captured against 0.524005 uncaptured: run-to-run noise from steps fitted into 300 s
(640 vs 623) is about 0.003, larger than the agent's whole May improvement (0.0014). Capture's
effect on the score is bounded only within that noise.

| Run | Vocabulary | Model | Steps | Score | Peak memory | Wall time |
|---|---|---|---|---|---|---|
| TinyStories, built from data | 8,192 | 18.9 M | 641 | 0.520 | 6.8 GB | 7.7 min |
| TinyStories, Phi-3 / Llama 2 | 32,011 | 46.3 M | 437 | 0.570 | 6.8 GB (half batch) | 9.1 min |
| TinyStories, GPT-2 | 50,257 | 67.3 M | 340 | 0.584 | 9.3 GB (half batch) | 10.4 min |
| Folktales, built from data | 8,192 | 18.9 M | 630 | 1.389 (other dataset) | 6.8 GB | 8.4 min |

- **Growth: pass.** Noise at 0 s, word salad at 10 s, grammar at 30 s, stories by 1 to 2 min.
- **Dataset: pass.** Lily and Mia at the park versus kings and viziers, on the same prompts.
- **Tokenizer: pass on the evidence chain, subtler in the finished writing.** Bigger vocabulary,
  bigger model, fewer steps, 10 to 12% worse score. The writing gap reads best at 10 to 30 s.
- **Vocabulary use on 100,000 TinyStories:** unused 4.9% (built from data), 70.4% (Phi-3),
  69.4% (GPT-2). Characters per token 4.18, 3.77, 4.06: the home-made vocabulary is smallest
  and packs the most text per token.
- **Untrained models speak their vocabulary.** At 0 s the random model's output is a portrait
  of the tokenizer: story words, folk-tale words, multilingual fragments (Phi-3), web and code
  fragments (GPT-2). Unplanned; make it part of the Tokens view.
- **Folktales is read 8 times in 5 minutes** and some phrases come back verbatim from the
  training text. Show this on purpose: mark copied phrases.
- GPT-2's vocabulary does not fit on 8 GB cards. Folktales card says CC0 1.0.

Design changes that follow: first-words view in Tokens; tokenizer comparisons open at 30 s;
copied-phrase marking for small datasets.

Scratch code (not for merge): worktree `../autoresearch-spike`, branch
`spike/training-decisions-check`.

## Remote training (updated 2026-10-07)

The overnight run needs a machine that stays up for hours with the agent on it.

- **Colab: tested and dropped.** On Colab Pro with a T4, setup, the pinned PyTorch and the
  agent loop all worked, and Claude Code signed in with a subscription via
  `claude setup-token`. But free Colab has no supported terminal and its sessions end when
  the tab closes, so it cannot host an overnight run. The starter's notebook was removed.
- **The T4 model did not train.** Root cause: the fp16 path (every GPU without bf16, including
  the RTX 20-series the starter supports) had no loss scaling. Fixed with dynamic loss scaling
  in `train.py` (worked example #6, starter #1); verified by forcing fp16 on an RTX 4000 Ada,
  not yet on real Turing or T4 hardware.
- **Rented pod** (RunPod, Lambda, Vast.ai): documented in the starter's `HARDWARE.md`. Needs
  SSH comfort.
- **Hugging Face Jobs: the likely default, pending a test.** Pay per second from prepaid credit,
  no subscription. bf16 GPUs: L4 24 GB at $0.80/h, A10G 24 GB at $1.00/h (about $6.40-8 a
  night). One command (`hf jobs run --flavor l4x1 --timeout 10h --secrets ...`), logs in the
  browser, SSH available. The night runs as one job: clone the learner's copy, set up, run the
  agent headless (`claude -p`), push results to GitHub. Needs three accounts: Hugging Face with
  credit, GitHub with a push token, Claude. Default job timeout is 30 minutes: always pass
  `--timeout`.

## Component 2 design: dataset and tokenizer options

**What the learner does:** `uv run prepare.py --dataset folktales --tokenizer phi3`. That
downloads what is needed, builds or fetches the tokenizer, and makes the pair active. Every
later `train.py`, `generate.py` and `chat.py` run uses the active pair. The agent never changes
it (`prepare.py` is read-only to the agent; `program.md` says the pair is the human's decision).

| Option | Values |
|---|---|
| `--dataset` | `tinystories` (default) · `folktales` |
| `--tokenizer` | `own` (default: BPE 8,192 trained on the dataset) · `phi3` (Phi-3 mini / Llama 2, 32,011 tokens) · `gpt2` (50,257) |

- **Resolution order,** for both dataset and tokenizer: command-line flag, then environment
  variable (`AUTORESEARCH_DATASET`, `AUTORESEARCH_TOKENIZER`), then the cache's active file
  (`active_dataset.txt`, `active_tokenizer.txt`), then the default.
- **Cache layout:** `datasets/<dataset>/tokenizer/` for `own` (unchanged, so existing caches keep
  working) and `datasets/<dataset>/tokenizer-<name>/` for the others.
- **Special tokens:** standard tokenizers get the same four reserved control tokens appended
  (BOS, EOS and two spares), so `phi3` loads as 32,015 ids and `gpt2` as 50,261.
- **Folktales:** `merve/folk-mythology-tales` (`merged_clean.txt`, CC0 1.0 per its card),
  paragraphs packed into documents of up to 1,500 characters: 9,195 documents, 12.4 M
  characters. The first 300 documents are validation; there is no test split.
- **Phi-3 tokenizer:** Microsoft's `tokenizer.json` (MIT), loaded with the Hugging Face
  `tokenizers` library, a new dependency. Its score counts one extra byte per document for the
  leading word-boundary marker, about a 0.1% bias; documented, not corrected.
- **`train.py`: one small change.** `Tokenizer.from_directory()` follows the active pair and the
  tokenizer object carries its `name` and `source`, which `capture.py` writes into the run
  file. The only `train.py` change is a bug fix found in verification: the GPU-tuning cache
  key now includes vocabulary size (a batch size tuned for 8,192 tokens was reused for
  Phi-3's 32,015, needing 13 GB on a 12 GB card; Windows spilled silently and the score fell
  from 0.574 to 0.710).
- **The scorer guards the pair.** `evaluate_bpb` in `prepare.py` refuses to score any pair other
  than the active one, so the agent cannot switch pair through its own `train.py`.
- **Known gaps, deferred:** single Folktales paragraphs longer than 1,500 characters stay whole
  (about 1.8% of tokens rarely train); `chat.py` does not check a checkpoint against the active
  pair; the tuning-cache key still ignores model size, so an agent that deepens the model can
  reuse a smaller model's batch size and spill silently on Windows (next fix).
- **Out of scope:** chatting with models trained on different pairs side by side (`chat.py`
  uses the active pair); frontier vocabularies.
- **Memory warning:** `prepare.py` warns that `gpt2` needs about 10 GB of GPU memory.

## Component: chat page upgrade (2026-10-07)

Added after the learning-objectives audit, ahead of the library: the chat page is where a
participant meets models today, so it should show a decision whose effect can be read.

- **Each model loads with its own pair.** An archived checkpoint `checkpoints/<stem>.pt` and
  its run file `runs/<stem>.json` share a stem (`<timestamp>_<commit>`), so the run file names
  the dataset and tokenizer. `checkpoint_pre_eval.pt` (the latest run) uses the active pair;
  archived checkpoints without a run file predate the choice and use the defaults. A model
  whose vocabulary size differs from its tokenizer's is listed as unavailable, never decoded
  as garbage.
- **Pickers on both panes** (defaults stay first kept and best), and **labels with the facts:**
  score, dataset, tokenizer, size, and the agent's description from `results.tsv`.
- **Watch it learn:** for a model with a run file, step through what it wrote at 0, 10, 30, 60
  and 120 s and at the end, on its four fixed prompts.
- **The header and the vocabulary browser follow the left pane's model.**
- Out of scope: new contrasts on the hosted demo (they need the library and a working deploy).

## Component 3 design: library (2026-10-07)

The models the explorer shows in class, trained ahead of time on TJ's GPU.

- **Where:** `library/` in the worked example. Each collection is a folder shaped like a
  learner's own repo (`runs/`, plus `results.tsv` when an agent made it), so the explorer reads
  the library and a learner's run the same way.
  - `library/baselines/runs/`: six run files, both datasets × three tokenizers, the starter
    kit's recipe (`SSSL`, `MATRIX_LR` 0.05, `WARMDOWN_RATIO` 0.45), no agent. Made by
    `library/make_baselines.sh` on branch `library/baselines`. They share one commit, so they
    carry no `results.tsv`: their run files hold the dataset, tokenizer and score.
  - `library/sessions/<dataset>/`: one overnight agent session per dataset, on the dataset's
    own tokenizer, starting from the starter's recipe and following the worked example's
    `program.md`. Each runs in its own worktree on a branch `session/<dataset>-<date>`, pushed
    as the lab record; its `results.tsv` and `runs/` are copied into the library.
- **Weights stay out of git** (`library/checkpoints/`, ignored): Phi-3 and GPT-2 checkpoints are
  150 to 230 MB each. The hosted chat (component 8) decides later which to serve.
- **`library/report.py`** writes what the browser cannot compute, from the local cache:
  `tokens.json` (each tokenizer's split of the four prompts, characters per token, share of the
  vocabulary the dataset never uses) and `copies.json` (spans of each sample that repeat eight or
  more words of the training text verbatim, for the copied-phrase marks).
- **Licences, recorded in `library/README.md`:** TinyStories CDLA-Sharing-1.0; Phi-3 tokenizer
  MIT; GPT-2 MIT; Folktales' card says CC0 1.0, though its source (D. L. Ashliman's Folktexts)
  carries its own copyright notice. The library publishes only generated samples and scores,
  never dataset text.
- **Results (2026-10-08):** the six baselines are in `library/README.md`. TinyStories, built
  from data, scored 0.5209 against May's 0.5201, within the 0.003 noise, so the baseline
  reproduces. On Folktales the order reverses: the borrowed vocabularies score better, with
  model size changing alongside. The Folktales agent session took 1.3859 to 1.2637 in 12
  experiments (about 41 times the noise), mostly by raising the matrix learning rate; it ran two
  hours before the headless agent's sign-in expired.
- **Learned while building:** the four prompts split alike in every tokenizer, so `tokens.json`
  adds four contrast sentences; the active dataset is machine-wide, so nothing else may train
  while a session runs; a headless agent (`claude -p`) must wait for training in the foreground,
  because ending its turn ends the session.

## Component 4 design: explorer (2026-10-07)

`/training/` in llm-explorables: a static page with no sign-in and no `/api`. It follows the
Tokenizer Map's shape: a DOM page (p5 `noCanvas`), D3 for the one chart, data fetched from its
own folder, `<base href="/training/">` so `/w/<slug>/training/` works, presentation mode
through `shared/Modes.js`. It is added to the index's sidebar and table, and to `Index.test.js`.

- **Data:** `training/library/` is a copy of the worked example's `library/`, made by
  `scripts/sync-training-library.mjs` from a pinned commit, with an `index.json` listing the
  collections and files (static hosting cannot list a folder).
- **Growth.** One run. A scrubber over 0 s, 10 s, 30 s, 1 min, 2 min and the end; the four
  prompts' writing at that moment. At 0 s the caption says the untrained model speaks its
  vocabulary. Copied phrases are marked where `copies.json` has them.
- **Compare.** Two runs, writing side by side on one shared scrubber, and their decisions as a
  diff (dataset, tokenizer, any recipe value that differs). Scores sit side by side only when
  the datasets match; otherwise the page says the scores do not compare and to read the
  writing. Tokenizer comparisons open at 30 s, where the check found the gap easiest to read.
- **Tokens.** For one dataset: the same prompt split by each tokenizer, the token counts,
  characters per token and unused share from `tokens.json`; then each untrained model's first
  words (the 0 s samples) as a portrait of its vocabulary.
- **Evolution.** An agent session from its `results.tsv`: the best-so-far staircase, discarded
  experiments as hollow dots above it, each labelled with the agent's description, and a band
  showing the 0.003 noise. Clicking a point shows its writing beside the best before it (when
  that experiment has a run file). The caption says plainly that these gains are too small to
  read, which is why the agent needs a score. The May session works here as is (16 rows, no
  run files, so no writing on click).
- **Your own runs.** Load a public GitHub repo by URL (the REST contents API lists `runs/`;
  unauthenticated, 60 requests an hour) or drop the repo folder on the page. Your runs join the
  pickers under "Your runs". Nothing leaves the browser.
- **Tests:** `node --test` on the pure parts (`results.tsv` parsing, the staircase, the recipe
  diff, `index.json` validation), plus a `Page.test.js` pinning the base and script order, and
  a presentation-mode layout snapshot like the other pages.
- **Out of scope:** live chat (stays on the hosted demo), private repos, training in the browser.
- **Build order:** the explorer is built against the baselines and the May session, so it does
  not wait for the overnight sessions.
