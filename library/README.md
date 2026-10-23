# The library

Small models trained ahead of time, so a class can explore what a training decision does
without a GPU. The [Training Decisions explorer](https://llmexplorables.aroughidea.com/training/)
reads this folder; `TRAINING-DECISIONS.md` at the repo root says what each decision is.

Each collection is a folder shaped like a session's record: `runs/` (one run file per
training run, written by `capture.py`) and, when an agent made it, `results.tsv`.

| Collection | What it holds |
|---|---|
| `baselines/` | Six runs: TinyStories and Folktales, each with three tokenizers. The windowed-attention starter recipe, five minutes each, no agent. Only the dataset and the tokenizer change. |
| `sessions/tinystories-may2026/` | The agent's 16 experiments (`WALKTHROUGH.md`): 15 on 23 May 2026 and one on 6 August. Made before run files existed, so it has scores but no writing. |
| `sessions/folktales-oct2026/` | The agent's 12 experiments on Folktales (own tokenizer) on the evening of 7 October 2026, from the windowed-attention starter recipe, with a run file for each. |
| `study-10min/` | Seven runs trained for 10 minutes instead of 5: the six windowed-attention baselines, and the 7 October Folktales session's best recipe. |
| `sessions/tinystories-10min-2026-10-12/` | The agent's 44 experiments on TinyStories (own tokenizer) on 12 October 2026, in 10-minute runs from the windowed-attention starter recipe, with a run file for each and every `train.py` it trained. |
| `sessions/folktales-10min-2026-10-12/` | The agent's 43 experiments on Folktales (own tokenizer) in a 10-hour session from 12 to 13 October 2026, kept the same way. |
| `baselines-b-10min/` | Six runs: TinyStories and Folktales, each with three tokenizers. The starter recipe since 20 October (full attention), ten minutes each, no agent. |
| `baselines-b-5min/` | Two runs: TinyStories and Folktales with their own tokenizers, the full-attention starter recipe at five minutes, to set against the ten-minute baselines. |
| `sessions/tinystories-b-10min-2026-10-20/` | The agent's 49 experiments on TinyStories (own tokenizer) in a 10-hour session from late on 20 October to the afternoon of 21 October 2026, from the full-attention starter recipe, kept the same way. |
| `sessions/folktales-b-10min-2026-10-21/` | The agent's 46 experiments on Folktales (own tokenizer) in a 10-hour session from 21 to 22 October 2026, from the full-attention starter recipe, kept the same way. |

## The six baselines

Trained 7 October 2026 on an RTX 4000 Ada laptop GPU (12 GB), branch `library/baselines`,
with `bash library/make_baselines.sh`, `train.py` set to the starter kit's recipe (commit `6ad8ddd`:
`WINDOW_PATTERN` SSSL, `MATRIX_LR` 0.05, `WARMDOWN_RATIO` 0.45). Two run files carry commit
`61e2270`, made while the series ran, with the same `train.py`: Folktales with GPT-2, and
TinyStories built from the data, which was rerun after its first file was overwritten.

| Dataset | Tokenizer | Vocabulary | Model | Steps in 5 min | Score | Peak memory |
|---|---|---|---|---|---|---|
| TinyStories | built from the data | 8,192 | 18.9 M | 639 | 0.5209 | 3.5 GB |
| TinyStories | Phi-3 / Llama 2 | 32,015 | 46.3 M | 436 | 0.5707 | 6.7 GB |
| TinyStories | GPT-2 | 50,261 | 67.3 M | 344 | 0.5821 | 9.1 GB |
| Folktales | built from the data | 8,192 | 18.9 M | 620 | 1.3788 | 3.5 GB |
| Folktales | Phi-3 / Llama 2 | 32,015 | 46.3 M | 431 | 1.2644 | 6.7 GB |
| Folktales | GPT-2 | 50,261 | 67.3 M | 344 | 1.2581 | 9.1 GB |

Scores (`val_bpb`, bits per byte, lower is better) compare only within one dataset. Two runs
of the same code differ by about 0.003, because five minutes holds a slightly different
number of steps each time.

On TinyStories the home-made vocabulary wins, as in the first check: a bigger vocabulary
makes a bigger model that fits fewer steps into five minutes. On Folktales it is the other
way round: the borrowed vocabularies score better. Why is not tested here. The bigger
vocabularies also make bigger models (46 and 67 M parameters against 18.9 M), so vocabulary and
model size change together in these runs. Read the writing before deciding.

## The Folktales session

Starting from the same recipe as the Folktales baseline, the agent cut the score from 1.3859
to 1.2637 in 12 experiments (8.8%), almost all of it by raising the matrix learning rate from
0.05 to 0.20, and confirmed each keep with a second run. That gain is about 40 times the
run-to-run noise, unlike the May TinyStories session's 0.27%. It ran for two hours, until
the agent's sign-in expired; a thirteenth experiment (depth 6 to 8) trained but was never
logged, so it is not here.

## The ten-minute study

Trained 8 October 2026 on the same GPU, branch `library/study-10min`, with
`bash library/make_study.sh`: the starter kit's recipe (commit `6ad8ddd`) on all six pairs, and the
Folktales session's best recipe (commit `f9352a8`, `MATRIX_LR` 0.20) on Folktales with its own
vocabulary, each for 10 minutes instead of 5. A person set the longer time with
`AUTORESEARCH_TIME_BUDGET=600`; 5 minutes was then the default, and 10 is now. Each run file records `time_budget_s` and adds a moment at 5 minutes.

| Dataset | Tokenizer | Recipe | Steps, 5 → 10 min | Passes through the data | Score, 5 → 10 min | Change |
|---|---|---|---|---|---|---|
| TinyStories | built from the data | starter kit | 639 → 1,210 | 4% → 8% of one | 0.5209 → 0.4774 | 8.4% better |
| TinyStories | Phi-3 / Llama 2 | starter kit | 436 → 859 | 2% → 5% of one | 0.5707 → 0.5143 | 9.9% better |
| TinyStories | GPT-2 | starter kit | 344 → 692 | 2% → 4% of one | 0.5821 → 0.5177 | 11.1% better |
| Folktales | built from the data | starter kit | 620 → 1,237 | 7 → 14 | 1.3788 → 1.8838 | 36.6% worse |
| Folktales | Phi-3 / Llama 2 | starter kit | 431 → 858 | 4.5 → 8.9 | 1.2644 → 1.5791 | 24.9% worse |
| Folktales | GPT-2 | starter kit | 344 → 692 | 3.8 → 7.7 | 1.2581 → 1.5478 | 23.0% worse |
| Folktales | built from the data | the agent's best | 639 → 1,237 | 7 → 14 | 1.2637 → 1.6242 | 28.5% worse |

The 5-minute figures are the baselines' and the session's first run of `f9352a8` (its confirming
run scored 1.2654). Passes are steps times 32,768 tokens, over the dataset's estimated size in
`tokens.json`. Each row is one run; two runs of the same code differ by about 0.003, and the
smallest change here is 0.0435. The run files carry the commit checked out while the study ran
(`5a08dc2` for the first run, `f27f644` for the rest), with `train.py` replaced by each recipe's.

- **On TinyStories, more time helped every vocabulary**, and helped the bigger ones most, since
  they had made the fewest steps. The home-made vocabulary still scores best, by less: 0.040 ahead
  of GPT-2's at 10 minutes, against 0.061 at 5. No model read even a tenth of the data, so every
  step was new text.
- **On Folktales, more time made every model worse** at text it had not seen; with the starter
  recipe, the more passes, the worse. Meanwhile the training loss, a different measure (per token,
  on the training text itself), kept falling: in the 10-minute run built from the data, from about
  2.5 at 5 minutes to 0.56 at the end. The models fitted their training text ever more closely
  while predicting new text worse: overfitting (see the
  [glossary](../TRAINING-DECISIONS.md#glossary)). Folktales is about 180 times smaller than
  TinyStories, so the same minutes mean far more passes.
- **The agent's recipe kept its lead, not its gain.** At 10 minutes it still beats the starter
  recipe on Folktales (1.6242 against 1.8838), but it scores worse than it did at 5 minutes
  (1.2637). At the same 14 passes it lost less than the starter recipe (28.5% against 36.6%).
  A recipe chosen by 5-minute runs answers a 5-minute question.
- **The writing does not show it plainly.** In one reading of the four final samples, the
  10-minute Folktales writing (built from the data) is not visibly worse than the 5-minute
  writing, and in places reads more like a folk tale. Copied phrases of eight or more words stay
  rare, zero to two per run, all stock phrases such as "there lived a King who had a daughter".
  Here the score and the reading disagree, which is a reason to keep both.
- **Two moments at 5 minutes.** The learning-rate schedule stretches with the time budget (it
  cools down over the last 45%), so at 5 minutes a 10-minute run is still at its full learning
  rate (training loss about 2.5, against 1.93 for the 5-minute run at its end), and its 5-minute
  writing is not the 5-minute run's. So the explorer's Compare sets two such runs side by side
  only before either run's end, and then at each run's end; Growth shows a 10-minute run's
  5-minute moment.

## The 12 October sessions

Run with `session.py` on the same GPU, each in its own bench folder (`ar-bench-<name>`), once
10-minute runs and 10-hour sessions were the default (`61c273e`): TinyStories from 09:55 to 19:35
on 12 October 2026, Folktales from 20:06 that evening to 05:48 on the 13th. Both started from the
starter kit's recipe (version `91ffb4c37015`, the `train.py` of `6ad8ddd`) with the dataset's own
tokenizer, and trained it three times before changing anything, to measure the noise. Each folder
is the session's record as `record.py` keeps it: `results.tsv`, a run file per row, `session.json`
(the settings, and hashes of the files the agent may not change) and `versions/`, each `train.py`
that trained, named by its version in `results.tsv`. The hosted demo's `deploy/sessions/` keeps
only each session's scoreboard and its start and best runs.

| Session | Experiments | Changes kept | Three starts differ by | Start → best | Change |
|---|---|---|---|---|---|
| TinyStories, 12 October | 44 | 12 | 0.0022 | 0.4767 → 0.4281 | 10.2% better |
| Folktales, 12 to 13 October | 43 | 15 | 0.0082 | 1.9158 → 1.2429 | 35.1% better |

The start is the mean of the three starting runs.

- **TinyStories got faster, not better tuned.** Thirty experiments on learning rates, the schedule
  and weight decay kept eight small changes, most marked within the noise (0.4767 to 0.4726). Then
  full multi-head attention (three key/value heads instead of one) nearly doubled the steps that fit
  in 10 minutes, from 1,258 to 2,271, and scored 0.4377; the agent's note puts it down to a faster
  attention kernel. Two more changes for speed (the built-in causal attention with full-length
  windows, and no logit soft-cap) reached 3,137 steps and 0.4285. The model grew from 18.9 M to
  26.3 M parameters on the way. A rerun of the best scored within 0.0001 of it.
- **Folktales: the learning rate again, much further.** As on 7 October, almost all the gain came
  from raising `MATRIX_LR`: ten keeps in a row took it from 0.05 to 3.0 and the score from 1.9158 to
  1.2955, 92% of the session's gain. (On 7 October, in 5-minute runs, it stopped at 0.20.) A longer
  cool-down, a final learning rate of zero, and Muon's and Adam's momentum took it to 1.2429, the
  last change within the noise; its confirming rerun scored 1.2485.
- **Each session measured its own noise**: 0.0022 on TinyStories and 0.0082 on Folktales, against
  the 0.003 that two 5-minute runs differ by. On TinyStories the three starts agree with the
  ten-minute study's run of the same `train.py` (0.4774). On Folktales they score about 0.03 worse
  than its 1.8838, with the same `prepare.py` apart from its default time budget; why is not
  tested here. Read a session's gains against its own start.

## Full attention: the starter recipe since 20 October

From 20 October the starter kit starts from full attention (`library/recipes/recipe-b.py`, version
`c9ba08e4f442`, called recipe B while it was measured). It is the earlier starter recipe with only the
12 October TinyStories session's three attention changes:
- each head has its own keys and values
- plain causal attention over the whole context in every layer, not windows
- no logit soft-cap

On this laptop it fits about 2.5 times the steps into the same minutes. The last four collections in
the table come from it; every other started from the earlier, windowed-attention recipe, `91ffb4c37015`. The collections below
were made 20 to 22 October 2026 on the same GPU, with Windows power throttling opted out (`prepare.py`)
and each start checked by `speed_gate.py`.

**The baselines** come from `bash library/make_baselines_b.sh train`, ten minutes each, after a
windowed-attention speed check (1,265 steps, against a floor of 1,190). Windowed attention at 10
minutes is the ten-minute study's.

| Dataset | Tokenizer | Model | Steps, windowed → full | Passes, full | Score, windowed → full | Change |
|---|---|---|---|---|---|---|
| TinyStories | built from the data | 26.3 M | 1,210 → 3,149 | 20% of one | 0.4774 → 0.4365 | 8.6% better |
| TinyStories | Phi-3 / Llama 2 | 72.1 M | 859 → 1,971 | 11% of one | 0.5143 → 0.4671 | 9.2% better |
| TinyStories | GPT-2 | 107.1 M | 692 → 1,470 | 9% of one | 0.5177 → 0.4742 | 8.4% better |
| Folktales | built from the data | 26.3 M | 1,237 → 3,141 | 35 | 1.8838 → 3.2740 | 73.8% worse |
| Folktales | Phi-3 / Llama 2 | 72.1 M | 858 → 1,972 | 20.5 | 1.5791 → 2.8606 | 81.1% worse |
| Folktales | GPT-2 | 107.1 M | 692 → 1,469 | 16.4 | 1.5478 → 2.5469 | 64.6% worse |

`MINUTES=5 PAIRS="tinystories:own folktales:own" bash library/make_baselines_b.sh train` made the
5-minute pair, for 5 against 10 minutes in one recipe:
- TinyStories went from 0.4651 to 0.4365, 6.2% better with the extra time.
- Folktales went from 2.4565 to 3.2740, 33.3% worse: 17.6 passes became 35.

The models are bigger than the windowed ones (26.3 M against 18.9 M with the home-made vocabulary)
because every head's keys and values are 384 numbers wide, and so are the value tables.

**The sessions** started from full attention with `session.py`, 10 hours of 10-minute runs, agent
`claude-fable-5`.

| Session | Experiments | Changes kept | Three starts differ by | Start → best | Change |
|---|---|---|---|---|---|
| TinyStories, 20 to 21 October | 49 | 12 | 0.0005 | 0.4366 → 0.4251 | 2.6% better |
| Folktales, 21 to 22 October | 46 | 16 | 0.0372 | 3.2870 → 1.2074 | 63.3% better |

The start is the mean of the three starting runs.

- **More training made TinyStories better and Folktales much worse.** TinyStories is so large that
  no model read more than a fifth of it, so every step was new text, and full attention's extra steps
  helped every vocabulary. Folktales is about 180 times smaller: the same speed meant 16 to 35 passes
  over the same tales, and every model overfitted. The more passes, the worse. The models fit their
  training text ever more closely and predict new text worse.
- **The Folktales agent cured the overfitting.** It added dropout, which the recipe did not have, and
  raised it a tenth at a time. Seven experiments took the score from 3.27 to 1.28. Then a deeper model
  without value tables, a learning rate cooled to zero, a running average of the weights, label
  smoothing and larger batches (910 steps: fewer passes) brought it to 1.2074, lower than either windowed Folktales
  session reached (1.2637 on 7 October, 1.2429 on 12 October).
- **The TinyStories agent had little left to find.** The speed it found on 12 October is now in the
  recipe, so its gains were small tuning:
  - lower learning rates (main weights 0.05 → 0.02, embeddings 1.0 → 0.25)
  - a cool-down to zero
  - a running average of the weights
  - Adam's β2 at 0.995

  Every change to the model's shape lost.
- **Overfitting also makes the score less stable.** Folktales' three starts differ by 0.037, about
  68 times TinyStories' 0.0005. Read each session's gains against its own noise.
- **The TinyStories session paused** from 03:20 to 07:27 when the agent's sign-in, not the long-lived
  token, expired. It was resumed with `--resume`. Each launch now prints which sign-in it uses.

Two tools stay for the next sets:
- `make_baselines_b.sh prep` readies every pair's data and vocabulary without training.
- `uv run library/speed_gate.py <floor> <run.json>...` tells a slowed machine from a worse recipe by
  the steps. The floors at 600 s, TinyStories with its own vocabulary, are 1,190 for windowed attention
  and 2,980 for full attention.

## The explorer's data

`uv run library/report.py` writes:

- `index.json`: every collection and its files (static hosting cannot list a folder), with a
  note naming a run where the files alone cannot (`run_notes` in `collection.json`: a study's
  runs share one commit).
- `tokens.json`: each tokenizer's split of the four prompts and four contrast sentences, characters per token and the
  share of the vocabulary the dataset never uses, over up to 100,000 training documents; and each
  dataset's estimated size in tokens (its training text, plus a start and an end marker per
  document), from which the explorer counts passes.
- `tokenizers/`: each home-made vocabulary as a Hugging Face `tokenizer.json`, written only after
  it splits 20,000 training documents (all 8,895 of Folktales') and ten edge cases exactly as
  training does. The end marker is named `<|endoftext|>`, the name the completion explorables
  recognise as the end of a text; the start marker is `<|startoftext|>`, by CLIP's convention.
  The ids are unchanged.
- `copies.json`: stretches of eight or more words in the Folktales samples that repeat the
  training text word for word. TinyStories' training text is too large to index this way.

It only reads the local cache, so it is safe to run while a session trains.

## Weights

The models themselves stay in `library/checkpoints/`, which git ignores: 63 MB with a home-made
vocabulary, 136 MB with Phi-3's and 192 MB with GPT-2's. The study's are `study-*.pt`.

## Licences

- TinyStories (`karpathy/tinystories-gpt4-clean`): CDLA-Sharing-1.0.
- Folktales (`merve/folk-mythology-tales`): the card says CC0 1.0. Its source, D. L.
  Ashliman's Folktexts, carries its own copyright notice.
- Phi-3 tokenizer (`microsoft/Phi-3-mini-4k-instruct`): MIT. GPT-2 tokenizer: MIT.

The library publishes only what the models wrote and their scores, never the dataset text.
