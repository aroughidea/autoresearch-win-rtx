# The library

Small models trained ahead of time, so a class can explore what a training decision does
without a GPU. The [Training Decisions explorer](https://llmexplorables.aroughidea.com/training/)
reads this folder; `TRAINING-DECISIONS.md` at the repo root says what each decision is.

Each collection is a folder shaped like a learner's own repo: `runs/` (one run file per
training run, written by `capture.py`) and, when an agent made it, `results.tsv`.

| Collection | What it holds |
|---|---|
| `baselines/` | Six runs: TinyStories and Folktales, each with three tokenizers. The starter kit's recipe, five minutes each, no agent. Only the dataset and the tokenizer change. |
| `sessions/tinystories-may2026/` | The agent's 16 experiments (`WALKTHROUGH.md`): 15 on 23 May 2026 and one on 6 August. Made before run files existed, so it has scores but no writing. |
| `sessions/folktales-oct2026/` | The agent's 12 experiments on Folktales (own tokenizer) on the evening of 7 October 2026, from the starter kit's recipe, with a run file for each. |

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

## The explorer's data

`uv run library/report.py` writes:

- `index.json`: every collection and its files (static hosting cannot list a folder).
- `tokens.json`: each tokenizer's split of the four prompts and four contrast sentences, characters per token and the
  share of the vocabulary the dataset never uses, over up to 100,000 training documents.
- `copies.json`: stretches of eight or more words in the Folktales samples that repeat the
  training text word for word. TinyStories' training text is too large to index this way.

It only reads the local cache, so it is safe to run while a session trains.

## Weights

The models themselves stay in `library/checkpoints/`, which git ignores: the Phi-3 and GPT-2
models are 150 to 230 MB each.

## Licences

- TinyStories (`karpathy/tinystories-gpt4-clean`): CDLA-Sharing-1.0.
- Folktales (`merve/folk-mythology-tales`): the card says CC0 1.0. Its source, D. L.
  Ashliman's Folktexts, carries its own copyright notice.
- Phi-3 tokenizer (`microsoft/Phi-3-mini-4k-instruct`): MIT. GPT-2 tokenizer: MIT.

The library publishes only what the models wrote and their scores, never the dataset text.
