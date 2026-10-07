# Training Decisions

What this project demonstrates, in plain language. Start here if you came from a workshop or from the [live demo](https://autoresearch-demo.fly.dev/).

## What you'll understand

**Changing a design decision changes the final experience in a specific way.** The goal is to understand that, not to find the best recipe. After the demo and the journey, you can:

1. **Follow how a model gets made:** get and understand a dataset, set the tokens, build a model, use it, and iterate on the build at the algorithmic level.
2. **Speak to how a dataset's qualities show up in the experience,** having trained on one dataset and later on another.
3. **Speak to how the tokenizer shapes the model:** its size, how far it trains in a fixed time, and how it writes.
4. **Experience consequences and compare decision sets:** watch a model grow over training time, and set two recipes side by side.
5. **Compare models through use,** in the chat page, alongside the score.
6. **Understand AI-driven research:** an agent changes the algorithm, so no deep expertise is needed, and keeps or discards each change by a score.
7. **Know where training happens and how it is recorded:** your own hardware, a rented GPU, and GitHub as the lab record.

## How a model gets made here

Each step is a decision.

1. **The dataset** is the text the model learns from. Two are built in: **TinyStories**, short children's stories written by GPT-4, and **Folktales**, public-domain folk and fairy tales. A model can only write like what it read.
2. **The tokens.** A model reads and writes in tokens: words and pieces of words from a fixed vocabulary. Three vocabularies are built in: one built from the dataset itself (8,192 tokens), Phi-3's (32,011, the vocabulary Llama 2 models use), and GPT-2's (50,257).
3. **The model and its recipe.** A small GPT, about 19 million parameters with the default vocabulary, trained for exactly 5 minutes on one GPU. The recipe is the settings in [`train.py`](train.py): learning rates, the model's shape, how the learning winds down at the end.
4. **Use.** Give it the start of a story and it continues. These models are only *pretrained* (step 1 of the three in Workshop 2's "How a model gets made"), so they continue text. They are not chat assistants.
5. **Iteration by an AI agent.** A coding agent edits the recipe, trains, reads the score, keeps or discards the change, and repeats all night. You don't need to know the algorithm: the agent changes it, following [`program.md`](program.md).

## What you can see change

Measured on a laptop RTX 4000 Ada in October 2026, same prompts and the same sampling settings every time:

| Decision | What changes | Where to see it |
|---|---|---|
| **Training time**, 0 s to 5 min | Random word fragments, then word salad, then grammar, then stories | Each experiment's file in `runs/` |
| **Dataset** | From the same prompts: Lily and Mia at the park, or kings, viziers and "legs like masts" | Train on each, then compare in `chat.py` |
| **Tokenizer** | Model size 18.9 M → 46.3 M → 67.3 M parameters; fewer training steps fit in 5 minutes; the score is 10–12% worse. An untrained model's random output shows what its vocabulary is made of. | `runs/` and the score |
| **The agent's recipe changes** | Usually too small to read. The session in [`WALKTHROUGH.md`](WALKTHROUGH.md) improved the score by 0.27%, less than two runs of the same code differ. | The [live demo](https://autoresearch-demo.fly.dev/): its two models read almost alike |

That last row is the reason the agent needs a score: most of what it finds is too small for a person to notice.

## The score

The score is `val_bpb` (validation bits per byte): how surprised the model is by stories it has never seen. **Lower is better.**

- It compares models trained on the **same dataset**, never across datasets. Across datasets, read the writing instead.
- Two runs of the same code differ by about **0.003**, so treat smaller differences as ties.

## Where training happens, and the record

- **Your own NVIDIA GPU** on Windows, the main path. A Mac uses a sibling fork.
- **A GPU rented by the hour**, if you have no suitable GPU: about $5–10 a night (see the starter kit's `HARDWARE.md`).
- **The record:** every experiment is a git commit; `results.tsv` is the scoreboard; `runs/` holds what each model wrote as it trained. Pushed to GitHub, the whole night is shareable and checkable.

## Three ways in

- **Try it,** with nothing to install: the [live demo](https://autoresearch-demo.fly.dev/) (the first load takes about 12 seconds while it wakes).
- **Read it:** [`WALKTHROUGH.md`](WALKTHROUGH.md), one real afternoon of the agent's research, every experiment explained. It is the most technical document here.
- **Run it:** the [starter kit](https://github.com/aroughidea/autoresearch-starter). You need an NVIDIA GPU or a rented one, and an account for a coding agent (Claude Code or Codex).

## Words used here

- **Token:** a word or piece of a word; models read and write in tokens.
- **Vocabulary:** the fixed set of tokens a model can use.
- **Recipe:** the training settings in `train.py`.
- **Score:** `val_bpb`, above.
- **Snapshot:** what a model wrote at one moment of training, saved in `runs/`. (The repo also saves *checkpoints*: a model's learned weights.)
- **Agent:** the coding agent that runs the experiments. In the workshops, "agent" also means the assistant you design; here it is the researcher.
