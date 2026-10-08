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
3. **The model and its training recipe.** A small GPT, about 19 million parameters with the default vocabulary, trained for exactly 5 minutes on one GPU. The *training recipe* is everything in [`train.py`](train.py) that decides how the data becomes a model: the model's shape (layers, width, attention), the optimizer and its learning rates, the batch size, and the learning-rate schedule. "Training recipe" is the usual term in machine learning; PyTorch's [How to Train State-Of-The-Art Models Using TorchVision's Latest Primitives](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/) (2021) is a well-known example.
4. **Use.** Give it the start of a story and it continues. These models are only *pretrained* (step 1 of the three in Workshop 2's "How a model gets made"), so they continue text. They are not chat assistants.
5. **Iteration by an AI agent.** A coding agent edits the recipe, trains a new model with it, reads the score, keeps or discards the change, and repeats until its session budget runs out (8 hours unless you set another). You don't need to know the recipe's details: the agent changes them, following [`program.md`](program.md).

## Training and improvement are not the same

Training happens in one place: inside each 5-minute run of `train.py` on the GPU, where the model's weights change as it reads the dataset. That is the only training in this project.

Everything above that is improvement by search: change something, train a new model with the change, score it, keep the change or discard it. Nothing a model learned carries into the next experiment; every run starts from random weights. Only the recipe carries forward.

| What changes | Who changes it | How it is judged | Training? | Clock |
|---|---|---|---|---|
| The model's weights | Gradient descent, inside `train.py` | Training loss, at every step | **Yes** | Training time: 0 s to 5 min |
| The training recipe in `train.py` | The coding agent | The score at the end of each run | No: search | Research time: experiments over hours |
| The agent's instructions in `program.md` | You, in this project | Whether a session's research went well; there is no agreed score | No: search | Sessions over nights |

Karpathy's README for autoresearch states the division of labour directly: `train.py` is edited by the agent, `program.md` by the human. The second row is close to what machine learning calls hyperparameter optimization and neural architecture search, which a standard survey describes by a search space, a search strategy and a way to estimate performance ([Elsken, Metzen and Hutter, 2019](https://jmlr.org/papers/v20/18-598.html)): here the edits to `train.py`, the agent, and five minutes of training followed by the score.

In principle a second agent could take the third row from you, editing `program.md` and keeping the instructions that produce better sessions. Research calls this prompt optimization; [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409) (Yang and others, 2023) is one example. It would still be search, not training. Each row up has a slower clock, fewer results to learn from and a vaguer score, which is why the third row is a person's job here.

## What you can see change

Measured on a laptop RTX 4000 Ada in October 2026, same prompts and the same sampling settings every time:

| Decision | What changes | Where to see it |
|---|---|---|
| **Training time**, 0 s to 5 min (one model learning) | Random word fragments, then word salad, then grammar, then stories | Each experiment's file in `runs/` |
| **Dataset** | From the same prompts: Lily and Mia at the park, or kings, viziers and "legs like masts" | Train on each, then compare in `chat.py` |
| **Tokenizer** | Model size 18.9 M → 46.3 M → 67.3 M parameters; fewer training steps fit in 5 minutes; the score is 10–12% worse. An untrained model's random output shows what its vocabulary is made of. | `runs/` and the score |
| **The agent's recipe changes** (research time: a new model each experiment) | Often too small to read. The May session in [`WALKTHROUGH.md`](WALKTHROUGH.md) improved the score by 0.27%, less than two runs of the same code differ. Sometimes large in the score: the October Folktales session improved it by 8.8% (about 41 times the noise), mostly through one learning rate, yet in one careful reading its first and best models still read much alike. | The [live demo](https://autoresearch-demo.fly.dev/) for May; [`library/sessions/`](library/) for both |

That last row is the reason the agent needs a score: much of what it finds is too small for a person to notice, and even a large gain in the score can be hard to see in the writing.

## The score

The score is `val_bpb` (validation bits per byte): how surprised the model is by stories it has never seen. **Lower is better.**

- It compares models trained on the **same dataset**, never across datasets. Across datasets, read the writing instead.
- Two runs of the same code differ by about **0.003** on TinyStories, so treat smaller differences as ties. Folktales may vary more: two runs of one recipe there differed by 0.007. That has not been measured properly.

## Where training happens, and the record

- **Your own NVIDIA GPU** on Windows, the main path. A Mac uses a sibling fork.
- **A GPU rented by the hour**, if you have no suitable GPU: about $5–10 a night (see the starter kit's `HARDWARE.md`).
- **A session's length:** the agent works to a budget (8 hours unless you set another), then finishes its last experiment, logs it and stops, so every session ends on a clean record.
- **The record:** every experiment is a git commit; `results.tsv` is the scoreboard; `runs/` holds what each model wrote as it trained. Pushed to GitHub, the whole night is shareable and checkable.

## Three ways in

- **Try it,** with nothing to install: the [live demo](https://autoresearch-demo.fly.dev/) (the first load takes about 12 seconds while it wakes).
- **Read it:** [`WALKTHROUGH.md`](WALKTHROUGH.md), one real afternoon of the agent's research, every experiment explained. It is the most technical document here.
- **Run it:** the [starter kit](https://github.com/aroughidea/autoresearch-starter). You need an NVIDIA GPU or a rented one, and an account for a coding agent (Claude Code or Codex).

## Words used here

- **Token:** a word or piece of a word; models read and write in tokens.
- **Vocabulary:** the fixed set of tokens a model can use.
- **Training:** changing a model's weights by having it read data. Here it happens only inside each 5-minute run.
- **Training recipe:** everything in `train.py` that decides how data becomes a model: the model's shape, the optimizer, learning rates, batch size and schedule.
- **Search:** improving something by trying changes and keeping those that score better. The agent searches over recipes; it does not train itself.
- **Score:** `val_bpb`, above.
- **Snapshot:** what a model wrote at one moment of training, saved in `runs/`. (The repo also saves *checkpoints*: a model's learned weights.)
- **Agent:** the coding agent that runs the experiments. In the workshops, "agent" also means the assistant you design; here it is the researcher.
