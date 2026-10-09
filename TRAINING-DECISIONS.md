# Training Decisions

What this project demonstrates, in plain language. Start here if you came from a workshop or from the [live demo](https://autoresearch-demo.fly.dev/).

## What you'll understand

**Changing a design decision changes the final experience in a specific way.** The goal is to understand that, not to find the best recipe. After the demo and the journey, you can:

1. **Follow how a model gets made:** get and understand a dataset, set the tokens, build a model, use it, and iterate on the build at the algorithmic level.
2. **Speak to how a dataset's qualities show up in the experience,** having trained on one dataset and later on another.
3. **Speak to how the tokenizer shapes the model:** its size, how far it trains in a fixed time, and how it writes.
4. **Experience consequences and compare decision sets:** watch a model grow over training time, and set two recipes side by side.
5. **Compare models through use,** in the completion demo (you type the start, the model continues), alongside the score.
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

Everything above that is experiments: change something, train a new model with the change, score it, keep the change or discard it. Machine learning calls this an outer loop around training: the inner loop trains a model's weights, the outer loop chooses the settings that training uses [16]. Nothing a model learned carries into the next experiment; every run starts from random weights. Only the recipe carries forward.

| What changes | Who changes it | How it is judged | Training? | Clock |
|---|---|---|---|---|
| The model's weights | Gradient descent, inside `train.py` | Training loss, at every step | **Yes** | Training time: 0 s to 5 min |
| The training recipe in `train.py` | The coding agent | The score at the end of each run | No: experiments, an outer loop | Research time: experiments over hours |
| The agent's instructions in `program.md` | You, in this project | Whether a session's research went well; there is no agreed score | No: a loop around that loop | Sessions over nights |

Karpathy's README for autoresearch states the division of labour directly: `train.py` is edited by the agent, `program.md` by the human. The second row is close to what machine learning calls hyperparameter optimization and neural architecture search, which a standard survey describes by a search space, a search strategy and a way to estimate performance ([Elsken, Metzen and Hutter, 2019](https://jmlr.org/papers/v20/18-598.html)): here the edits to `train.py`, the agent, and five minutes of training followed by the score.

In principle a second agent could take the third row from you, editing `program.md` and keeping the instructions that produce better sessions. Research calls this prompt optimization; [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409) (Yang and others, 2023) is one example. It would still be experiments in an outer loop, not training. Each row up has a slower clock, fewer results to learn from and a vaguer score, which is why the third row is a person's job here.

## What you can see change

Measured on a laptop RTX 4000 Ada in October 2026, same prompts and the same sampling settings every time:

| Decision | What changes | Where to see it |
|---|---|---|
| **Training time**, 0 s to 5 min (one model learning) | Random word fragments, then word salad, then grammar, then stories | Each experiment's file in `runs/` |
| **Training for longer**, 10 min instead of 5 (a person's study, not the agent's) | On TinyStories every score improves, by 8–11%. On Folktales every score gets 23–37% worse: 8 to 14 passes over a small dataset, and the models overfit. Their writing does not read plainly worse. | [`library/README.md`](library/README.md#the-ten-minute-study) |
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

## Glossary

The pages use everyday words where they are already common in the field. Each is listed here with the professional term and a source, so a reader can follow it into the literature. Numbers in brackets point to the sources below.

| Word used here | Professional term | What it means here |
|---|---|---|
| **Dataset** | Training corpus | The text a model learns from: TinyStories [1] or Folktales. |
| **Token, tokenizer** | Subword tokenization; byte-pair encoding (BPE) [2] | A model reads and writes in tokens, words and pieces of words, produced by a tokenizer. |
| **Vocabulary** | Tokenizer vocabulary; SentencePiece for Phi-3 / Llama 2 [4] | The fixed set of tokens a model can use: 8,192 built from the dataset, 32,011 for Phi-3 (Llama 2's 32,000 plus 11 added tokens), 50,257 for GPT-2. |
| **Part of a letter** | Byte-level BPE [3] | Tokenizers that work in bytes, like GPT-2's and the home-made ones, can split a letter such as "é" across two tokens. |
| **Training** | Training by gradient descent [5] | Changing a model's weights by having it read data. Here it happens only inside each run: 5 minutes, or 10 in the ten-minute study. |
| **Training time** | A fixed training-time budget [6] | The 5 minutes each run trains for, whatever the GPU. Choosing a recipe by short runs can favour what pays off early, a known bias [18]; the [ten-minute study](library/README.md#the-ten-minute-study) trains the same recipes for 10 minutes to check. |
| **Step** | Optimizer step, one per minibatch [5] | One update of the model's weights, from one batch of training text. A 5-minute run here makes 344 to 639 steps: a bigger vocabulary makes a bigger model, so fewer steps fit. |
| **Batch size** | Minibatch size [5] | The text read for each step: 32,768 tokens here (`TOTAL_BATCH_SIZE` in `train.py`). |
| **Epoch, pass** | Epoch [5] | One full pass through the training data. In 5 minutes a model reads 2 to 4% of TinyStories, but goes through Folktales 4 to 7 times, depending on its tokenizer. With a fixed amount of computing, a bigger model reads fewer tokens; the best balance between the two is studied as compute-optimal training [17]. |
| **Overfitting** | Overfitting [5] | Getting better at the training text while getting worse at text the model has not seen. In the ten-minute study every Folktales model made 8 to 14 passes, and every score got worse; on TinyStories, which no model got a tenth of the way through, every score improved. |
| **Training recipe** | Training recipe or training procedure [7][8] | Everything in `train.py` that decides how data becomes a model: the model's shape (its architecture), the optimizer, learning rates, batch size and schedule (its hyperparameters). |
| **Experiments, research time** | Hyperparameter optimization, also called hyperparameter search [9], and neural architecture search [10]: an outer loop around training [16] | Improving the recipe by trying a change, training a new model with it, and keeping the change if the score improves. The agent experiments with the recipe; it does not train itself. We avoid "search" on its own (it reads as web search), "tuning", which readers may confuse with fine-tuning (more training), and "optimization" on its own, which also names what the training optimizer does. |
| **Score** | Validation bits per byte (BPB) [11] | How surprised the model is by text it has not seen, per byte of text; lower is better. Counting bytes rather than tokens lets tokenizers be compared [6]. |
| **Noise** | Run-to-run variance | Two runs of the same code differ by about 0.003, measured on this laptop. |
| **Continues text** | A pretrained (base) model, not instruction-tuned [12] | These models continue whatever you type. Chat assistants need further training to follow instructions. |
| **Sampling settings** | Temperature and top-k sampling [13] | How the next token is picked from the model's prediction. Every snapshot and every run uses the same settings. |
| **Copied phrase** | Verbatim memorization [14] | A run of eight or more words a model repeats word for word from its training text. |
| **Agent's instructions** | The agent's prompt; improving it is prompt optimization [15] | `program.md`, which a person edits here. Changing it and keeping what works is another outer loop, not training. |
| **Snapshot** | Samples at a training checkpoint | What a model wrote at one moment of training, saved in `runs/`. A *checkpoint* is a model's saved weights. |

### Sources

1. Ronen Eldan and Yuanzhi Li. [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) 2023.
2. Rico Sennrich, Barry Haddow and Alexandra Birch. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). ACL 2016.
3. Alec Radford and others. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2). OpenAI, 2019.
4. Taku Kudo and John Richardson. [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226). EMNLP 2018.
5. Ian Goodfellow, Yoshua Bengio and Aaron Courville. [Deep Learning](https://www.deeplearningbook.org/), chapter 5, "Machine Learning Basics" (capacity, overfitting and underfitting), and chapter 8, "Optimization for Training Deep Models" (minibatches, epochs). MIT Press, 2016.
6. Andrej Karpathy. [autoresearch](https://github.com/karpathy/autoresearch), README (fixed 5-minute budget; `val_bpb` is independent of vocabulary size). 2026.
7. Vasilis Vryniotis. [How to Train State-Of-The-Art Models Using TorchVision's Latest Primitives](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/). PyTorch blog, 2021.
8. Ross Wightman, Hugo Touvron and Hervé Jégou. [ResNet strikes back: An improved training procedure in timm](https://arxiv.org/abs/2110.00476). 2021.
9. Matthias Feurer and Frank Hutter. "Hyperparameter Optimization", chapter 1 of [Automated Machine Learning: Methods, Systems, Challenges](https://www.automl.org/book/). Springer, 2019.
10. Thomas Elsken, Jan Hendrik Metzen and Frank Hutter. [Neural Architecture Search: A Survey](https://jmlr.org/papers/v20/18-598.html). JMLR 20(55), 2019.
11. Leo Gao and others. [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027). 2020 (evaluates in bits per byte).
12. Long Ouyang and others. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). NeurIPS 2022.
13. Angela Fan, Mike Lewis and Yann Dauphin. [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) (top-k sampling). ACL 2018.
14. Nicholas Carlini and others. [Quantifying Memorization Across Neural Language Models](https://arxiv.org/abs/2202.07646). ICLR 2023.
15. Chengrun Yang and others. [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409). ICLR 2024.
16. Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi and Massimiliano Pontil. [Bilevel Programming for Hyperparameter Optimization and Meta-Learning](https://proceedings.mlr.press/v80/franceschi18a.html). ICML 2018 (training as the inner problem, hyperparameters as the outer one).
17. Jordan Hoffmann and others. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). NeurIPS 2022.
18. Yuhuai Wu, Mengye Ren, Renjie Liao and Roger Grosse. [Understanding Short-Horizon Bias in Stochastic Meta-Optimization](https://arxiv.org/abs/1803.02021). ICLR 2018 (choosing hyperparameters by short runs is biased toward small learning rates).

"Agent" has two meanings in the workshops: there it is the assistant you design; here it is the coding agent that runs the experiments.
