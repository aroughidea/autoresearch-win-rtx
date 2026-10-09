# The experiment record: versions of the recipe, run entries and a pointer, in place of git

Status: approved 2026-10-08 (a folder of files is the record) · Thomas J McLeish with Claude
Diagram: the "Proposed" view of [the session map](https://claude.ai/artifact/A6V2sQNYRYCHhhaQudtKUo)

## Decision

An autoresearch session keeps its own record: every version of `train.py` that trained, one entry
per run, and a pointer to the best version so far. Git leaves the experiment loop. It keeps one job:
holding the lab's own software (`prepare.py`, `capture.py`, `program.md` and the rest) on GitHub.

Keeping a change moves the pointer. Undoing a change resets `train.py` from the version the pointer
marks. `train.py` on disk is only the working copy.

## Why

- **A complete record.** Today an undone experiment loses its code: `git reset` drops it, and only
  its scoreboard row and run file remain. In the record, every version stays, kept or undone.
- **The record shows what actually trained.** `capture.py` stores the code at the moment training
  starts. With git, the agent commits before training, and an edit made after that commit would go
  unrecorded.
- **One history, readable without git.** A participant can open the session's folder and read each
  version of the recipe as an ordinary file. Model weights never enter the record.
- **It matches professional practice.** This is *experiment tracking*, and the pointer is what
  MLflow's model registry calls an alias: "a mutable, named reference to a particular version",
  its example being `champion` ([MLflow, Model Registry](https://mlflow.org/docs/latest/ml/model-registry)).

## What stays the same

Training, scoring, `prepare.py`, the writing snapshots, the lab's rules in `program.md` (session
budget, the noise rule, one experiment at a time, ending on time), the agent's freedom to change
anything in `train.py`, and the explorer's views.

## The record: a folder of files

Each session has its own folder, `sessions/<name>/`:

```
sessions/tinystories-10min-2026-10-09/
  session.json        dataset, tokenizer, run length, budget, start and end times, fixed-file hashes
  versions/<id>.py    every version of train.py that trained, named by its id
  runs/<run>.json     one entry per run: today's run file plus the fields below
  best.json           the pointer: the best version's id, its run and its score
  results.tsv         the scoreboard, written by the tool after every keep or undo; never edited by hand
```

- **Version id:** the first 12 characters of the SHA-256 hash of `train.py`'s text, with line endings
  made uniform. Identical code gets the same id, so a confirmation rerun shares its version.
- **New fields in a run entry:** `version`, `parent` (the version the pointer marked when this
  run's change began), `status` (`keep`, `discard` or `crash`; `lab.py undo` records `discard`, the scoreboard's existing word) and `description` (the agent's note).
  Kept runs also record their checkpoint's path and hash.
- **Lineage:** following `parent` from any version leads back to the first version; the kept versions in
  order are the story of how the recipe improved.
- **Compatibility:** run files keep their `<time>_<id>` names and keep a `commit` field holding the
  version id, and `results.tsv` keeps its columns with the version id in the `commit` column. So
  `chat.py`, the explorer and `library/report.py`, which join on that column, work unchanged. They
  can move to `version` later.
- **A SQLite file is optional,** built from the folder by `lab.py export --sqlite` when someone
  wants to search across sessions. The folder stays the source.

The folder is the record. It holds only text; weights stay in `checkpoints/` on the laptop. It
can be pushed to GitHub to share it, but git is not needed to run or read it.

## The tool: `lab.py`

A small program the agent may not edit, like `capture.py`. The agent calls it instead of git.

| Command | What it does |
|---|---|
| `uv run lab.py status` | The best version and its score, whether `train.py` differs from it, and the time left in the session |
| `uv run lab.py history` | One line per run: version, parent, score, keep or undo, description. The agent's research memory. |
| `uv run lab.py diff [A] [B]` | What changed between two versions; by default, between the best version and `train.py` |
| `uv run lab.py show <id>` | Print one version |
| `uv run lab.py keep "<description>"` | The last finished run's version becomes the best: the pointer moves, the model is copied to `checkpoints/`, the entry is marked `keep` |
| `uv run lab.py undo "<description>"` | The entry is marked `discard` (or `crash` if the run did not finish), and `train.py` is reset from the best version |

`lab.py start <name>` (the runner calls it) creates the session folder, stores the starter recipe as
the first version, points `best.json` at it with no score, and records the fixed files' hashes.

The decision to keep or undo stays with the agent, under `program.md`'s rules: the noise rule and
the simplicity criterion are judgments, not a score comparison the tool could make. The tool refuses
only clear mistakes: keeping or undoing when no run has finished since the last decision, and keeping
a run that crashed.

## What `capture.py` adds

At the start of every training run it reads `train.py`, stores it in `versions/` if it is new, and
opens the run's entry with its version and parent. It then checks the fixed files against the hashes
in `session.json` and refuses to train if any has changed. At the end of the run it adds the score,
as today.

The fixed files are `prepare.py`, `capture.py`, `lab.py`, `program.md`, `pyproject.toml` and
`uv.lock`. The recipe is `train.py` alone, as in Karpathy's rules: a helper file the agent might
create is outside the rules and is not part of any version.

## The loop in `program.md`

1. `uv run lab.py status`: `train.py` should match the best version.
2. Read `uv run lab.py history` (and `diff` if useful), then change one thing in `train.py`.
3. `uv run train.py > run.log 2>&1`. `capture.py` records the version and the run.
4. Read the score from `run.log`. If the run crashed, fix it or `uv run lab.py undo "<what was tried>"`.
5. Better under the noise rule: `uv run lab.py keep "<description>"`. Otherwise `uv run lab.py undo "<description>"`.

No git commands. The session's budget, the 20-minute stop and the summary at the end are unchanged.

## Requirements carried over from git

Git did eight things without being asked. The record has to do each one on purpose.

| Git gave | The record's answer |
|---|---|
| A snapshot of the whole folder, so a reset also repaired accidental edits | `capture.py` checks the fixed files' hashes before every run and refuses to train if one changed |
| The agent's research memory (`git log`, `git diff`) | `lab.py history` and `lab.py diff`; `program.md` tells the agent to read the history before each change |
| Lineage: the branch was the chain of kept improvements | Each run records its `parent` |
| A fingerprint for every version | Version ids are content hashes |
| Never half-written | Each write goes to a temporary file that is then renamed; `best.json` changes last |
| A habit agents already have | A handful of plainly named commands, and a rehearsal session before any overnight use |
| Branching, for several agents at once | Out of scope: one agent per laptop. Several agents would need a pointer each. |
| No code to maintain, and Karpathy's loop unchanged | A small tool with tests, and a loop that no longer matches Karpathy's; the docs say so and why |

## Where git remains

- The lab's software, on GitHub, as today.
- Optionally, publishing a session: push its folder (text only) to share it. The explorer's "Your
  runs" already loads a folder from the laptop, as well as from GitHub.
- Session branches and worktrees are no longer needed: the runner makes a session folder instead.
- `checkpoint_pre_eval.pt` leaves git (it is ignored), which also settles the open question of
  7 October.

## Effects elsewhere

- **The runner** (`run-session.sh`) calls `lab.py start` instead of making a branch, and its
  session rules drop the git instructions.
- **The explorer** reads the new fields when present. Showing code diffs between versions is a
  later addition.
- **The library** is unchanged; past sessions stay as they are.
- **Learning objective 7** reads, today, "your own hardware, a rented GPU, and GitHub as the lab
  record". Proposed: *your own hardware or a rented GPU, and each session's record (every version of
  the recipe and every run) in its own folder, which you can publish on GitHub.*
- **Karpathy's original** uses git for the loop. The worked example and the starter kit would say
  plainly that this fork keeps a complete record instead, and why.

## Order of work

1. The worked example: `lab.py`, the `capture.py` additions and the `program.md` loop, with tests.
2. A rehearsal: a two-hour daytime session with 5-minute runs.
3. The runner, then an overnight session.
4. The starter kit, ported once the rehearsal and one overnight session have passed.
5. The explorer's reading of the new fields.

## Testing and acceptance

**Unit tests:**
- version ids: the same code with different line endings gets one id;
- `keep` moves the pointer and copies the checkpoint;
- `undo` restores `train.py` exactly;
- the tool refuses a second decision on the same run, and refuses to keep a crash;
- a write that fails leaves the old `best.json` intact;
- a changed fixed file stops training;
- `results.tsv` matches the entries;
- `capture.py` stores the version at the start of training;
- a confirmation rerun shares the version id.

**The rehearsal passes when:**
- every run has an entry with its code;
- the pointer always names the best kept version;
- after every undo, `train.py` equals the best version;
- `results.tsv` matches the entries;
- the agent used no git command and never touched a fixed file.

## Out of scope

A database server, several agents at once, a code-diff view in the explorer, moving past sessions
into the new format, and a store for model weights beyond `checkpoints/`.

## Decisions

1. **A folder of files is the record** (TJ, 2026-10-08). SQLite stays optional, built from the folder.
2. **Learning objective 7** takes the wording above when the docs change; TJ edits it in that review.
3. **Session folders stay on the laptop** unless someone shares them; publishing is a choice, not a step.
4. **The worked example first,** then the rehearsal, then the starter kit.
