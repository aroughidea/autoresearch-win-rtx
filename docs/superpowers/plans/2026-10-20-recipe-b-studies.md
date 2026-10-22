# Recipe B Studies: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remake the library's long training studies with recipe B as the starting recipe, on a laptop held at one speed: six baselines at 10 minutes, then a 10-hour agent session on each dataset, all brought into the library.

**Architecture:** The work comes in two parts.

- **Part 1, prep.** Done ahead, with no training:
  - every training run opts out of Windows power throttling, from `prepare.py` (a fixed file every `train.py` imports, so the agent cannot edit it away)
  - recipe B is committed as a file, pinned by a test to the version measured on 13 October
  - a speed gate turns a slow machine into a failing check
  - the starter kit switches to recipe B
  - a baseline driver keeps its data preparation apart from its training
  - the six pairs' data and vocabularies are in the cache
  - the two session folders are built and checked
- **Part 2, launch.** TJ starts each training run whenever it suits, with one command each. The baselines and the two sessions are independent of each other and can run in any order.

**Tech Stack:** Python 3 + pytest (`uv run python -m pytest`; the bare `pytest.exe` is blocked by Smart App Control in the starter), bash drivers in `library/`, `session.py`, `lab.py`, `record.py`, `uv run library/report.py`.

**Spec:**
- TJ, 20 October:
  - "it looks like recipe b is a good way to start"
  - "we can set up and rerun the long form studies, i want a plan first"
  - then: "opting out of power throttle seems fine"
  - "im ok with the folktales overfit challenge, that is a learning to be illustrated not hidden"
  - "can we separate prep from training runs and get all the prep ready? we will then launch training when best fit"
- `docs/TODO.md`, "The starter recipe's attention setting".
- The bench `ar-bench-attention-b-2026-10-13`: `make_recipe_b.py`, `measure.log`, `check-*.log`, `gpu.csv`.
- `library/README.md`.

## What the record already says

- **Recipe B** is the starter recipe (`train.py` at `6ad8ddd`, version `91ffb4c37015`) with only the TinyStories night's three attention changes:
  - full key and value heads, `N_KV_HEAD = 3` (`d967b0b7055d`)
  - plain causal attention with all-long windows, `WINDOW_PATTERN = "L"` (`14c8214ecc32`)
  - no logit soft-cap (`8fa477d3ed43`)

  The bench's `make_recipe_b.py` writes it; its version is `c9ba08e4f442`. It has 26.3 M parameters, against 18.9 M for the starter.
- **Steps at full speed**, TinyStories with its own vocabulary, 10 minutes, this laptop:
  - recipe A: 1,214, 1,258 and 1,258 (12 October)
  - recipe B: about 3,100 (192 ms a step on 13 October); the session's equivalent version made 3,137.
- **Scores at full speed:** none yet for recipe B. The 13 October measurement (B 0.4704 to 0.4713 at about 1,400 steps; A 0.5083 at 761) ran slowed and does not compare.
- **The slowdown has two candidates, and both are now covered.**
  - TJ's reading: the disk was full. C: now has 324 GB free and F: 1 TB, and the page file (120 GB) is on F:.
  - On 13 October, between 16:47 and 16:58, the same recipe A ran at 790 ms a step launched normally, at 479 ms (the 12 October speed) opted out of Windows power throttling (the bench's `unthrottled.py`), and at 810 ms launched normally again.
  - The GPU held about 2.3 GHz throughout (`gpu.csv`).
  - Both causes are now handled: the disk has room, and runs opt out (Task 1). The speed gate (Task 3) catches anything else.
- **Folktales overfits at 10 minutes.** With the starter recipe it scores 1.3788 at 5 minutes and 1.8838 at 10, after 14 passes. Recipe B fits about 2.5 times the steps, so about 35 passes in 10 minutes, and may score worse still. This is to be shown, not avoided (TJ, 20 October).
- **The machine today:** 64 GB of RAM. Docker Desktop is running (the vLLM model proxy is started by hand). No long-lived agent token is set (`CLAUDE_CODE_OAUTH_TOKEN`).

## Decisions (settled 2026-10-20)

1. **The slowdown is fixed in code.** `prepare.py` opts every training process out of power throttling, in both repos.
2. **The starter kit switches to recipe B before the sessions.** It is prepared on a branch in Task 4 and merged at launch, so the sessions start from the same starter kit participants download.
3. **Folktales runs 10-minute runs**, like TinyStories. Its overfitting is part of what the library shows, so its README section says so beside the scores.
4. **The agent's model is `claude-fable-5`**, as on 12 October, so the sessions compare.
5. **Names:**
   - `library/baselines-b-10min/`
   - sessions `tinystories-b-10min-<date>` and `folktales-b-10min-<date>`, where the date is the launch day
   - bench folders `F:/repos/ar-bench-tinystories-b-10min` and `F:/repos/ar-bench-folktales-b-10min`

   The 5-minute baselines, `study-10min/` and the 12 October sessions stay as they are, as dated collections.

Decisions 2, 4 and 5 were the plan's proposals, which TJ's "go with the proposals" (20 October) is read as covering.

## Global Constraints

- One thing trains at a time: the GPU and the active dataset are machine-wide. Docker's vLLM model proxy stays off during training. `session.py --check` refuses only at 2 GB or more held by others, so check `nvidia-smi` by eye.
- **At launch:**
  - the machine is plugged in, with Windows' power mode fixed for the whole run and the vents clear
  - C: (the data cache) and F: (the benches and the page file) each have at least 50 GB free
- No fixed file (`prepare.py`, `capture.py`, `record.py`, `lab.py`, `program.md`, `pyproject.toml`, `uv.lock`) changes while a session is active; `record.py` checks their hashes.
- Every run trains 600 s (`time_budget_s`).
- **Speed floors**, TinyStories with its own vocabulary at 600 s:
  - recipe A at least 1,190 steps (95% of 1,258)
  - recipe B at least 2,980 (95% of 3,137)
  - a session's three starts within 5% of each other and above the floor where there is one
- Part 1 trains nothing. Every prep step that touches the GPU or the cache ends with the default pair (TinyStories, own) active again.
- One branch per repo:
  - worked example: `plan/recipe-b-studies` (#35, the plan and its prep code)
  - starter kit: `starter/recipe-b`

  TJ merges. Line endings LF. Commit messages end with the session's Co-Authored-By trailer.
- `llm-explorables`' `training/library` is synced only after the new `/training/` is built (its build plan, "After the laptop sync").

## Review Focus

1. **A session started while Windows still slows the process.** Expected: the speed gate stops it after the three starts, not ten hours later. Tested in Task 3; the gate is a step in Launches 2 and 3.
2. **`prepare.py` imported off Windows** (`report.py` on another machine, the tests anywhere). Expected: nothing is called and nothing is printed. Tested in Task 1.
3. **Recipe B drifting from what was measured** (an edit to either copy). Expected: a test fails on its version. Tested in Tasks 2 and 4.
4. **A bench copied from a working tree** (stray `__pycache__`, `.env`, checkpoints, a `.git`). Expected: the bench comes from `git archive` of a pinned commit, so it holds only tracked files. Checked in Task 6.
5. **The driver downloading or rebuilding data mid-training**, or moving the wrong run file when a run fails. Expected: `train` stops before the first run if any pair is not already cached; each pair gets its own named file; a failed run leaves none. Tested by the dry run in Task 3.

---

## Part 1: Prep

### Task 1: Training opts out of Windows power throttling (worked example)

**Files:**
- Modify: `prepare.py`.
- Test: `tests/test_power_throttling.py`, new.
- Modify: `README.md`. The "Keep the machine at one speed" paragraph names the 13 October finding and TJ's reading of it, and says the lab now opts out by itself.

**Interfaces:**
- Produces: `prepare.opt_out_of_power_throttling(kernel32=None) -> bool`. It returns True when Windows accepted the request and False otherwise, including off Windows. It is called once at import when `sys.platform == "win32"`. Task 4 copies the same diff to the starter kit.

- [ ] **Step 1: Failing tests** in `tests/test_power_throttling.py`:
  - `test_it_asks_windows_to_stop_throttling`: a stand-in `kernel32` whose `GetCurrentProcess()` returns 7 and whose `SetProcessInformation(handle, info_class, state_ref, size)` records its arguments and returns 1. Assert the call returns True, and the one recorded call is `(7, 4, Version 1, ControlMask 0x1, StateMask 0, size 12)`. Here 4 is ProcessPowerThrottling, 0x1 is EXECUTION_SPEED, and 12 is three ULONGs. Read the structure's fields through `state_ref._obj`.
  - `test_a_refusal_returns_false_and_does_not_raise`: `SetProcessInformation` returns 0, so the call returns False.
  - `test_off_windows_nothing_is_called`: `monkeypatch.setattr(sys, "platform", "linux")`, then a `kernel32` stand-in that raises on any attribute. The call returns False.
- [ ] **Step 2:** `uv run python -m pytest tests/test_power_throttling.py -v`. Expected: it fails with `AttributeError: module 'prepare' has no attribute 'opt_out_of_power_throttling'`.
- [ ] **Step 3: Implement it in `prepare.py`**, after the imports, with the bench's `unthrottled.py` structure and constants.
  - The real `kernel32` is `ctypes.WinDLL("kernel32", use_last_error=True)`, created only on Windows.
  - On refusal, print one line naming the Windows error and carry on; a run must never fail over it.
- [ ] **Step 4:** `uv run python -m pytest`. Expected: everything passes.
- [ ] **Step 5: Check in a real process:** `uv run python -c "import prepare, sys; sys.exit(0 if prepare.opt_out_of_power_throttling() else 1)"`. Expected: exit 0, with no line printed. This trains nothing.
- [ ] **Step 6: Commit.**

### Task 2: Recipe B as a file (worked example)

**Files:**
- Create: `library/recipes/recipe-b.py`, the bench's `recipe-b.py`, LF.
- Create: `library/recipes/make_recipe_b.py`, the bench's script. It reads the starter's `train.py` (`git show 6ad8ddd:train.py`) instead of `recipe-a.py`, so B can be remade from its source.
- Test: `tests/test_recipes.py`.

**Interfaces:**
- Produces: the path `library/recipes/recipe-b.py`, which Tasks 3 and 4 use.

- [ ] **Step 1: Failing tests:**
  - `test_recipe_b_is_the_version_measured`: `record.version_id(Path("library/recipes/recipe-b.py").read_text(encoding="utf-8")) == "c9ba08e4f442"`.
  - `test_recipe_b_makes_only_the_three_attention_changes`: the file contains `N_KV_HEAD = 3`, `WINDOW_PATTERN = "L"` and `is_causal=is_causal`, and contains no `softcap`. Every other upper-case setting matches the starter's: `MATRIX_LR = 0.05`, `WARMDOWN_RATIO = 0.45`, `DEPTH = 6`, `HEAD_DIM = 128`.
- [ ] **Step 2:** `uv run python -m pytest tests/test_recipes.py -v`. Expected: it fails with `FileNotFoundError`.
- [ ] **Step 3:** Add both files. Check that `uv run library/recipes/make_recipe_b.py` reproduces `recipe-b.py` byte for byte.
- [ ] **Step 4:** Run the tests. Expected: they pass.
- [ ] **Step 5: Commit.**

### Task 3: The speed gate and the recipe B baseline driver (worked example)

**Files:**
- Create: `library/speed_gate.py`.
- Create: `library/make_baselines_b.sh`, modelled on `library/make_study.sh`.
- Test: `tests/test_speed_gate.py`.

**Interfaces:**
- Produces `speed_gate.at_speed(runs: list[dict], floor: int, spread: float = 0.05) -> tuple[bool, str]`. It reads `final.num_steps` from each run. It fails when any run is under the floor, or when the runs differ by more than `spread` of the largest. The message names the steps, the floor and the README section.
- Produces the command line `uv run library/speed_gate.py <floor> <run.json>...`: exit status 0 at speed and 1 not, printing the message.
- Produces `bash library/make_baselines_b.sh prep|train`:
  - `prep` runs `prepare.py` for the six pairs, downloading or building whatever is missing. It trains nothing and ends with the default pair active.
  - `train` first runs the same step and stops if any pair's output does not say its data and tokenizer were already there. Then it trains the speed check and the six pairs.
  - Run files go to `library/baselines-b-10min/runs/<dataset>-<tokenizer>.json`, weights to `library/checkpoints/b10-<dataset>-<tokenizer>.pt`, logs to `library/logs/b10-*.log`.
  - `DRY_RUN=1 bash library/make_baselines_b.sh train` prints each planned run without preparing or training.

- [ ] **Step 1: Failing tests:**
  - `test_runs_at_speed`: `at_speed([{"final": {"num_steps": 1258}}], 1190) == (True, …)`.
  - `test_a_run_under_the_floor_fails_by_its_steps`: 1,183 steps fails, and the message contains "1,183" and "1,190".
  - `test_starts_too_far_apart_fail`: 1,258, 1,258 and 1,193 against a floor of 1,100 fail, since they are 5.2% apart.
  - `test_cli_exit_status`: through `subprocess.run([sys.executable, "library/speed_gate.py", "1190", <tmp run file>])`, exit 0 at 1,258 steps and 1 at 1,183.
- [ ] **Step 2:** Run the tests. Expected: they fail (no module).
- [ ] **Step 3: Implement `speed_gate.py`.**
- [ ] **Step 4: Write `make_baselines_b.sh`:**
  - `train` sets `AUTORESEARCH_TIME_BUDGET=600`.
  - The speed check comes first: `git show 6ad8ddd:train.py > train.py`, then one run on TinyStories with its own vocabulary, gated by `speed_gate.py 1190`. Its file goes to `library/logs/b10-speed-check-a.json`, not into the collection.
  - Then `cp library/recipes/recipe-b.py train.py` and the six pairs, TinyStories with its own vocabulary first, gated by `speed_gate.py 2980`.
  - At the end, `git checkout -- train.py` and back to the default pair, as `make_study.sh` does.
  - Each run is moved to its pair's name only if a new run file appeared.
- [ ] **Step 5: Run the checks.** `DRY_RUN=1 bash library/make_baselines_b.sh train` prints seven runs in order, each with its recipe version (`91ffb4c37015`, then `c9ba08e4f442`) and its destination, and touches nothing. Then run the tests. Expected: they pass.
- [ ] **Step 6: Commit.**

### Task 4: The starter kit starts from recipe B (`autoresearch-starter`, branch `starter/recipe-b`)

**Files:**
- Modify: `prepare.py`, and add `tests/test_power_throttling.py`: Task 1's diff, byte for byte.
- Modify: `train.py`, replaced by the worked example's `library/recipes/recipe-b.py`.
- Test: `tests/test_repo_files.py`, adding `test_the_starter_recipe_is_recipe_b`: `record.version_id(Path("train.py").read_text(encoding="utf-8")) == "c9ba08e4f442"`.
- Modify: `README.md`.
  - The one-speed paragraphs get Task 1's wording.
  - The steps quoted under "My baseline runs disagree", and the rehearsal paragraph, describe recipe A's history and should say so.
  - Item 6 of the ideas list (`WINDOW_PATTERN "L"`) now describes the default, so rewrite or drop it.
- Modify (worked example): `docs/TODO.md` and `library/README.md` say the starter kit switches to recipe B on this branch, and that sessions before it started from `91ffb4c37015`.

**Interfaces:**
- Consumes: Task 1's diff to `prepare.py` and its test; Task 2's `library/recipes/recipe-b.py`.
- Produces: the starter commit that Task 6's benches are cut from.

- [ ] **Step 1:** Write the failing version test and run it. Expected: it fails, since the version is still `91ffb4c37015`.
- [ ] **Step 2:** Copy in Task 1's diff and recipe B, then run the full tests. Expected: they pass.
- [ ] **Step 3:** `grep -n -E "SSSL|N_KV_HEAD = 1|softcap|1,214|1,258" README.md DEMO.md HARDWARE.md`. Each hit is either updated or confirmed to describe history.
- [ ] **Step 4:** Commit, push, and open a PR in `autoresearch-starter`, not merged. Note the commit for Task 6.

### Task 5: The six pairs' data and vocabularies in the cache

- [ ] **Step 1:** `nvidia-smi` shows no other process training, and no session is active (`session-logs/runner.lock` is absent in every bench).
- [ ] **Step 2:** `bash library/make_baselines_b.sh prep`. Expected: six pairs prepared, each reporting its data and tokenizer as already there or freshly built, and "active: tinystories / own" at the end.
- [ ] **Step 3:** Record the free space on C: and F: in the ledger.

### Task 6: The two session folders

- [ ] **Step 1: Make each bench from Task 4's commit**, git-tracked files only, with no `.git`. For each `<name>` in `tinystories-b-10min` and `folktales-b-10min`: `mkdir F:/repos/ar-bench-<name> && git -C F:/repos/autoresearch-starter archive <Task 4 commit> | tar -x -C F:/repos/ar-bench-<name>`.
- [ ] **Step 2: Check it is clean:**
  - `find F:/repos/ar-bench-<name> -name .git -o -name .env -o -name __pycache__ -o -name '*.pt'` prints nothing
  - the version of the bench's `train.py` is `c9ba08e4f442`
- [ ] **Step 3:** `uv sync` in each bench.
- [ ] **Step 4:** `uv run session.py <dataset> own --check --model claude-fable-5` in each bench. Expected: every check passes; the missing `CLAUDE_CODE_OAUTH_TOKEN` is a note that does not stop it. The sign-in test spends one one-word reply. Nothing is started.

---

## Part 2: Launch (TJ, whenever it suits)

### Before any launch

- [ ] Merge #35 (this plan and its prep) and the starter PR (Task 4). If the starter PR changed after the benches were made, make them again (Task 6).
- [ ] For a session: run `claude setup-token` (it opens a browser) and set `CLAUDE_CODE_OAUTH_TOKEN` in the terminal that will run it.
- [ ] Readiness:
  - the machine is plugged in, the power mode is fixed, and the vents are clear
  - Docker Desktop is quit; `nvidia-smi` shows no other process
  - C: and F: each have at least 50 GB free

### Launch 1: The recipe B baselines (about 1 hour 50 minutes)

- [ ] In the worked example, on `master` with #35 merged: `bash library/make_baselines_b.sh train`. Expected: the speed check passes, and six run files appear, each with `"time_budget_s": 600`. The driver stops by itself if the machine is slow.

### Launch 1b: The 5-minute pair (about 30 minutes; added 21 October)

For the one-recipe page (TJ, 21 October: "proceed with one-recipe version"), tab 4's 5-against-10-minute comparison, with Folktales' overfitting, needs the starter recipe at 5 minutes too. A 10-minute run's 5-minute moment has no score, and it is still at full learning rate.

- [ ] After Launch 1, with this change merged: `MINUTES=5 PAIRS="tinystories:own folktales:own" bash library/make_baselines_b.sh train`. Expected: two run files in `library/baselines-b-5min/runs/`, each with `"time_budget_s": 300`. TinyStories is gated at 47.5% of Launch 1's TinyStories steps.

### Launch 2: The TinyStories session (10 hours)

- [ ] In `F:/repos/ar-bench-tinystories-b-10min`: `uv run session.py tinystories own --name tinystories-b-10min-<date> --model claude-fable-5`.
- [ ] **The gate**, after about 40 minutes, once `uv run lab.py status` shows the three starts kept: `uv run F:/repos/autoresearch-win-rtx/library/speed_gate.py 2980 sessions/<name>/runs/*_c9ba08e4f442.json`. If it fails: `uv run session.py --stop`, fix the machine, and start again under a new `--name`.

### Launch 3: The Folktales session (10 hours)

- [ ] As Launch 2, in `F:/repos/ar-bench-folktales-b-10min`, with `folktales own` and `--name folktales-b-10min-<date>`. The gate's floor is the Folktales recipe B baseline's steps less 5% if Launch 1 has run; otherwise there is no floor, and the three starts must be within 5% of each other.

### After each launch: into the library (an agent, then TJ)

- [ ] **The baselines:**
  - `library/baselines-b-10min/collection.json`, kind `baselines`; show TJ the title and note first
  - `uv run library/report.py`
  - a `library/README.md` section, "The recipe B baselines", in the ten-minute study's form: steps, passes and score against recipe A at 10 minutes. On Folktales it says plainly how the extra passes show up as overfitting.
- [ ] **Each session:**
  - follow #34's checklist: copy `runs/`, `results.tsv`, `session.json` and `versions/`, leaving out `__pycache__`
  - check that every scoreboard row has its own run file
  - show TJ the titles and notes before writing them
  - `uv run library/report.py`
  - a README table row and a short section, from the mean of the three starts
- [ ] **Update `docs/TODO.md`,** and open a PR each time.
- [ ] **Check with the mockup** in `llm-explorables`, without changing `training/`: `python docs/training/how-a-model-gets-made/build.py --library <checkout>/library --session sessions/<name> --artifact <temp>`. It must build, and tab 5 must read right at 1280 and 390. Then rebuild the default so `git status` is clean.

## Time

| Part | Who | Time |
|---|---|---|
| Part 1 | an agent | a few hours; no training |
| Before any launch | TJ | minutes: two merges and a token |
| Launch 1 | the laptop | about 1 hour 50 minutes |
| Launches 2 and 3 | the laptop | 10 hours each |
| Into the library | an agent, then TJ | about an hour each |

## Afterwards (not in this plan)

- The new `/training/` (its build plan's decision 6) can ship with `baselines-b-10min/`, so tabs 1 to 4 and tab 5 all read "ten minutes".
- The hosted demo (`deploy/`) serves the 12 October models; serving the new nights' is a separate change.
