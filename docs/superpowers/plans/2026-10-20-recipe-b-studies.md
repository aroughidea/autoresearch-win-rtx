# Recipe B Studies: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remake the library's long training studies with recipe B as the starting recipe, on a laptop held at one speed: six baselines at 10 minutes, then a 10-hour agent session on each dataset, all brought into the library.

**Architecture:** Three small code changes land first:
- Every training run opts out of Windows power throttling, from `prepare.py` (a fixed file that every `train.py` imports, so the agent cannot edit it away). The change goes into both repos.
- Recipe B is committed as a file, with a test pinning it to the version measured on 13 October.
- A speed gate turns "the machine is slow" into a check that fails.

Then comes the machine work. One driver trains a speed check and the six baselines. Two sessions run from a fresh copy of the starter kit. Each result joins the library the way the 12 October sessions did (#34).

**Tech Stack:** Python 3 + pytest (`uv run python -m pytest`; the bare `pytest.exe` is blocked by Smart App Control in the starter), bash drivers in `library/`, `session.py`, `lab.py`, `record.py`, `uv run library/report.py`.

**Spec:** TJ, 20 October: "it looks like recipe b is a good way to start" and "we can set up and rerun the long form studies, i want a plan first". Also `docs/TODO.md`, "The starter recipe's attention setting"; the bench `ar-bench-attention-b-2026-10-13` (`make_recipe_b.py`, `measure.log`, `check-*.log`, `gpu.csv`); and `library/README.md`.

## What the record already says

- **Recipe B** is the starter recipe (`train.py` at `6ad8ddd`, version `91ffb4c37015`) with only the TinyStories night's three attention changes:
  - full key and value heads, `N_KV_HEAD = 3` (`d967b0b7055d`)
  - plain causal attention with all-long windows, `WINDOW_PATTERN = "L"` (`14c8214ecc32`)
  - no logit soft-cap (`8fa477d3ed43`)

  The bench's `make_recipe_b.py` writes it; its version is `c9ba08e4f442`. It has 26.3 M parameters, against 18.9 M for the starter.
- **Steps at full speed**, TinyStories with its own vocabulary, 10 minutes, this laptop:
  - recipe A: 1,214, 1,258 and 1,258 (12 October)
  - recipe B: about 3,100 (192 ms a step on 13 October, unthrottled); the session's equivalent version made 3,137.
- **Scores at full speed:** none yet for recipe B. The 13 October measurement (B 0.4704 to 0.4713 at about 1,400 steps; A 0.5083 at 761) ran slowed and does not compare with anything.
- **The slowdown was Windows power throttling (EcoQoS), not the disk.** On 13 October between 16:47 and 16:58 the same recipe A ran three ways, with nothing else changed:
  - launched normally: 790 ms a step
  - opted out of power throttling (the bench's `unthrottled.py`): 479 ms, the 12 October speed
  - launched normally again: 810 ms

  The GPU held about 2.3 GHz, mostly unthrottled, throughout (`gpu.csv`). Both READMEs' "we did not find what slowed the machine" predates this.
- **Folktales overfits at 10 minutes.** With the starter recipe it scores 1.3788 at 5 minutes and 1.8838 at 10, after 14 passes. Recipe B fits about 2.5 times the steps, so about 35 passes in 10 minutes. On Folktales it may well score worse than recipe A at 10 minutes. Untested.
- **The machine today:** C: has 324 GB free and F: 1 TB, the page file is on F: (120 GB), and there is 64 GB of RAM.

## Decisions for TJ before Task 1

1. **Fix the slowdown in code.** Proposed: yes. `prepare.py` opts every training process out of power throttling (Task 1), in both repos, so no session depends on how Windows treats a background process. Alternative: README advice only.
2. **When the starter kit switches to recipe B.** Proposed: after the baselines (Task 4) and before the sessions, so the sessions start from the same starter kit participants download. Do it only if no pair scores markedly worse than recipe A at 10 minutes (the study's figures). Alternative: switch now.
3. **Folktales run length.** Proposed: decide at Task 4's gate, from Folktales (own vocabulary) under recipe B at 10 minutes against recipe A at 10 minutes (1.8838) and at 5 minutes (1.3788). If 10 minutes overfits badly, the Folktales session runs 5-minute runs (`--run-minutes 5`), and its three starts are its baseline.
4. **The agent's model.** Proposed: `claude-fable-5`, as on 12 October, so the sessions compare. Another model is a separate study (`docs/TODO.md`).
5. **Names.** Proposed:
   - `library/baselines-b-10min/`
   - sessions `tinystories-b-10min-<date>` and `folktales-b-<5|10>min-<date>` (`session.py --name`)

   The 5-minute baselines, `study-10min/` and the 12 October sessions stay as they are, as dated collections.

## Global Constraints

- One thing trains at a time: the GPU and the active dataset are machine-wide. Docker's vLLM model proxy (hand-started) stays off. `session.py --check` refuses only at 2 GB or more held by others, so check `nvidia-smi` by eye.
- The machine stays plugged in, with Windows' power mode fixed for the whole run and the vents clear.
- No fixed file (`prepare.py`, `capture.py`, `record.py`, `lab.py`, `program.md`, `pyproject.toml`, `uv.lock`) changes while a session is active; `record.py` checks their hashes.
- Every run trains 600 s (`time_budget_s`), unless decision 3 gives Folktales 300.
- **Speed floors**, TinyStories with its own vocabulary at 600 s:
  - recipe A at least 1,190 steps (95% of 1,258)
  - recipe B at least 2,980 (95% of 3,137)
  - a session's three starts within 5% of each other and above the floor
- Push feature branches only; TJ merges. Line endings LF. Commit messages end with the session's Co-Authored-By trailer.
- `llm-explorables`' `training/library` is synced only after the new `/training/` is built (its build plan, "After the laptop sync").

## Review Focus

1. **A session started while Windows still slows the process.** Expected: the speed gate stops it after the three starts, not ten hours later. Tested in Task 3; the gate is a step in Tasks 6 and 7.
2. **`prepare.py` imported off Windows** (`report.py` on another machine, the tests anywhere). Expected: nothing is called and nothing is printed. Tested in Task 1.
3. **Recipe B drifting from what was measured** (an edit to the file). Expected: a test fails on its version. Tested in Task 2.
4. **A bench copied from a working tree** (stray `__pycache__`, `.env`, checkpoints, a `.git`). Expected: the bench comes from `git archive` of a pinned commit, so it holds only tracked files. A step in Task 6.
5. **The driver overwriting a run file**, or moving the wrong one when a run fails. Expected: each pair gets its own named file, and a failed run leaves none. Tested by the dry run in Task 3.

---

### Task 1: Training opts out of Windows power throttling (both repos)

**Files:**
- Modify: `prepare.py`, identical in both repos (`autoresearch-win-rtx` on `master`, then `autoresearch-starter` on `main`).
- Test: `tests/test_power_throttling.py`, new, the same in both.
- Modify: `README.md`, both. The "Keep the machine at one speed" paragraph (worked example) and "A PC that runs at one speed" plus "My baseline runs disagree" (starter) name the 13 October finding and say the lab now opts out by itself.

**Interfaces:**
- Produces: `prepare.opt_out_of_power_throttling(kernel32=None) -> bool`. It returns True when Windows accepted the request and False otherwise, including off Windows. It is called once at import when `sys.platform == "win32"`.

- [ ] **Step 1: Failing tests** in `tests/test_power_throttling.py`:
  - `test_it_asks_windows_to_stop_throttling`: a stand-in `kernel32` whose `GetCurrentProcess()` returns 7 and whose `SetProcessInformation(handle, info_class, state_ref, size)` records its arguments and returns 1. Assert the call returns True, and the one recorded call is `(7, 4, Version 1, ControlMask 0x1, StateMask 0, size 12)`. Here 4 is ProcessPowerThrottling, 0x1 is EXECUTION_SPEED, and 12 is three ULONGs. Read the structure's fields through `state_ref._obj`.
  - `test_a_refusal_returns_false_and_does_not_raise`: `SetProcessInformation` returns 0, so the call returns False.
  - `test_off_windows_nothing_is_called`: `monkeypatch.setattr(sys, "platform", "linux")`, then a `kernel32` stand-in that raises on any attribute. The call returns False.
- [ ] **Step 2:** `uv run python -m pytest tests/test_power_throttling.py -v`. Expected: it fails with `AttributeError: module 'prepare' has no attribute 'opt_out_of_power_throttling'`.
- [ ] **Step 3: Implement it in `prepare.py`**, after the imports, with the bench's `unthrottled.py` structure and constants.
  - The real `kernel32` is `ctypes.WinDLL("kernel32", use_last_error=True)`, created only on Windows.
  - On refusal, print one line naming the Windows error and carry on; a run must never fail over it.
- [ ] **Step 4:** `uv run python -m pytest`. Expected: everything passes.
- [ ] **Step 5: On the laptop, launch a run the plain way** (the way that was slow): `AUTORESEARCH_TIME_BUDGET=60 uv run train.py` with TinyStories and its own vocabulary. Expected: the run log's `dt` is at most 500 ms a step after warm-up, and no refusal line appears.
- [ ] **Step 6: Commit and push** branch `fix/power-throttling` in both repos, and open a PR in each. The starter's PR says the diff is the worked example's.

### Task 2: Recipe B as a file (worked example)

**Files:**
- Create: `library/recipes/recipe-b.py`, the bench's `recipe-b.py`, LF.
- Create: `library/recipes/make_recipe_b.py`, the bench's script. It reads the starter's `train.py` (`git show 6ad8ddd:train.py`) instead of `recipe-a.py`, so B can be remade from its source.
- Test: `tests/test_recipes.py`.

**Interfaces:**
- Produces: the path `library/recipes/recipe-b.py`, which Tasks 3 and 5 copy over `train.py`.

- [ ] **Step 1: Failing tests:**
  - `test_recipe_b_is_the_version_measured`: `record.version_id(Path("library/recipes/recipe-b.py").read_text(encoding="utf-8")) == "c9ba08e4f442"`.
  - `test_recipe_b_makes_only_the_three_attention_changes`: the file contains `N_KV_HEAD = 3`, `WINDOW_PATTERN = "L"` and `is_causal=is_causal`, and contains no `softcap`. Every other upper-case setting matches the starter's values: `MATRIX_LR = 0.05`, `WARMDOWN_RATIO = 0.45`, `DEPTH = 6`, `HEAD_DIM = 128`.
- [ ] **Step 2:** `uv run python -m pytest tests/test_recipes.py -v`. Expected: it fails with `FileNotFoundError`.
- [ ] **Step 3:** Add both files. Check that `uv run library/recipes/make_recipe_b.py` reproduces `recipe-b.py` byte for byte.
- [ ] **Step 4:** Run the tests. Expected: they pass.
- [ ] **Step 5: Commit** on branch `library/recipe-b`.

### Task 3: The speed gate and the recipe B baseline driver

**Files:**
- Create: `library/speed_gate.py`.
- Create: `library/make_baselines_b.sh`, modelled on `library/make_study.sh`.
- Test: `tests/test_speed_gate.py`.

**Interfaces:**
- Produces `speed_gate.at_speed(runs: list[dict], floor: int, spread: float = 0.05) -> tuple[bool, str]`. It reads `final.num_steps` from each run. It fails when any run is under the floor, or when the runs differ by more than `spread` of the largest. The message names the steps, the floor and the README section.
- Produces the command line `uv run library/speed_gate.py <floor> <run.json>...`: exit status 0 at speed and 1 not, printing the message.
- Produces `bash library/make_baselines_b.sh`, which writes `library/baselines-b-10min/runs/<dataset>-<tokenizer>.json`, weights to `library/checkpoints/b10-<dataset>-<tokenizer>.pt`, and logs to `library/logs/b10-*.log`. `DRY_RUN=1` prints each planned run without training.

- [ ] **Step 1: Failing tests:**
  - `test_runs_at_speed`: `at_speed([{"final": {"num_steps": 1258}}], 1190) == (True, …)`.
  - `test_a_run_under_the_floor_fails_by_its_steps`: 1,183 steps fails, and the message contains "1,183" and "1,190".
  - `test_starts_too_far_apart_fail`: 1,258, 1,258 and 1,193 against a floor of 1,100 fail, since they are 5.2% apart.
  - `test_cli_exit_status`: through `subprocess.run([sys.executable, "library/speed_gate.py", "1190", <tmp run file>])`, exit 0 at 1,258 steps and 1 at 1,183.
- [ ] **Step 2:** Run the tests. Expected: they fail (no module).
- [ ] **Step 3: Implement `speed_gate.py`.**
- [ ] **Step 4: Write `make_baselines_b.sh`:**
  - `AUTORESEARCH_TIME_BUDGET=600`.
  - First a speed check: `git show 6ad8ddd:train.py > train.py`, then one run on TinyStories with its own vocabulary, gated by `speed_gate.py 1190`. Its file goes to `library/logs/b10-speed-check-a.json`, not into the collection. The driver stops if the machine is slow.
  - Then `cp library/recipes/recipe-b.py train.py` and the six pairs, TinyStories with its own vocabulary first, gated by `speed_gate.py 2980`.
  - At the end, `git checkout -- train.py` and back to the default dataset, as `make_study.sh` does.
  - Each run is moved to its pair's name only if a new run file appeared.
- [ ] **Step 5: Run the checks.** `DRY_RUN=1 bash library/make_baselines_b.sh` prints seven runs in order, each with its recipe version (`91ffb4c37015`, then `c9ba08e4f442`) and its destination, and trains nothing. Then run the tests. Expected: they pass.
- [ ] **Step 6: Commit** on branch `library/recipe-b`, then push and open the PR (Tasks 2 and 3 together).

### Task 4: Train the recipe B baselines; the gate

Machine time is about 1 hour 50 minutes: seven runs of about 13.5 minutes, plus `prepare.py` for each vocabulary. Tasks 1 to 3 must be merged first.

- [ ] **Step 1: Readiness.**
  - The machine is plugged in, the power mode is fixed, and Docker is stopped.
  - `nvidia-smi` shows no other process.
  - `git status` is clean on `master` at the merged commit.
- [ ] **Step 2: Run** `bash library/make_baselines_b.sh`. Expected: the speed check passes, and six run files appear, each with `"time_budget_s": 600`.
- [ ] **Step 3: Check each run file.** `uv run library/speed_gate.py 2980 library/baselines-b-10min/runs/tinystories-own.json` passes. The other vocabularies have no floor yet: record their steps.
- [ ] **Step 4: Write `library/baselines-b-10min/collection.json`**, kind `baselines`. Show TJ the title and note first, for example "Six baselines, recipe B, ten minutes". Then run `uv run library/report.py` and check `index.json` lists the collection with six runs.
- [ ] **Step 5: Add a section to `library/README.md`**, "The recipe B baselines", with a table in the ten-minute study's form: dataset, tokenizer, steps (A at 10 minutes → B), passes, score (A at 10 minutes → B) and change. Recipe A's figures come from `study-10min/`. Commit, push, PR.
- [ ] **Step 6: The gate.** Bring TJ the table, with decision 2 (switch the starter kit) and decision 3 (the Folktales run length), and wait for both.

### Task 5: The starter kit starts from recipe B (if decision 2 says so)

**Files (`autoresearch-starter`, branch `starter/recipe-b`):**
- Modify: `train.py`, replaced by `library/recipes/recipe-b.py`.
- Modify: `README.md`. Update the numbers that describe the starter recipe (the steps quoted under "My baseline runs disagree" and the rehearsal paragraph). Item 6 of the ideas list (`WINDOW_PATTERN "L"`) now describes the default, so rewrite or drop it.
- Test: `tests/test_repo_files.py`, adding `test_the_starter_recipe_is_recipe_b`: `record.version_id(Path("train.py").read_text(encoding="utf-8")) == "c9ba08e4f442"`.
- Modify (worked example): `library/README.md` and `docs/TODO.md` say which starter commit switched, and that sessions before it started from `91ffb4c37015`.

- [ ] **Step 1:** Write the failing test and run it. Expected: it fails, since the version is still `91ffb4c37015`.
- [ ] **Step 2:** Replace `train.py` and run the full tests. Expected: they pass.
- [ ] **Step 3:** `grep -n -E "SSSL|N_KV_HEAD = 1|softcap|1,214|1,258" README.md DEMO.md HARDWARE.md`. Each hit is either updated or confirmed to describe history.
- [ ] **Step 4:** Commit, push, open a PR, and note the commit for Task 6.

### Task 6: Session 1, TinyStories, 10 hours

- [ ] **Step 1: Make the bench from a pinned commit** (git-tracked files only, no `.git`):
  `mkdir F:/repos/ar-bench-tinystories-b-10min-<date> && git -C F:/repos/autoresearch-starter archive <Task 5 commit> | tar -x -C F:/repos/ar-bench-tinystories-b-10min-<date>`.
  If decision 2 kept the starter on recipe A, copy `library/recipes/recipe-b.py` over the bench's `train.py`.
- [ ] **Step 2: Check.** In the bench, run `uv sync`, then `uv run session.py tinystories own --check`. Expected: every check passes.
- [ ] **Step 3: Start the session.** Set `CLAUDE_CODE_OAUTH_TOKEN` (a long-lived token from `claude setup-token`), then run `uv run session.py tinystories own --name tinystories-b-10min-<date> --model claude-fable-5`.
- [ ] **Step 4: The gate**, after about 40 minutes, once `lab.py status` shows the three starts kept: `uv run <worked example>/library/speed_gate.py 2980 sessions/<name>/runs/*_c9ba08e4f442.json` (the three starts carry recipe B's version). If it fails, run `uv run session.py --stop`, fix the machine, and start again under a new `--name`.
- [ ] **Step 5: In the morning,** `uv run lab.py check`. Expected: the record is consistent, and the scoreboard has one run file per row. Keep the agent's final summary.

### Task 7: Session 2, Folktales, 10 hours

- [ ] Task 6's steps with `folktales own`, `--name folktales-b-<5|10>min-<date>`, and `--run-minutes 5` if decision 3 says so. Then the gate:
  - **At 10 minutes:** the floor is the Folktales B baseline's steps less 5%.
  - **At 5 minutes:** there is no floor; the three starts must still be within 5% of each other.

### Task 8: Into the library

- [ ] In the worked example, on branch `library/recipe-b-sessions`, follow the checklist used for #34 for both sessions:
  - copy `runs/`, `results.tsv`, `session.json` and `versions/`, leaving out any `__pycache__`
  - check that every scoreboard row has its own run file, matched by commit and score
  - show TJ the `collection.json` titles and notes before writing them
  - run `uv run library/report.py`; check that `index.json` lists both sessions with every run, and that `copies.json` has the Folktales runs
  - add a README table row and a short section for each, starting from the mean of the three starts
- [ ] Update `docs/TODO.md`: done, with each night's start, best, gain against its noise, and the agent's model. Then commit, push and open a PR.
- [ ] Check with the mockup in `llm-explorables` without changing `training/`: `python docs/training/how-a-model-gets-made/build.py --library <checkout>/library --session sessions/<name> --artifact <temp>` for each session. Both must build, and tab 5 must read right at 1280 and 390. Then rebuild the default so `git status` is clean.

## Time

| Part | Who | Time |
|---|---|---|
| Tasks 1 to 3 | an agent, then TJ's review and merge | about 2 hours of work |
| Task 4 | the laptop | about 1 hour 50 minutes |
| Task 5 | an agent, then TJ | under an hour |
| Tasks 6 and 7 | the laptop | 10 hours each: a day and a night, or two nights |
| Task 8 | an agent, then TJ | about an hour |

About three days in all, most of it the two sessions.

## Afterwards (not in this plan)

- The new `/training/` (its build plan's decision 6) can ship with `baselines-b-10min/`, so tabs 1 to 4 and tab 5 all read "ten minutes".
- The hosted demo (`deploy/`) serves the 12 October models; serving the new nights' is a separate change.
