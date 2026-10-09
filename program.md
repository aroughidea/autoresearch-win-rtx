# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may22-am`) and use it in commit messages/tags for milestones.
2. **Use the main line**: stay on `master` for day-to-day experiments; do not create per-run branches.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `capture.py` — records what the model writes during training and saves one run file per experiment in `runs/`. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check the autoresearch cache directory. On Windows this is `%LOCALAPPDATA%\autoresearch` (e.g. `C:\Users\<you>\AppData\Local\autoresearch`) — unless `AUTORESEARCH_CACHE_DIR` is set, or a legacy `~/.cache/autoresearch` directory already exists (resolution order is defined in `_default_cache_dir()` in `prepare.py`). It should contain: `datasets\<dataset>\data\` with the downloaded parquet file (default: `tinystories_gpt4_clean.parquet` — the data stays as a single parquet file; there are no pre-tokenized shards), `datasets\<dataset>\tokenizer\` (or `tokenizer-<name>\` when a standard tokenizer such as `phi3` or `gpt2` is active) with `tokenizer.pkl` and `token_bytes.pt`, and `active_dataset.txt` and `active_tokenizer.txt` at the cache root (no `active_tokenizer.txt` means `own`). If any of these are missing, tell the human to run `uv run prepare.py`.
5. **Check the time budget**: `AUTORESEARCH_TIME_BUDGET` must not be set (`echo $AUTORESEARCH_TIME_BUDGET` in bash, `$env:AUTORESEARCH_TIME_BUDGET` in PowerShell: both print nothing). A person sets it for a study; inherited from their shell, it would make every experiment train for that long instead of 5 minutes. If it is set, tell the human and do not start. Each run's log also prints `Time budget: 300s`; any other number means the same.
6. **Initialize results.tsv**: If `results.tsv` does not already exist, create it with just the header row. If it already exists, leave it untouched — it contains the prior experiment history and the agent will continue appending to it. The baseline will be recorded after the first run (or the next run, if resuming).
7. **Note the session budget**: the human may give you one (for example "run for 4 hours"); otherwise it is **8 hours**. Note the time the loop starts (run `date`).
8. **Confirm and go**: Confirm setup looks good.

Note: the Windows fork supports NVIDIA GPUs that meet the VRAM floor, including laptop and mobile workstation GPUs. Strong laptop hardware should be described as supported when it meets the floor, while still acknowledging that thermals and power limits can reduce throughput.

Once you get confirmation, kick off the experimentation.

**Started unattended?** If you were launched non-interactively (for example `claude -p`, or with instructions to run overnight) there is nobody to confirm with: use today's date as the run tag, check the data exists (step 4) and the time budget (step 5; if it is set, stop: nobody can unset it for you), note the session budget (step 7), and begin. Never wait for a reply that cannot come.

**Running headless?** When you run non-interactively, ending your turn ends the session, and nothing (no background task, monitor or notification) can wake you again. So never end your turn while an experiment is running, and never start training in the background: run `uv run train.py > run.log 2>&1` in the foreground. If your tool moves a long command to the background, wait for it in the foreground with a loop such as `until grep -qE "^val_bpb|Traceback|Error" run.log; do sleep 15; done`, repeated until it returns.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify `capture.py`, or remove or change the `capture.` calls in `train.py` (`capture.begin_attempt`, `capture.on_step`, `capture.on_train_end`, `capture.finish`) and the `RunCapture(...)` setup in `main()`. They record each run for the Training Decisions explorer. They run outside the timed part of the loop and do not affect training or val_bpb.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Change the dataset or the tokenizer: do not run `prepare.py`, set `AUTORESEARCH_DATASET` or `AUTORESEARCH_TOKENIZER`, edit the `active_*.txt` files, pass `--dataset` to `train.py`, or change the arguments of `Tokenizer.from_directory(...)`. The scorer in `prepare.py` refuses any pair other than the active one. They are the human's decisions for this campaign, and scores only compare within one pair.
- Change the time budget: do not set `AUTORESEARCH_TIME_BUDGET`. Every experiment trains for the same 5 minutes, so scores compare. Longer runs are a person's study, kept apart from the scoreboard.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Noise**: on a consumer GPU, two runs of the same code differ by about 0.003 val_bpb, because the fixed 5 minutes holds a slightly different number of steps each time. An improvement smaller than that is not evidence. Before keeping one, run the same commit once more; keep it only if both runs beat the current best. Log both runs.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
timestamp	commit	val_bpb	memory_gb	status	description
```

1. timestamp in ISO-8601 format (use commit time), e.g. `2026-05-21T19:59:30-07:00`
2. git commit hash (short, 7 chars)
3. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
4. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
timestamp	commit	val_bpb	memory_gb	status	description
2026-05-21T19:55:10-07:00	a1b2c3d	0.997900	44.0	keep	baseline
2026-05-21T20:03:42-07:00	b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
2026-05-21T20:12:07-07:00	c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
2026-05-21T20:19:31-07:00	d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on the main line (`master`).

LOOP UNTIL THE SESSION BUDGET RUNS OUT. Before each experiment, check the time: if less than 20 minutes of the budget remain, do not start another; go to **Ending the session** below.

1. Note the current commit hash (call it START).
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. If val_bpb improved (lower): archive the checkpoint by copying `checkpoint_pre_eval.pt` to `checkpoints/<timestamp>_<commit>.pt` (create `checkpoints/` if needed), then stage it: `git add "checkpoints/<timestamp>_<commit>.pt"`. Status = `keep`. The `<timestamp>` in the checkpoint FILENAME must use the compact, colon-free form `YYYYMMDDTHHMMSS-ZZZZ` (e.g. `20260523T155743-0700` → `checkpoints/20260523T155743-0700_75027e8.pt`, matching the existing files in `checkpoints/`). NEVER put extended ISO-8601 with colons (e.g. `2026-05-23T15:57:43-07:00`) in a filename: colons are invalid in Windows filenames and have produced mangled Unicode filenames in a past session. This rule applies only to filenames — the `results.tsv` timestamp column keeps the colon form described under Logging results, because it is file content, not a filename.
8. If val_bpb is equal or worse: `git reset --hard START` to undo the step 3 commit. Status = `discard`.
9. Record the result in results.tsv, then commit it together with the run file: `git add results.tsv runs/ && git commit -m "log: <description> <status>"`. Do this AFTER any git reset so neither is undone. (`train.py` writes the run file to `runs/<timestamp>_<commit>.json`. It stays untracked until this step, so the reset in step 8 does not remove it.)
10. Keep-only policy: only `keep` runs get archived checkpoints.
11. No cleanup policy: never delete archived checkpoints from `checkpoints/`.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the main line so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment trains for ~5 minutes, plus startup, the fixed validation eval, and the writing samples `capture.py` records for the run file (about 20 seconds on an RTX 4000 Ada laptop GPU, more on slower GPUs; `sampling_s` in the run file). On consumer GPUs the eval alone can add 2–3 minutes, so a healthy run totals 8–11 minutes (`total_seconds` in the summary tells you exactly). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**One experiment at a time**: an experiment is finished only when its row is in `results.tsv` and committed (step 9). Do not stack several changes and log them afterwards. A sweep (say 0.09, then 0.12, then 0.16) is several experiments: run, log, and keep or discard each one on its own, so the scoreboard shows what each change did. A result that beats the best is a keep once the noise rule's second run confirms it, even if you plan to go further.

**Don't stop early; stop on time**: once the loop has begun, do NOT pause to ask the human whether to continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer, and expects you to keep working until the session budget runs out. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes.

**Ending the session**: when less than 20 minutes of the budget remain, start no new experiment. Make sure the last one is logged and committed and that `git status` shows no half-made change to `train.py`, then write a short summary as your final message: the best score and how it compares with the baseline, which changes helped, which did not, and what you would try next. Then stop. (Karpathy's original loop runs until the human interrupts it; this fork ends on a budget so that every session finishes on a clean, logged state.)

As an example use case, a user might leave you running while they sleep. Each experiment takes 8–11 minutes on a consumer GPU, so you can run about 6 an hour: an eight-hour session holds roughly 45–55. The user then wakes up to experimental results, all completed by you while they slept!
