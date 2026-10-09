# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may22-am`) and use it in commit messages/tags for milestones.
2. **Work in this folder**: the session's record lives in `sessions/<name>/`. Never use git for experiments; `lab.py` keeps and undoes changes.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `capture.py` — records what the model writes during training and saves one run file per experiment in `runs/`. Do not modify.
   - `record.py` and `lab.py` — the experiment record and its command line. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check the autoresearch cache directory. On Windows this is `%LOCALAPPDATA%\autoresearch` (e.g. `C:\Users\<you>\AppData\Local\autoresearch`) — unless `AUTORESEARCH_CACHE_DIR` is set, or a legacy `~/.cache/autoresearch` directory already exists (resolution order is defined in `_default_cache_dir()` in `prepare.py`). It should contain: `datasets\<dataset>\data\` with the downloaded parquet file (default: `tinystories_gpt4_clean.parquet` — the data stays as a single parquet file; there are no pre-tokenized shards), `datasets\<dataset>\tokenizer\` (or `tokenizer-<name>\` when a standard tokenizer such as `phi3` or `gpt2` is active) with `tokenizer.pkl` and `token_bytes.pt`, and `active_dataset.txt` and `active_tokenizer.txt` at the cache root (no `active_tokenizer.txt` means `own`). If any of these are missing, tell the human to run `uv run prepare.py`.
5. **Check the time budget**: `AUTORESEARCH_TIME_BUDGET` must not be set (`echo $AUTORESEARCH_TIME_BUDGET` in bash, `$env:AUTORESEARCH_TIME_BUDGET` in PowerShell: both print nothing). A person sets it for a study; inherited from their shell, it would make every experiment train for that long instead of 5 minutes. If it is set, tell the human and do not start. Each run's log also prints `Time budget: 300s`; any other number means the same.
6. **Check the session**: run `uv run lab.py status`. If it says there is no active session, start one: `uv run lab.py start <tag> --dataset <active dataset> --tokenizer <active tokenizer> --hours <session budget>`. The session's `results.tsv` is written by `lab.py`; never edit it.
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

## The record

Every run is recorded automatically. `capture.py` stores the code that trained (in `sessions/<name>/versions/`) and the run's entry (in `sessions/<name>/runs/`) as training starts and ends. Your decision completes the entry:

- `uv run lab.py keep "<description>"`: the run's version becomes the best version, and its model is copied to `checkpoints/`.
- `uv run lab.py undo "<description>"`: the run is recorded as `discard` (or `crash` if it never finished), and `train.py` is reset to the best version.

The description is a few words on what the experiment tried. `lab.py` then rewrites the session's `results.tsv` (`timestamp commit val_bpb memory_gb status description`, where `commit` is the version id). Read the record with `uv run lab.py history` (one line per run) and `uv run lab.py diff` (what changed).

## The experiment loop

Each session keeps its record in `sessions/<name>/`.

LOOP UNTIL THE SESSION BUDGET RUNS OUT. Before each experiment, check the time with `uv run lab.py status`: if less than 20 minutes of the budget remain, do not start another; go to **Ending the session** below.

1. `uv run lab.py status`: `train.py` should be the same as the best version, with no run waiting for a decision.
2. Read `uv run lab.py history` (and `uv run lab.py diff` when useful), then change one thing in `train.py`.
3. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context). `capture.py` records the version and the run.
4. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
5. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, run `uv run lab.py undo "<what was tried>"`.
6. If val_bpb improved (lower), under the noise rule: `uv run lab.py keep "<description>"`. If it is equal or worse: `uv run lab.py undo "<description>"`.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, undo. The best version moves forward so that you can iterate.

**Timeout**: Each experiment trains for ~5 minutes, plus startup, the fixed validation eval, and the writing samples `capture.py` records for the run file (about 20 seconds on an RTX 4000 Ada laptop GPU, more on slower GPUs; `sampling_s` in the run file). On consumer GPUs the eval alone can add 2–3 minutes, so a healthy run totals 8–11 minutes (`total_seconds` in the summary tells you exactly). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, run `uv run lab.py undo "<what was tried>"` (the record marks it a crash) and move on.

**One experiment at a time**: an experiment is finished only when you have run `lab.py keep` or `lab.py undo` (step 6); `capture.py` will not train a different `train.py` until you have. Do not stack several changes and log them afterwards. A sweep (say 0.09, then 0.12, then 0.16) is several experiments: run, log, and keep or discard each one on its own, so the scoreboard shows what each change did. A result that beats the best is a keep once the noise rule's second run confirms it, even if you plan to go further.

**Don't stop early; stop on time**: once the loop has begun, do NOT pause to ask the human whether to continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer, and expects you to keep working until the session budget runs out. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes.

**Ending the session**: when less than 20 minutes of the budget remain, start no new experiment. Make sure `uv run lab.py status` shows no run waiting for a decision and `train.py` the same as the best version, then write a short summary as your final message: the best score and how it compares with the baseline, which changes helped, which did not, and what you would try next. Then stop. (Karpathy's original loop runs until the human interrupts it; this fork ends on a budget so that every session finishes on a clean, logged state.)

As an example use case, a user might leave you running while they sleep. Each experiment takes 8–11 minutes on a consumer GPU, so you can run about 6 an hour: an eight-hour session holds roughly 45–55. The user then wakes up to experimental results, all completed by you while they slept!
