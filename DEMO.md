# DEMO.md — Instructor's Runbook

A 15-minute lecture segment: one live autonomous-research iteration, narrated end to end.
Everything here is grounded in `README.md` (setup, agent commands) and `program.md` (the
experiment loop protocol). Numbers cited below reflect the repo state as of 2026-08-06:
16 rows in `results.tsv`, current best **`e9fffd9` — val_bpb 0.518708** ("warmdown 0.55
may23-pm"), baseline **`75027e8` — 0.520096**. (`val_bpb` = validation bits per byte,
the one score every experiment optimizes; **lower is better**.)

**The one number to memorize before walking in: the current best val_bpb.** Check it with:

```powershell
Get-Content results.tsv -Tail 5
```

---

## 1. The night before

### 1a. Fresh results — run the agent overnight, or skip

**Skip this step if `results.tsv` already has recent rows** (last row within a day or two
and the chart/panes verify in step 1b). The demo does not require overnight results — it
requires *any* coherent session history, and the may23-pm session (15 rows + one resumed
row) already tells a complete story.

To run an overnight session anyway, use the exact commands from README's **Running the
agent** section, from the repo root:

**Claude Code — unattended / overnight** (skips all permission prompts, logs to `agent.log`):

```powershell
claude --dangerously-skip-permissions "Read program.md, do setup checks, and start a new experiment loop. Log each result in results.tsv." 2>&1 | Tee-Object -FilePath agent.log
```

**Codex CLI — full-access mode, with logging:**

```powershell
codex exec -s danger-full-access "Read program.md, do setup checks, and start a new experiment loop. Log each result in results.tsv." 2>&1 | Tee-Object -FilePath agent.log
```

Either way: **Ctrl+C stops the agent** in that terminal. If `train.py` is mid-run, Ctrl+C
that too — the loop commits after each experiment, so state stays consistent at any
experiment boundary.

### 1b. Morning verification checklist

Run through all six, in order:

- [ ] **results.tsv is fresh and well-formed:**
  ```powershell
  Get-Content results.tsv -Tail 5
  ```
  Note the current best (lowest val_bpb among `keep` rows) — you will quote it live.

- [ ] **chat.py loads:**
  ```powershell
  uv run chat.py
  ```
  Browser opens at `http://localhost:8000`. Confirm:
  - [ ] **Progress chart renders** at the top (it reads `results.tsv`; a reload
        re-fetches it, so it always shows the latest rows).
  - [ ] **Baseline and Best panes resolve** — labels should read
        "Baseline — baseline may23-pm" and "Best — warmdown 0.55 may23-pm" (or your
        newest session's equivalents). The model behind each pane resolves by matching
        commit hashes in `results.tsv` against filenames in `checkpoints/`. **Caution:**
        if a matching `.pt` file is missing, the pane does *not* disappear — it silently
        falls back to generating from `checkpoint_pre_eval.pt` (the last run's weights,
        possibly a discarded config). So verify the files directly:
  ```powershell
  Test-Path checkpoints/20260523T155743-0700_75027e8.pt, checkpoints/20260523T175831-0700_e9fffd9.pt
  ```
  Both must print `True`. (The Best pane is hidden only when `results.tsv` has no
  `keep` rows at all.)

- [ ] **Pick 2–3 sample prompts and TEST them.** TinyStories models (trained on short,
      GPT-4-written children's stories) are charming but
      uneven — test against the **best archived checkpoint**, not
      `checkpoint_pre_eval.pt` (that file holds whatever the *last* run produced, which
      may be a discarded config). Candidates to try:
  ```powershell
  uv run generate.py "Once upon a time" --checkpoint checkpoints/20260523T175831-0700_e9fffd9.pt
  uv run generate.py "The little dog" --checkpoint checkpoints/20260523T175831-0700_e9fffd9.pt
  uv run generate.py "One day, a little girl named Lily" --checkpoint checkpoints/20260523T175831-0700_e9fffd9.pt
  ```
  Run each 2–3 times (sampling varies). Keep the two or three that produce coherent,
  story-shaped output. **Sampling-settings caveat:** `generate.py` defaults to
  temperature 0.9 / top-k 50, but chat.py's sliders default to temperature 0 and
  top-k 0 (deterministic greedy decoding — flatter, more repetitive output). Test your
  finalists once in the chat.py UI too, or plan to set its temperature slider to ~0.9
  during the reveal so the output matches what you rehearsed.
  Write your winners here: ______________________

- [ ] **Smoke-test the training path** (README's recommended validation — ~1 minute):
  ```powershell
  uv run train.py --smoke-test
  ```

- [ ] **Wake the hosted demo and leave the tab open.** Load
      <https://autoresearch-demo.fly.dev/> once in a browser tab. It is the same UI on one
      small shared Fly.io machine that **suspends when idle and takes ~12 seconds to wake**;
      after that every page load is ~0.07 s. Fly holds the request while it wakes, so a
      cold visit is slow, never an error. Confirm the progress chart and both panes
      render, then leave the tab open. It will suspend again overnight — **reload it
      during room setup (section 4)** so it is warm if you show it live or need it as
      the Tier 2 fallback.

- [ ] **Capture the fallback screenshots** — see section 3 for the list.

---

## 2. The segment (~15 min, timed)

**Starting screen state:** one visible PowerShell terminal at the repo root (font
enlarged); VS Code with `program.md`, `results.tsv`, and `train.py` open in tabs;
`chat.py` already running hidden (section 4); browser tab at `localhost:8000` ready but
not yet shown.

### [0–2 min] The pitch

Open `program.md` on screen. The line to deliver:

> "This Markdown file is the program; the agent is the runtime."

Scroll to the **`## The experiment loop`** section and point at `LOOP FOREVER`. Framing:
a coding agent reads this file, edits the training code, trains for 5 minutes, keeps or
discards based on one number, and repeats — all night, unattended. You are about to run
one iteration of that loop by hand.

### [2–3 min] START the live experiment

In the visible terminal, run the loop exactly as `program.md` specifies:

```powershell
# Step 1 of the loop: note the current commit (START)
$start = git rev-parse --short HEAD
```

**Step 2: edit one hyperparameter live.** In VS Code, open `train.py`, find the
Hyperparameters block (~line 802):

```
MATRIX_LR = 0.045   →   MATRIX_LR = 0.05
```

This is a safe edit with a visible effect: the agent already mapped this region
(0.035 / 0.04 / 0.045 / 0.0475 / 0.06 are all in `results.tsv`; the neighbors 0.0475 and
0.06 both scored ~0.5202–0.5205 and were discarded). Expect a **discard** — which is the
more instructive live outcome anyway: the loop's whole point is that most experiments fail
cheaply.

```powershell
# Step 3: commit the experiment (pick a run tag per program.md, e.g. aug07-demo)
git add train.py; git commit -m "exp aug07-demo: matrix lr 0.05"

# Capture the two values the results row will need, BEFORE any reset:
$exp = git rev-parse --short HEAD
$ts  = git log -1 --format=%cI

# Step 4: run it, output redirected exactly as program.md says
uv run train.py > run.log 2>&1
```

Announce: "~5.5 minutes wall clock — 5 minutes of training plus ~26 seconds of
startup and evaluation. While it trains, let's look at what the agent has already done."

### [3–8 min] While it trains: the paper trail

Four artifacts, ~1 minute each:

1. **`results.tsv` — the scoreboard.** Open in VS Code. One row per experiment:
   timestamp, commit, val_bpb, VRAM, keep/discard/crash, description. Walk the may23-pm
   session: baseline 0.520096 → best 0.518708 across 15 experiments in ~2 hours. Point
   out that most rows say `discard` — that is the loop working, not failing.
2. **`git log --oneline` — the lab notebook.** Run it in a second terminal (or scroll a
   pre-run one). Every kept experiment is an `exp ...` commit; every result is a
   `log: ...` commit. The repo at HEAD *is* the research state — there is no other
   database.
3. **WALKTHROUGH.md's session table** — the annotated version of the same session, one
   row per experiment with what was tried and why. This is the supplemental reading;
   flash it now so students recognize it later.
4. **The progress chart** — bring the hidden browser tab (`localhost:8000`) forward
   briefly. Green dots are kept improvements, grey are discards, the green step line is
   the running best. Two extreme rows (the depth-7 run and the resumed warmdown-0.65
   run) are clipped off the y-axis and drawn as grey triangles pinned to the top — don't
   be surprised by them live. Then hide the tab again for the reveal.

Optional filler if time remains: `Get-Content run.log -Tail 3` in a spare terminal to
show the live progress line (step count, loss, tok/sec) — cosmetic line-wrapping in the
log file is normal.

### [8–10 min] Read out the result, make the call

When the run finishes (prompt returns):

```powershell
grep "^val_bpb:" run.log
```

(If `grep` is not on PATH in PowerShell: `Select-String '^val_bpb:' run.log`.)

Read the number aloud next to the current best (**0.518708**, or your newest best from
the morning checklist). Make the keep/discard call live — this is the whole loop
compressed into one sentence: *"Is it lower? No. Discard."*

**Discard path** (the likely one — run it live):

```powershell
git reset --hard $start
Add-Content results.tsv "$ts`t$exp`t<val_bpb>`t6.6`tdiscard`tmatrix lr 0.05 aug07-demo live"
git add results.tsv; git commit -m "log: matrix lr 0.05 aug07-demo discard"
```

(Substitute the actual val_bpb; memory_gb is `peak_vram_mb ÷ 1024` from the same summary
block — recent runs are all 6.6.) Note for the room: the losing code is erased from
history, but the *log row is never rolled back* — failures are data.

**If it actually improved** (lower val_bpb): congratulate the room, then run the keep
path from section 5 verbatim.

### [10–14 min] The reveal

Bring the browser tab (`localhost:8000`) forward. **Reload the page** — the chart now
includes the row you just logged, live on screen.

Type one of your tested prompts into the prompt box — with the temperature slider set
the way you tested in 1b — and generate. Both panes stream
side by side, token by token: **Baseline** (first kept model, `75027e8`) vs **Best**
(`e9fffd9`). Let the streaming finish without talking over it. If time allows, click the
**Tokens** toggle on one pane to show the raw tokenization, and nudge the temperature
slider for a second generation.

The point to land: both models trained for exactly 5 minutes; the only difference is
the configuration the agent found.

### [14–15 min] Close

> "Everything you saw is one public repo; the ambitious among you can run this tonight."

Starter kit = README's **Your first run** section — three commands:

```powershell
uv sync
uv run prepare.py
uv run train.py
```

...then README's **Running the agent** section to launch the loop. Requirements: an
NVIDIA GPU meeting the VRAM floor (README's Platform support table), Python 3.10+, uv,
git, git-lfs. WALKTHROUGH.md is the supplemental reading.

---

## 3. Fallbacks — decide by T-2 min before the segment

Make the call **two minutes before you start**, not mid-segment. Four tiers:

**Tier 1 — live run fails or machine misbehaves** (run crashes, GPU busy, run overruns
its slot — program.md's own rule: past 10 minutes, kill it and treat as a failure):
skip the [2–3] and [8–10] blocks. Yesterday's results are already in the TSV and chart —
narrate the session from **WALKTHROUGH.md's session table** instead, then do the reveal
([10–14]) as planned. You lose the live gamble, not the story.

**Tier 2 — chat.py fails, but the machine and the network are fine**: switch to the
hosted copy at **<https://autoresearch-demo.fly.dev/>** — the tab you woke in 1b. It is
the same UI serving the same two models (Baseline `75027e8` / Best `e9fffd9`), so the
reveal looks exactly like the one you rehearsed, side-by-side panes and all. Two
caveats: generation there is capped (max 500 tokens, top-k ≤ 200), and if the tab has
gone cold the first load takes ~12 seconds — start it loading *before* you say anything.
The chart on that page shows the committed rows, not the row you logged live, so narrate
the live row from `results.tsv` in VS Code instead of reloading for it.

**Tier 3 — chat.py fails and you have no usable network**: do the reveal in the terminal
with `generate.py`, running baseline vs best sequentially on the same prompt:

```powershell
uv run generate.py "<your tested prompt>" --checkpoint checkpoints/20260523T155743-0700_75027e8.pt
uv run generate.py "<your tested prompt>" --checkpoint checkpoints/20260523T175831-0700_e9fffd9.pt
```

**Tier 4 — total machine failure**: present from screenshots. Capture these the night
before (after the 1b checklist passes), full-window, at presentation font size:

1. chat.py full page — progress chart plus both panes populated with a tested prompt.
2. Close-up of the Baseline vs Best output side by side (Text view).
3. The Best pane in **Tokens** view.
4. `results.tsv` open in VS Code.
5. `git log --oneline` output in the terminal.
6. Tail of `run.log` showing the `---` summary block (`val_bpb:` line visible).
7. `program.md` scrolled to `LOOP FOREVER` (for the pitch).

---

## 4. Demo machine prep

- **SAC-safe Python.** If Windows Smart App Control is enabled, uv's standalone Python
  fails with `DLL load failed ... Application Control policy` (unsigned binaries). Per
  README's troubleshooting note: install the signed interpreter from python.org matching
  `.python-version` (Python 3.14), then:
  ```powershell
  uv venv --python <path to signed python.exe>
  uv sync
  ```
  Confirm with `uv run train.py --smoke-test`.
- **Close GPU-hungry apps** — games, video calls, ML notebooks, anything with heavy
  hardware acceleration. Check with `nvidia-smi` (or Task Manager → Performance → GPU):
  you want the card near-idle before the live run. Training peaks at ~6.6 GB VRAM.
- **Font size up** in both the terminal and the browser (Ctrl+= in the browser; terminal
  profile font settings). Verify from the back of the room, not arm's length.
- **Pre-launch chat.py, hidden:**
  ```powershell
  uv run chat.py --no-browser --port 8000
  ```
  in a terminal you then minimize, with the browser tab loaded and hidden behind your
  other windows. (If port 8000 is already in use, chat.py auto-shifts to the next free
  port and prints which one — point your browser tab at that port instead.) Two things
  to know about the pre-launched server:
  - It discovers checkpoints **at startup** — it will not see files archived later
    without a restart (only matters if the live run is a keep; see section 5).
  - Don't run generations in it *during* the live training run — generating loads the
    baseline/best models onto the GPU and they stay cached, competing with training for
    VRAM. Idle, it holds only the primary checkpoint (~19M params) and is harmless.
- **Disable notifications** — Windows Do Not Disturb (Win+N → Do not disturb, or
  Settings → System → Notifications), plus anything self-updating (Slack, mail, Teams).

---

## 5. After the demo

### Close out the live experiment per the loop protocol

If you completed [8–10 min] live, verify the state is consistent and you are done:

```powershell
git log --oneline -3          # top commit should be "log: matrix lr 0.05 ... <status>"
Get-Content results.tsv -Tail 2
git status                    # should be clean
```

If the segment was cut short or a fallback fired, finish the protocol now. Both paths,
exactly as `program.md` specifies (`$start`, `$exp`, `$ts` as captured in [2–3 min]; if
the shell was lost, recover them from `git log`):

**Discard path** (val_bpb equal or worse than the current best):

```powershell
git reset --hard $start
Add-Content results.tsv "$ts`t$exp`t<val_bpb>`t<memory_gb>`tdiscard`tmatrix lr 0.05 aug07-demo live"
git add results.tsv; git commit -m "log: matrix lr 0.05 aug07-demo discard"
```

*(If the training run never produced a score at all — crashed or was killed — log
`0.000000` and `0.0` with status `crash` per program.md, and still reset to `$start`.)*

**Keep path** (val_bpb improved on the current best): archive the checkpoint first —
filename timestamps use the compact, colon-free `YYYYMMDDTHHMMSS-ZZZZ` form (colons are
invalid in Windows filenames); the TSV timestamp column keeps its colons:

```powershell
$stamp = git log -1 --format=%cd --date=format:%Y%m%dT%H%M%S%z
Copy-Item checkpoint_pre_eval.pt "checkpoints/${stamp}_${exp}.pt"
git add "checkpoints/${stamp}_${exp}.pt"
Add-Content results.tsv "$ts`t$exp`t<val_bpb>`t<memory_gb>`tkeep`tmatrix lr 0.05 aug07-demo live"
git add results.tsv; git commit -m "log: matrix lr 0.05 aug07-demo keep"
```

After a keep, restart chat.py so the new checkpoint is discovered and the **Best** pane
picks it up.

### Refresh downstream artifacts

- Optionally re-run `analysis.ipynb` (**Run All** in VS Code) to refresh the progress
  plot with the demo row included.
- The hosted demo does not update itself — see **Give the class the link** below if you
  want today's row or checkpoint reflected there.

### Give the class the link

The segment's afterlife is one URL: **https://autoresearch-demo.fly.dev/** — public,
free, no login, the same Baseline-vs-Best comparison they just watched, plus the
progress chart and the vocabulary browser. Send it with the follow-up material so
students can keep poking at the models on their own machines; the site's banner already
links "how these were made" to `WALKTHROUGH.md`. Tell them the first load takes ~12
seconds while the idle machine wakes, and that it is a text-completion model, not a
chat assistant — feed it the opening of a story, not a question.

It is a snapshot, not a mirror of this repo: it serves whatever checkpoints were
committed when it was last built. If today's run produced a new best and you want
students prompting *that* model, the deployment source lives in `deploy/` (Dockerfile +
fly.toml, built from the repo root) and the redeploy procedure is in
`deploy/README.md`.
