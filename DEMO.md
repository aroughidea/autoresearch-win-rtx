# DEMO.md — Instructor's Runbook

A 15-minute lecture segment: one turn of the autoresearch loop, run by hand and narrated end
to end. The live turn runs in a small **demo session** of 3-minute runs, so it fits the slot;
the paper trail comes from a real **overnight session** of 10-minute runs; and the reveal is the
hosted demo, <https://autoresearch-demo.fly.dev/>. Everything here follows `README.md`,
`program.md` and the session map, [`docs/session-map.html`](docs/session-map.html).

**The one number to know walking in: the demo session's best score.** Check it with
`uv run lab.py status` (section 1c).

**One rule to respect throughout: one session is active at a time.** Training records into
the active session (`sessions/active.txt`). Never start or switch sessions while another
session's agent is running, or its next run lands in the wrong record.

---

## 1. Before the lecture

### 1a. The demo session's baseline (about 20 minutes, before anything else)

Train the demo session's baseline first, while nothing else is running. In one PowerShell
terminal at the repo root:

```powershell
$env:AUTORESEARCH_TIME_BUDGET = 180          # 3-minute runs, for this session only
uv run lab.py start demo-<date> --dataset tinystories --tokenizer own --run-minutes 3 --hours 48
uv run train.py > run.log 2>&1               # three times: the baseline, unchanged
uv run train.py > run.log 2>&1
uv run train.py > run.log 2>&1
uv run lab.py keep "baseline, three runs"
uv run lab.py status                          # the demo session's best and its noise
Remove-Item Env:AUTORESEARCH_TIME_BUDGET      # back to the 10-minute default in this terminal
```

Each run takes about 6 minutes: 3 of training, then the score and the sample stories. The
demo session starts from whatever `train.py` holds; its scores compare only with each other,
never with 10-minute runs.

**Choose the live change now, and don't try it:** one line in `train.py` you can explain in
a sentence, such as `MATRIX_LR` up or down by a small step, or `WARMDOWN_RATIO`. Keep or
undo, either outcome teaches. Write it here: ______________________

### 1b. The overnight session (for the paper trail; skip if a recent one exists)

Once the demo baseline is kept, start a real session: 10 hours of 10-minute runs.

```powershell
uv run session.py tinystories own --check     # every check, changing nothing
uv run session.py tinystories own
```

`session.py` makes it the active session and starts the agent unattended (README, **Running
the agent**). It starts from whatever `train.py` holds; to start from the starter kit's recipe
instead, put that recipe in `train.py` first. Leave the machine alone until it finishes.

### 1c. Morning checklist

Run through all seven, in order:

- [ ] **The overnight session finished cleanly.** While it is still active:
  ```powershell
  uv run lab.py status       # its best, its noise, nothing waiting
  uv run lab.py check        # "the record is consistent"
  uv run lab.py history      # pick 2 or 3 runs to show: a kept change, an undone one
  ```
  Note its folder name, `sessions/<night>/`, and the run ids you will open.

- [ ] **Switch back to the demo session** and note its best:
  ```powershell
  uv run lab.py use demo-<date>
  uv run lab.py status       # the number to know walking in
  ```

- [ ] **Smoke-test the training path** (~1 minute; this also catches Smart App Control
      refusing torch's own files, which happened for a few hours on 9 October 2026):
  ```powershell
  uv run train.py --smoke-test
  ```

- [ ] **Wake the hosted demo and leave the tab open.** Load
      <https://autoresearch-demo.fly.dev/>. It is `chat.py` on one small shared machine that
      **suspends when idle and takes ~12 seconds to wake**; after that pages load at once.
      Confirm the chart and both panes render, then leave the tab open. Reload it during
      room setup (section 4).

- [ ] **Pick 2–3 prompts and test them there.** These models write short children's stories,
      unevenly. Try "Once upon a time", "The little dog", "One day, a little girl named
      Lily", two or three times each, with the temperature slider where you will set it live
      (about 0.9 reads better than the default 0). Generation there is capped at 500 tokens.
      Write your winners here: ______________________

- [ ] **Open the session map** from the laptop: `docs/session-map.html` in the browser. It
      works offline (its fonts fall back if the network is down).

- [ ] **Capture the fallback screenshots** — see section 3.

---

## 2. The segment (~16 min, timed)

**Starting screen state:** one PowerShell terminal at the repo root (font enlarged), with the
demo session active and `$env:AUTORESEARCH_TIME_BUDGET = 180` set in it (without it the live run
would be a 10-minute run, and the record refuses it in a 3-minute session); the browser with two tabs, the session map and the hosted demo (warm,
hidden); VS Code with `train.py`, `program.md` and the folder `sessions/<night>/` open.

### [0–2 min] The pitch: the session map

Put the map on screen and walk it top to bottom:

> "A coding agent changes one thing in the training recipe, trains a brand-new model for ten
> minutes, scores it, and keeps or undoes the change. Then again, for ten hours, alone.
> Every box says what it saves, and nothing is thrown away. You are about to watch one turn
> of this loop, by hand and faster: three minutes instead of ten."

Show `program.md` for five seconds: *"The agent reads its instructions from this file, and they
don't change during the session."*

### [2–3 min] START the live turn

```powershell
$env:AUTORESEARCH_TIME_BUDGET   # must print 180 (set it in room setup, section 4)
uv run lab.py status          # the best so far, and the noise
```

In VS Code, make the one-line change you chose in 1a and save. Then:

```powershell
uv run train.py > run.log 2>&1
```

Announce: "About six minutes: three of training, then the score and a few sample stories.
While it trains, let's look at what the agent did last night."

### [3–10 min] While it trains: last night's record

Four things, about a minute and a half each, all in `sessions/<night>/`:

1. **The scoreboard, `results.tsv`.** One row per run: the version, the score, keep or
   discard (or crash), what was tried. The first three rows are the same recipe, trained
   three times: their spread is the noise, how far apart two runs of the same recipe land.
   Most rows say `discard`; that is the loop working, not failing.
2. **One run's entry, `runs/<run>.json`.** Scroll to `snapshots`: what this model wrote at 0
   seconds (random fragments), 10, 30, 60, 120 seconds, 5 minutes, and the end. One model,
   learning.
3. **Two versions side by side.** In VS Code, select two files in `versions/` and compare
   them: the one line the agent changed between a version and the next.
4. **`best.json`, the pointer.** Which version is best right now. Keeping a change moves it;
   undoing one leaves it where it is.

Optional filler: `Get-Content run.log -Tail 3` shows the live run's progress line.

### [10–12 min] Read out the result, make the call

When the run finishes:

```powershell
Select-String '^val_bpb:' run.log
```

Read it aloud next to the best from `lab.py status`. Make the call live: *"Is it lower? Yes:
keep."* or *"No: undo."*

```powershell
uv run lab.py keep "matrix lr 0.05 (live)"     # if it is lower
uv run lab.py undo "matrix lr 0.05 (live)"     # if it is not
uv run lab.py history                          # the new row
```

The line to land: *"Nothing is lost. The version we just tried stays in the record, kept or
not, and `train.py` is back to the best version."* If the gain is smaller than the noise,
say so: the rule keeps it anyway, and the description should say it was small.

### [12–15 min] The reveal: the hosted demo

**Pending: which models the hosted demo serves.** It still serves the May session's two
models, from before the record; the choice waits for a few build sessions (`docs/TODO.md`).
Until the demo is redeployed, this beat cannot run as written. The plan, in two beats:

1. **The agent's night.** A build session's baseline and best, side by side on one of your
   tested prompts, with the temperature you tested. Let both finish without talking over
   them. Expect them to read almost alike: the agent's gains are usually smaller than you can
   read, which is why it judges by a score.
2. **A decision you can read.** Switch one pane to a Folktales model: the same recipe and
   vocabulary, a different dataset. The writing changes plainly. That is the point of the
   whole project: changing a design decision changes the final experience in a specific way
   (TRAINING-DECISIONS.md).

If there is time, click **Tokens** on one pane to show the raw tokenization.

### [15–16 min] Close

> "Everything you saw is in one public repo, and a starter kit you can copy tonight."

The starter kit, <https://github.com/aroughidea/autoresearch-starter>: "Use this template",
then its quickstart (`uv sync`, `uv run prepare.py`, `uv run train.py`), then
`uv run session.py tinystories own` for a night. Requirements: an NVIDIA GPU meeting the VRAM
floor, uv, and a Claude Code account; or a GPU rented by the hour (its HARDWARE.md). The
hosted demo link is the take-away.

---

## 3. Fallbacks — decide by T-2 min before the segment

Make the call **two minutes before you start**, not mid-segment.

**Tier 1 — the live run fails or overruns** (a crash, the GPU busy, or no score by minute
10): stop it with Ctrl+C, record it with `uv run lab.py undo "matrix lr 0.05 (live, stopped)"`,
and say so: a stopped run is recorded as a crash, and the record stays consistent. Spend the
time on last night's record instead, then do the reveal as planned. You lose the live gamble,
not the story.

**Tier 2 — no network, or the hosted demo is down:** do the reveal locally with the same
models (pending with the reveal, `docs/TODO.md`). `chat.py` shows the active session, so make
the build session active first, then start the page:

```powershell
uv run lab.py use <night>
uv run chat.py
```

It shows that session's chart and its kept models. After the lecture,
`uv run lab.py use demo-<date>` makes the demo session active again.

**Tier 3 — no `chat.py` either:** the reveal in the terminal, baseline then best on the same
prompt:

```powershell
uv run generate.py "<your tested prompt>" --checkpoint checkpoints/<the baseline's run>.pt
uv run generate.py "<your tested prompt>" --checkpoint checkpoints/<the best run>.pt
```

**Tier 4 — total machine failure:** present from screenshots. Capture these after the 1c
checklist passes, full-window, at presentation font size:

1. The session map, top of the page with the whole diagram.
2. `uv run lab.py status` and `uv run lab.py history` for the demo session.
3. Last night's `results.tsv` open in VS Code.
4. One run's entry with its `snapshots` showing.
5. Two versions compared side by side in VS Code.
6. The tail of `run.log` with the `---` summary block (`val_bpb:` visible).
7. The hosted demo with both panes filled from a tested prompt, and one pane in Tokens view.

---

## 4. Demo machine prep

- **SAC-safe Python.** If Windows Smart App Control is on, uv's standalone Python fails with
  `DLL load failed ... Application Control policy`. Per README's troubleshooting note: install
  the signed interpreter from python.org matching `.python-version`, then
  `uv venv --python <path to signed python.exe>` and `uv sync`. The morning smoke test (1c)
  confirms training works.
- **Close GPU-hungry apps** — games, video calls, notebooks, browsers with heavy hardware
  acceleration. Check with `nvidia-smi`: you want the card near-idle before the live run.
- **Font size up** in the terminal, the browser and VS Code. Check from the back of the room.
- **Set the demo's run length** in the presenting terminal: `$env:AUTORESEARCH_TIME_BUDGET = 180`.
- **Reload the hosted demo tab** during room setup, so it is warm.
- **Disable notifications** — Windows Do Not Disturb, plus anything self-updating.

---

## 5. After the demo

### Close out the live turn

```powershell
uv run lab.py status       # nothing waiting for a decision; train.py the same as the best version
uv run lab.py check        # "the record is consistent"
```

If the segment was cut short before the call, make it now: `uv run lab.py keep "..."` or
`uv run lab.py undo "..."` (a run that never finished is recorded as a crash). The demo session
can stay for the next lecture; `uv run lab.py end` ends it, and its folder stays either way.

### Give the class the links

- **The hosted demo**, <https://autoresearch-demo.fly.dev/>: public, free, no login, the same
  Baseline-and-Best comparison they just watched, plus the chart and the vocabulary browser.
  The first load takes ~12 seconds while the machine wakes, and it is a text-completion model,
  not a chat assistant: feed it the opening of a story, not a question.
- **The starter kit**, <https://github.com/aroughidea/autoresearch-starter>.

The hosted demo is a snapshot: it serves whatever models it was last built with (still the
May session's, until the step in `docs/TODO.md`). The deployment source is in `deploy/` and the
procedure in `deploy/README.md`.
