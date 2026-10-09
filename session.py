"""
session.py: run an autoresearch session unattended, with Claude Code.

  uv run session.py tinystories own                  a 10-hour session of 10-minute runs
  uv run session.py folktales own --hours 6 --run-minutes 5
  uv run session.py tinystories own --check          every check; nothing is changed or started
  uv run session.py tinystories own --show-prompt    print the agent's instructions

It prepares the dataset and tokenizer, starts the session's record (`lab.py start`) unless the
session exists, and launches the agent headless (`claude -p`). If the agent's turn ends early it
relaunches it with --continue and a restart prompt: the record holds the state, so nothing is lost.
It starts no launch in the session's last 20 minutes, ends the agent 25 minutes after the session's
end, and stops at once on a sign-in failure or when three launches in a row end within two minutes.
Logs go to session-logs/<name>/.

For a whole night, sign in with a long-lived token first: run `claude setup-token` and set
CLAUDE_CODE_OAUTH_TOKEN to what it prints. The usual sign-in can expire after about two hours
unattended. The agent runs without asking permission for each command (--dangerously-skip-permissions),
so run it only in this project's folder.
"""

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import record

LAST_MINUTES = 20        # no launch in the session's last 20 minutes (program.md starts no run then)
GRACE_MINUTES = 25       # room for the last run to finish and be recorded
QUICK_SECONDS = 120      # a launch that ends sooner than this counts as a quick failure
POLL_SECONDS = 30
LOGS = Path("session-logs")
SIGN_IN_WORDS = ("authenticat", "oauth", "login", "log in", "401", "403", "credential", "unauthori", "expired")
LIMIT_WORDS = ("usage limit", "rate limit", "limit reached", "429")


def session_name(dataset, tokenizer, run_minutes, day):
    return f"{dataset}-{tokenizer}-{run_minutes}min-{day:%Y-%m-%d}"


def _ends(s):
    return _dt.datetime.fromisoformat(s.meta["ends"])


def prompts(s):
    """The first instructions, and the ones after a restart."""
    m = s.meta
    minutes = int(m["run_minutes"])
    end = f"{_ends(s):%H:%M on %A}"
    length_rule = length_note = ""
    if minutes != 10:
        length_rule = (
            f"\n6. Every run in this session trains for {minutes} minutes, on purpose: the person set "
            f"AUTORESEARCH_TIME_BUDGET={minutes * 60} for this whole session. Where program.md says a run trains for "
            f"10 minutes, read {minutes}, and its 20-minute timeout becomes {minutes + 10} minutes: kill a run only after "
            "that long. Do not change or unset the variable. Scores compare within this session only.")
        length_note = f" Every run trains for {minutes} minutes (rule 6)."
    first = (
        "Read program.md, do the setup checks, and start a new experiment loop. Record each decision with "
        "`uv run lab.py keep` or `uv run lab.py undo`; never edit results.tsv.\n\n"
        "You were started unattended: nobody can answer questions, so never stop to ask one. "
        f"Run tag: {s.name}. Session budget: {m['hours']:g} hours, ending at {end}. "
        "These session rules override program.md where they differ:\n"
        f"1. This folder holds the session's record in sessions/{s.name}/. Use `uv run lab.py` for status, history, "
        "diff, keep and undo; never use git.\n"
        f"2. The active pair is {m['dataset']} with the {m['tokenizer']} tokenizer. It is already prepared; do not "
        "change it. Scores compare only within this dataset.\n"
        "3. The session's first version is the recipe train.py held when the session started; the first three runs "
        "train it as it is, to measure the noise (program.md, The first runs).\n"
        "4. You run headless. If you end your turn, the session ends, and no background task, monitor or "
        "notification can wake you. So never end your turn while an experiment runs, and never start training in "
        "the background. Run it in the foreground: `uv run train.py > run.log 2>&1` with the Bash tool timeout at "
        f"its maximum (600000 ms). A run takes {minutes + 3} to {minutes + 6} minutes, so it may outlast that; if "
        "the command is moved to the background, wait for it in the foreground with a loop such as "
        "`until grep -qE \"^val_bpb|Traceback|Error\" run.log; do sleep 15; done` (timeout 600000), repeated until "
        "it returns, then read the result.\n"
        f"5. The session ends at {end}. Follow program.md's \"Ending the session\": when less than 20 minutes "
        "remain, start no new experiment, make sure the last run has a decision, write your summary, and stop."
        + length_rule)
    again = (
        "You were restarted after your last turn ended. First check whether an experiment was left running or "
        "unfinished: run `uv run lab.py status` and `uv run lab.py history`, and finish the bookkeeping "
        "(`lab.py keep` or `lab.py undo`) for any run that finished. Then continue the experiment loop, following "
        "program.md and the session rules you were given at the start, especially rule 4: run training in the "
        f"foreground and never end your turn while it runs. The session still ends at {end}.{length_note}")
    return first, again


def agent_env(s, base):
    """The agent's environment: the session's run length, set only when it is not the default."""
    env = dict(base)
    minutes = int(s.meta["run_minutes"])
    if minutes != 10:
        env["AUTORESEARCH_TIME_BUDGET"] = str(minutes * 60)
    else:
        env.pop("AUTORESEARCH_TIME_BUDGET", None)
    return env


def _last_result(path, start):
    """The last result line claude wrote to the log after byte `start`."""
    try:
        with open(path, "rb") as f:
            f.seek(start)
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            return obj
    return None


def run_agent(s, *, logs, launch, clock, sleep, kill, agent="claude"):
    """Launch the agent, and relaunch it until the session's time is up. Returns why it stopped."""
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    jsonl, err = logs / "agent.jsonl", logs / "agent.err"
    ends = _ends(s).timestamp()
    last_launch, hard_stop = ends - LAST_MINUTES * 60, ends + GRACE_MINUTES * 60
    first, again = prompts(s)
    env = agent_env(s, os.environ)
    n = quick = 0
    while clock() < last_launch:
        n += 1
        began = clock()
        cmd = [agent, "--dangerously-skip-permissions"] + ([] if n == 1 else ["--continue"]) + \
              ["-p", "--output-format", "stream-json", "--verbose"]
        start = jsonl.stat().st_size if jsonl.exists() else 0
        with open(jsonl, "a", encoding="utf-8") as out, open(err, "a", encoding="utf-8") as errors:
            proc = launch(cmd, first if n == 1 else again, env, out, errors)
            while proc.poll() is None:
                if clock() >= hard_stop:
                    kill(proc)
                    return f"hard stop: the agent was still running {GRACE_MINUTES} minutes after the session's end"
                sleep(POLL_SECONDS)
        took = clock() - began
        result = _last_result(jsonl, start) or {}
        text = str(result.get("result", "")).lower()
        _note(logs, f"launch {n} ended after {took:.0f} s: {text[:200]}")
        if result.get("is_error") and any(w in text for w in SIGN_IN_WORDS):
            return "sign-in failed: start again with a fresh token from `claude setup-token`"
        quick = quick + 1 if took < QUICK_SECONDS else 0
        if quick >= 3:
            return "three launches in a row ended within two minutes: see session-logs for why"
        pause = 900 if any(w in text for w in LIMIT_WORDS) else POLL_SECONDS
        if last_launch - clock() > pause:
            sleep(pause)
    return "time is up: no launch in the session's last 20 minutes"


def _note(logs, message):
    line = f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line)
    with open(Path(logs) / "runner.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _launch(cmd, prompt, env, stdout, stderr):
    """Start claude with the prompt on stdin: on Windows `claude` is a .cmd file, and cmd.exe would
    break a prompt of several lines passed as an argument."""
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                            text=True, encoding="utf-8")
    proc.stdin.write(prompt)
    proc.stdin.close()
    return proc


def _kill(proc):
    """End the agent and everything it started (training included)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


def _signin(agent):
    """One short test reply, the same way the session will talk to the agent."""
    try:
        done = subprocess.run([agent, "-p", "--output-format", "json"], input="Reply with the single word OK.",
                              capture_output=True, text=True, encoding="utf-8", timeout=180)
        reply = json.loads(done.stdout or "{}")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return f"error: {exc}"[:160]
    return "ok" if reply.get("is_error") is False else "error: " + str(reply.get("result", "no reply"))[:120]


def _gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def checks(*, run_minutes, hours, env, which, signin, disk_free_gb, locked, gpu_used_mib=lambda: None):
    """(passed, message) for each check; a message starting 'note:' does not stop the session."""
    out = [(1 <= run_minutes <= 60, f"each run trains for {run_minutes} min (1 to 60)"),
           (1 <= hours <= 48, f"the session lasts {hours:g} h (1 to 48)")]
    out.append((True, "CLAUDE_CODE_OAUTH_TOKEN is set") if env.get("CLAUDE_CODE_OAUTH_TOKEN") else
               (True, "note: CLAUDE_CODE_OAUTH_TOKEN is not set; the usual sign-in can expire after about two hours "
                      "unattended (run `claude setup-token`)"))
    out.append((not env.get("AUTORESEARCH_TIME_BUDGET"),
                "AUTORESEARCH_TIME_BUDGET is not set in this terminal" if not env.get("AUTORESEARCH_TIME_BUDGET")
                else "AUTORESEARCH_TIME_BUDGET is set in this terminal: unset it and use --run-minutes instead"))
    claude = which("claude")
    out.append((bool(claude), f"claude found: {claude}" if claude else "claude is not on the PATH (install Claude Code)"))
    out.append((bool(which("uv")), "uv found" if which("uv") else "uv is not on the PATH"))
    out.append((not locked(), "no other session runner" if not locked() else
                f"another session runner is running ({LOGS / 'runner.lock'}); if none is, delete that folder"))
    free = disk_free_gb()
    out.append((free >= 10, f"disk free: {free} GB (need 10)"))
    used = gpu_used_mib()
    out.append((True, "note: GPU memory in use unknown (no nvidia-smi)") if used is None else
               (used < 2048, f"GPU memory in use: {used} MiB" + ("" if used < 2048 else " (close other GPU work first)")))
    if claude:
        reply = signin()
        out.append((reply == "ok", "sign-in works (one test reply)" if reply == "ok" else f"sign-in test failed: {reply}"))
    return out


def _run(cmd):
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(prog="session.py", description="Run an autoresearch session unattended.")
    parser.add_argument("dataset")
    parser.add_argument("tokenizer", nargs="?", default="own")
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--run-minutes", type=int, default=10)
    parser.add_argument("--name", help="the session's name (default: <dataset>-<tokenizer>-<minutes>min-<date>)")
    parser.add_argument("--check", action="store_true", help="run every check, change nothing")
    parser.add_argument("--show-prompt", action="store_true", help="print the agent's instructions, change nothing")
    args = parser.parse_args(argv)
    name = args.name or session_name(args.dataset, args.tokenizer, args.run_minutes, _dt.date.today())
    root = record.SESSIONS / name

    if args.show_prompt:
        s = record.Session(root) if (root / "session.json").exists() else types.SimpleNamespace(name=name, meta={
            "dataset": args.dataset, "tokenizer": args.tokenizer, "run_minutes": args.run_minutes,
            "hours": args.hours, "ends": (_dt.datetime.now().astimezone()
                                          + _dt.timedelta(hours=args.hours)).isoformat(timespec="seconds")})
        first, again = prompts(s)
        print(first + "\n\n--- after a restart ---\n" + again)
        return 0

    agent = shutil.which("claude")
    results = checks(run_minutes=args.run_minutes, hours=args.hours, env=os.environ, which=shutil.which,
                     signin=lambda: _signin(agent), disk_free_gb=lambda: shutil.disk_usage(".").free // 2**30,
                     locked=lambda: (LOGS / "runner.lock").exists(), gpu_used_mib=_gpu_used_mib)
    print(f"Checks for session {name}: {args.hours:g} h of {args.run_minutes}-minute runs on "
          f"{args.dataset} / {args.tokenizer}")
    for passed, message in results:
        print(("  ok    " if passed else "  FAIL  ") + message)
    if not all(passed for passed, _ in results):
        print("Not ready.")
        return 1
    if args.check:
        print("Ready. Nothing was changed.")
        return 0

    LOGS.mkdir(exist_ok=True)
    lock = LOGS / "runner.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        print(f"another session runner is running ({lock})")
        return 1
    try:
        if _run(["uv", "run", "prepare.py", "--dataset", args.dataset, "--tokenizer", args.tokenizer]) != 0:
            print("prepare.py failed: not starting")
            return 1
        if (root / "session.json").exists():
            s = record.Session(root)
            same = (s.meta["dataset"], s.meta["tokenizer"], int(s.meta["run_minutes"])) == \
                   (args.dataset, args.tokenizer, args.run_minutes)
            if not same:
                print(f"session {name} exists with another dataset, tokenizer or run length: choose another --name")
                return 1
            record.use_session(name)
            print(f"continuing session {name}, which ends {_ends(s):%H:%M on %A}")
        elif _run(["uv", "run", "lab.py", "start", name, "--dataset", args.dataset, "--tokenizer", args.tokenizer,
                   "--run-minutes", str(args.run_minutes), "--hours", f"{args.hours:g}"]) != 0:
            print("lab.py start failed: not starting")
            return 1
        s = record.active_session()
        reason = run_agent(s, logs=LOGS / name, launch=_launch, clock=time.time, sleep=time.sleep, kill=_kill,
                           agent=agent)
        _note(LOGS / name, f"stopped: {reason}")
        _run(["uv", "run", "lab.py", "status"])
        _run(["uv", "run", "lab.py", "check"])
        return 0
    finally:
        lock.rmdir()


if __name__ == "__main__":
    sys.exit(main())
