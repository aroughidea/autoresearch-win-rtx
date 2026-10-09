"""
lab.py: the experiment record's command line. The agent uses it instead of git.

  uv run lab.py status                 the best version, whether train.py differs from it, time left
  uv run lab.py history                one line per run; * marks the best
  uv run lab.py diff [A] [B]           what changed (default: the best version against train.py)
  uv run lab.py show <id>              print one version of train.py
  uv run lab.py keep "<description>"   the last run's version becomes the best
  uv run lab.py undo "<description>"   record the last run as discarded; reset train.py to the best version
  uv run lab.py export [--sqlite F]    rewrite results.tsv; optionally build a SQLite file
  uv run lab.py check                  report anything inconsistent in the record
  uv run lab.py restore                put back a fixed file (program.md, prepare.py, ...) that changed
  uv run lab.py end                    end the session: its folder stays; train.py then runs outside any session

A person, or the session runner, starts a session:
  uv run lab.py start <name> --dataset tinystories --tokenizer own [--run-minutes 10] [--hours 10]

The agent must not modify this file (see program.md).
"""

import argparse
import datetime as _dt
import sys
from pathlib import Path

import record


def _session():
    session = record.active_session()
    if session is None:
        raise record.RecordError("lab: no active session; start one with "
                                 "`uv run lab.py start <name> --dataset <d> --tokenizer <t>`")
    return session


def _score(value):
    return f"{value:.6f}" if value is not None else "none yet"


def cmd_start(args):
    s = record.start_session(args.name, dataset=args.dataset, tokenizer=args.tokenizer,
                             run_minutes=args.run_minutes, hours=args.hours)
    print(f"started {s.name}: {s.meta['dataset']}/{s.meta['tokenizer']}, {s.meta['run_minutes']}-minute runs, "
          f"ends {s.meta['ends']}; first version {record.read_best(s)['version']}")


def cmd_status(args):
    s = _session()
    best = record.read_best(s)
    working = record.version_id(Path("train.py").read_text(encoding="utf-8"))
    ends = _dt.datetime.fromisoformat(s.meta["ends"])
    left = max(0, int((ends - _dt.datetime.now().astimezone()).total_seconds() // 60))
    print(f"session:   {s.name} ({s.meta['dataset']}/{s.meta['tokenizer']}, {s.meta['run_minutes']}-minute runs)")
    print(f"best:      {best['version']}, val_bpb {_score(best['val_bpb'])}")
    n = record.noise(s)
    print("noise:     " + (f"{n['runs']} baseline runs, {n['low']:.6f} to {n['high']:.6f} (spread {n['spread']:.6f})"
                           if n else "not measured yet (train the baseline three times, then keep)"))
    print("train.py:  " + ("same as the best version" if working == best["version"]
                           else f"differs from the best version (it would be version {working})"))
    waiting = record.undecided(s)
    if waiting:
        print(f"waiting:   {len(waiting)} finished run(s) need `lab.py keep` or `lab.py undo`")
    if record.unfinished(s) is not None:
        print("running:   a run has started and not finished (still training, or crashed)")
    print(f"time left: {left} min (the session ends {ends:%H:%M})")


def cmd_history(args):
    s = _session()
    rows = record.history(s)
    width = max([len("run")] + [len(r["run"]) for r in rows])
    print(f"  {'run':<{width}} {'version':<13} {'parent':<13} {'val_bpb':<9} {'status':<8} description")
    for r in rows:
        mark = "*" if r["best"] else " "
        score = f"{r['val_bpb']:.6f}" if r["val_bpb"] is not None else "-"
        print(f"{mark} {r['run']:<{width}} {r['version']:<13} {r['parent']:<13} {score:<9} "
              f"{r['status'] or 'waiting':<8} {r['description']}")


def cmd_diff(args):
    print(record.diff(_session(), args.a, args.b), end="")


def cmd_show(args):
    print(record.read_version(_session(), args.id), end="")


def _decided(result):
    best = result["best"]
    return f"best is version {best['version']}, val_bpb {_score(best['val_bpb'])}"


def cmd_keep(args):
    result = record.decide(_session(), True, args.description)
    print(f"kept {len(result['runs'])} run(s); {_decided(result)}")
    if result["checkpoint"] is None:
        print("model not archived: checkpoint_pre_eval.pt was not saved by the kept run")


def cmd_undo(args):
    result = record.decide(_session(), False, args.description)
    if result["status"] == "reset":
        print(f"nothing to decide; train.py is back to the best version; {_decided(result)}")
    else:
        print(f"recorded {result['status']} for {len(result['runs'])} run(s); train.py is back to the best "
              f"version; {_decided(result)}")


def cmd_export(args):
    s = _session()
    record.write_results_tsv(s)
    print(f"wrote {s.results_path}")
    if args.sqlite:
        print(f"wrote {record.export_sqlite(s, args.sqlite)}")


def cmd_check(args):
    problems = record.check(_session())
    if problems:
        for p in problems:
            print(f"problem: {p}")
        raise record.RecordError(f"lab: {len(problems)} problem(s) in the record")
    print("the record is consistent")


def cmd_restore(args):
    restored = record.restore_fixed_files(_session())
    print("restored " + ", ".join(restored) + " from the session's copy" if restored
          else "nothing to restore: the fixed files are as the session started")


def cmd_end(args):
    name = record.end_session()
    print(f"ended {name}: its folder stays in sessions/{name}/; train.py now runs outside any session"
          if name else "no session is active")


def main(argv=None):
    # Piped output (how the agent reads it) defaults to cp1252 on Windows, which cannot print
    # every description; UTF-8 can.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass
    parser = argparse.ArgumentParser(prog="lab.py", description="The experiment record.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start", help="start a session")
    p.add_argument("name")
    p.add_argument("--dataset", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--run-minutes", type=int, default=10)
    p.add_argument("--hours", type=float, default=10)
    p.set_defaults(func=cmd_start)
    for name, func in (("status", cmd_status), ("history", cmd_history), ("check", cmd_check),
                       ("restore", cmd_restore), ("end", cmd_end)):
        sub.add_parser(name).set_defaults(func=func)
    p = sub.add_parser("diff")
    p.add_argument("a", nargs="?")
    p.add_argument("b", nargs="?")
    p.set_defaults(func=cmd_diff)
    p = sub.add_parser("show")
    p.add_argument("id")
    p.set_defaults(func=cmd_show)
    for name, func in (("keep", cmd_keep), ("undo", cmd_undo)):
        p = sub.add_parser(name)
        p.add_argument("description")
        p.set_defaults(func=func)
    p = sub.add_parser("export")
    p.add_argument("--sqlite")
    p.set_defaults(func=cmd_export)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except record.RecordError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
