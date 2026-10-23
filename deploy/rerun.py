"""Re-run a kept version of a featured session, saving its model at moments of training.

LLM Explorables' model demo shows one run's model at 10 seconds, 1 minute and 5 minutes, and at
the end, which is the session's own kept model (its contract's "Within a run"). A session saves only the end, so this trains the version again on
the training machine, outside any session, with capture.py's moment switch on:

    python deploy/rerun.py <run id>

<run id> is one of deploy/sessions/<session>/runs/'s. It runs that version's own code (versions/
<version>.py) on the run's dataset for the run's time budget, then files what it made: the re-run's
run file in deploy/sessions/<session>/reruns/, and its three models (<re-run id>@10s, @60s and
@300s.pt) in deploy/checkpoints/. chat.py serves them as <run id>@10s and so on; the end is the
session's own kept model, so the re-run's is not filed. Run it from the
repo root of a checkout with no session (no sessions/active.txt): the run writes checkpoint_pre_eval.pt
there as any run does.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"
MOMENTS = (10, 60, 300)
# Runs a version file as train.py would be run: its folder is not on the path, the repo root is.
LAUNCH = ("import runpy, sys; version, dataset = sys.argv[1:3]; "
          "sys.argv = [version, '--dataset', dataset]; runpy.run_path(version, run_name='__main__')")


def find_run(run_id: str) -> tuple[Path, dict]:
    """The session folder in deploy/sessions/ holding this run, and its run file."""
    found = sorted(DEPLOY.glob(f"sessions/*/runs/{run_id}.json"))
    if not found:
        raise SystemExit(f"no run {run_id} in deploy/sessions/*/runs/")
    path = found[0]
    return path.parent.parent, json.loads(path.read_text(encoding="utf-8"))


def plan(run_id: str, work: Path) -> dict:
    """What to run: the version file, the dataset, and the environment that turns the switch on."""
    session, run = find_run(run_id)
    version = session / "versions" / f"{run['version']}.py"
    if not version.is_file():
        raise SystemExit(f"no version code for {run_id}: {version} is missing")
    env = {**os.environ,
           "AUTORESEARCH_RUNS_DIR": str(work),
           "AR_SAVE_MOMENTS": ",".join(str(m) for m in MOMENTS),
           "AR_RERUN_OF": run_id,
           "AUTORESEARCH_TIME_BUDGET": str(int(run["time_budget_s"])),
           # The run's own pair, whatever this machine has active (active_dataset.txt, active_tokenizer.txt).
           "AUTORESEARCH_DATASET": run["dataset"],
           "AUTORESEARCH_TOKENIZER": (run.get("tokenizer") or {}).get("name") or "own"}
    return {"session": session, "run": run, "version": version, "dataset": run["dataset"], "env": env,
            "command": [sys.executable, "-c", LAUNCH, str(version), run["dataset"]]}


def file_rerun(work: Path, session: Path, run: dict, checkpoints: Path) -> dict:
    """Move the re-run's run file to <session>/reruns/ and its three models to checkpoints/. Returns the run file."""
    made = sorted(work.glob("*.json"))
    if len(made) != 1:
        raise SystemExit(f"expected one run file in {work}, found {len(made)}")
    rerun = json.loads(made[0].read_text(encoding="utf-8"))
    expected = list(MOMENTS)
    got = [m["at_s"] for m in rerun.get("moments") or []]
    if (rerun.get("tokenizer") or {}).get("name") != (run.get("tokenizer") or {}).get("name", "own"):
        raise SystemExit(f"the re-run trained with the {(rerun.get('tokenizer') or {}).get('name')} vocabulary, "
                         f"not the run's; nothing was filed")
    if rerun.get("rerun_of") != run["run_id"] or got != expected:
        raise SystemExit(f"the re-run saved moments {got} (expected {expected}); "
                         f"{rerun.get('moments_error') or 'see the training output above'}. Nothing was filed.")
    checkpoints.mkdir(parents=True, exist_ok=True)
    for moment in rerun["moments"]:
        shutil.move(str(work / moment["checkpoint"]), checkpoints / moment["checkpoint"])
    (session / "reruns").mkdir(exist_ok=True)
    shutil.move(str(made[0]), session / "reruns" / made[0].name)
    return rerun


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_id", help="a run in deploy/sessions/*/runs/, e.g. 20261022T052816-0700_6c874e790ab4")
    args = parser.parse_args(argv)
    if Path("sessions/active.txt").exists():
        raise SystemExit("a session is active in this folder (sessions/active.txt): run this from a checkout with no session")
    work = DEPLOY / ".rerun-work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    p = plan(args.run_id, work)
    print(f"Re-running {args.run_id} ({p['dataset']}, version {p['run']['version']}, "
          f"{p['run']['time_budget_s']} s), saving its model at {', '.join(f'{m} s' for m in MOMENTS)}.")
    result = subprocess.run(p["command"], env=p["env"], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"the run ended with exit code {result.returncode}; nothing was filed")
    rerun = file_rerun(work, p["session"], p["run"], DEPLOY / "checkpoints")
    shutil.rmtree(work, ignore_errors=True)
    final, original = rerun.get("final") or {}, p["run"].get("final") or {}
    print(f"\nRe-run {rerun['run_id']}: {final.get('num_steps')} steps, val_bpb {final.get('val_bpb')} "
          f"(the session's run: {original.get('num_steps')} steps, {original.get('val_bpb')}).")
    print(f"Filed {p['session'].relative_to(ROOT).as_posix()}/reruns/{rerun['run_id']}.json and "
          + ", ".join(f"deploy/checkpoints/{m['checkpoint']}" for m in rerun["moments"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
