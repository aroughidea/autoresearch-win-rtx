"""library/make_baselines_b.sh: what a dry run plans, for the 10-minute set and the 5-minute pair."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
# On Windows `bash` may be WSL's, which has neither this checkout's tools nor its paths: use Git Bash there.
BASH = str(GIT_BASH) if sys.platform == "win32" and GIT_BASH.exists() else (None if sys.platform == "win32" else shutil.which("bash"))
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash (Git Bash on Windows)")
STARTER, RECIPE_B = "91ffb4c37015", "c9ba08e4f442"


def dry_run(tmp_path, **env):
    """DRY_RUN=1 train, with a `uv` on the PATH that always fails: a dry run must not need it."""
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "uv").write_text("#!/bin/sh\necho 'uv was called' >&2\nexit 1\n", encoding="utf-8", newline="\n")
    (stub / "uv").chmod(0o755)
    full = dict(os.environ, DRY_RUN="1", **env)
    full["PATH"] = str(stub) + os.pathsep + full["PATH"]
    return subprocess.run([BASH, "library/make_baselines_b.sh", "train"], cwd=ROOT, env=full,
                          capture_output=True, text=True, encoding="utf-8")


def planned(out):
    """(name, recipe version, destination) for each run the dry run lists."""
    lines = out.splitlines()
    heads = [l for l in lines if l.startswith("== ") and "recipe" in l]
    dests = [l.split("-> ")[1].split(",")[0] for l in lines if l.strip().startswith("-> ")]
    return [(h.split()[1], h.split("recipe ")[1].split(")")[0], d) for h, d in zip(heads, dests)]


def test_the_10_minute_set_is_a_speed_check_then_six_runs(tmp_path):
    done = dry_run(tmp_path)
    assert done.returncode == 0, done.stderr
    assert "uv was called" not in done.stderr
    runs = planned(done.stdout)
    assert runs[0] == ("b10-speed-check-a", STARTER, "library/logs/b10-speed-check-a.json")
    assert [r[2] for r in runs[1:]] == [f"library/baselines-b-10min/runs/{p}.json" for p in (
        "tinystories-own", "tinystories-phi3", "tinystories-gpt2", "folktales-own", "folktales-phi3", "folktales-gpt2")]
    assert all(r[1] == RECIPE_B for r in runs[1:])


def test_the_5_minute_pair_is_two_runs_into_its_own_collection(tmp_path):
    done = dry_run(tmp_path, MINUTES="5", PAIRS="tinystories:own folktales:own")
    assert done.returncode == 0, done.stderr
    assert "uv was called" not in done.stderr
    assert planned(done.stdout) == [
        ("tinystories-own", RECIPE_B, "library/baselines-b-5min/runs/tinystories-own.json"),
        ("folktales-own", RECIPE_B, "library/baselines-b-5min/runs/folktales-own.json"),
    ]
    assert "5 min" in done.stdout


def test_other_run_lengths_are_refused(tmp_path):
    done = dry_run(tmp_path, MINUTES="7")
    assert done.returncode == 2
    assert "MINUTES" in done.stdout + done.stderr


def test_the_5_minute_tinystories_floor_is_half_the_10_minute_run_less_5_percent(tmp_path):
    ten = tmp_path / "tinystories-own.json"
    ten.write_text('{"final": {"num_steps": 3150, "val_bpb": 0.4361}}', encoding="utf-8")
    done = dry_run(tmp_path, MINUTES="5", PAIRS="tinystories:own folktales:own", B10_TINYSTORIES=str(ten))
    assert done.returncode == 0, done.stderr
    floors = [l.split("speed floor ")[1] for l in done.stdout.splitlines() if "speed floor" in l]
    assert floors == ["1496", "-"]          # 3,150 x 0.475 = 1,496.25; Folktales has no known speed
