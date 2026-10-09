"""Repository files that the experiment record depends on."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _ignored(path):
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT).returncode == 0


def test_the_record_is_ignored_but_the_library_sessions_are_not():
    assert _ignored("sessions/s1/runs/r.json")
    assert not _ignored("library/sessions/s1/runs/r.json")
    assert _ignored("checkpoint_pre_eval.pt")


def test_library_scripts_do_not_check_out_the_untracked_weights_file():
    for name in ("make_study.sh", "make_baselines.sh"):
        for line in (ROOT / "library" / name).read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("git checkout"):
                assert "checkpoint_pre_eval.pt" not in line, (name, line)
