import re
import subprocess
from datetime import datetime, timedelta, timezone

import capture


PDT = timezone(timedelta(hours=-7))


def test_compact_timestamp_negative_offset():
    moment = datetime(2026, 5, 23, 15, 57, 43, tzinfo=PDT)
    assert capture.compact_timestamp(moment) == "20260523T155743-0700"


def test_compact_timestamp_positive_half_hour_offset():
    moment = datetime(2026, 5, 23, 15, 57, 43, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert capture.compact_timestamp(moment) == "20260523T155743+0530"


def test_compact_timestamp_naive_gets_local_offset():
    stamp = capture.compact_timestamp(datetime(2026, 5, 23, 15, 57, 43))
    assert re.fullmatch(r"\d{8}T\d{6}[+-]\d{4}", stamp)


def test_make_run_id_with_commit():
    moment = datetime(2026, 5, 23, 15, 57, 43, tzinfo=PDT)
    assert capture.make_run_id("75027e8", moment) == "20260523T155743-0700_75027e8"


def test_make_run_id_without_git_is_colon_free():
    run_id = capture.make_run_id(None, datetime(2026, 5, 23, 15, 57, 43, tzinfo=PDT))
    assert run_id.endswith("_nogit")
    assert ":" not in run_id


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_git_commit_info_in_a_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    expected = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()

    commit, committed_at = capture.git_commit_info(cwd=tmp_path)

    assert commit == expected
    assert committed_at.tzinfo is not None


def test_git_commit_info_outside_a_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert capture.git_commit_info(cwd=tmp_path) == (None, None)
