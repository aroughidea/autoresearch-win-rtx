"""program.md's loop uses the record, not git."""
from pathlib import Path

TEXT = Path(__file__).resolve().parent.parent.joinpath("program.md").read_text(encoding="utf-8")
LOOP = TEXT.split("## The experiment loop", 1)[1].split("\n## ", 1)[0]


def test_the_loop_uses_lab_not_git():
    assert "git commit" not in LOOP and "git reset" not in LOOP and "git add" not in LOOP
    for command in ("uv run lab.py status", "uv run lab.py history", "uv run lab.py keep", "uv run lab.py undo"):
        assert command in LOOP


def test_record_and_lab_are_read_only_to_the_agent():
    assert "`record.py`" in TEXT and "`lab.py`" in TEXT


def test_setup_has_no_git_era_instructions():
    setup = TEXT.split("## Setup", 1)[1].split("\n## ", 1)[0]
    assert "commit messages" not in setup
    assert "one run file per experiment in `runs/`" not in setup


def test_ten_minute_runs_and_ten_hour_sessions():
    assert "5 minutes" not in TEXT and "8 hours" not in TEXT and "300s" not in TEXT
    assert "10 minutes" in TEXT and "**10 hours**" in TEXT and "Time budget: 600s" in TEXT


def test_no_git_era_words_and_the_restore_route():
    assert "same commit" not in TEXT and "revert" not in TEXT
    assert "uv run lab.py restore" in TEXT


def test_no_rerun_rule_and_a_three_run_baseline():
    assert "noise rule" not in TEXT and "both runs beat" not in TEXT
    assert "three times" in TEXT and "lab.py status" in TEXT
