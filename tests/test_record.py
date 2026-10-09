"""record.py: the experiment record (session folder, versions, pointer, run entries)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

import record

PDT = timezone(timedelta(hours=-7))
WHEN = datetime(2026, 10, 9, 20, 0, 0, tzinfo=PDT)
CODE = "MATRIX_LR = 0.05\nprint('train')\n"


def _project(tmp_path, code=CODE):
    """A project folder with train.py and the fixed files."""
    (tmp_path / "train.py").write_text(code, encoding="utf-8", newline="")
    for name in record.FIXED_FILES:
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")
    return tmp_path


def _start(tmp_path, **overrides):
    _project(tmp_path)
    kwargs = dict(dataset="tinystories", tokenizer="own", run_minutes=5, hours=8,
                  recipe_path=tmp_path / "train.py", base=tmp_path / "sessions", project_root=tmp_path, now=WHEN)
    kwargs.update(overrides)
    return record.start_session("tinystories-5min-2026-10-09", **kwargs)


def test_version_id_ignores_line_endings():
    assert record.version_id("a\r\nb\r\n") == record.version_id("a\nb\n")
    assert len(record.version_id("a\n")) == 12


def test_start_session_stores_the_recipe_and_points_at_it(tmp_path):
    s = _start(tmp_path)
    vid = record.version_id(CODE)
    assert (s.versions_dir / f"{vid}.py").read_text(encoding="utf-8") == CODE
    assert record.read_best(s) == {"version": vid, "run": None, "val_bpb": None}
    assert s.meta["dataset"] == "tinystories" and s.meta["run_minutes"] == 5
    assert s.meta["ends"] == (WHEN + timedelta(hours=8)).isoformat(timespec="seconds")
    assert set(s.meta["fixed"]) == set(record.FIXED_FILES)
    assert (tmp_path / "sessions" / "active.txt").read_text(encoding="utf-8").strip() == s.name
    assert s.results_path.read_text(encoding="utf-8").splitlines() == ["\t".join(record.RESULTS_HEADER)]


def test_a_session_cannot_be_started_twice(tmp_path):
    _start(tmp_path)
    with pytest.raises(record.RecordError, match="already exists"):
        _start(tmp_path)


def test_active_session_none_without_active_txt(tmp_path):
    assert record.active_session(tmp_path / "sessions") is None


def test_active_session_names_a_missing_folder(tmp_path):
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "active.txt").write_text("gone\n", encoding="utf-8")
    with pytest.raises(record.RecordError, match="gone"):
        record.active_session(tmp_path / "sessions")


def test_crlf_train_py_is_one_version(tmp_path):
    s = _start(tmp_path)
    assert record.store_version(s, CODE.replace("\n", "\r\n")) == record.version_id(CODE)
    assert len(list(s.versions_dir.glob("*.py"))) == 1


def test_changed_fixed_files(tmp_path):
    s = _start(tmp_path)
    assert record.changed_fixed_files(s, tmp_path) == []
    (tmp_path / "prepare.py").write_text("# edited\n", encoding="utf-8")
    assert record.changed_fixed_files(s, tmp_path) == ["prepare.py"]


def test_write_atomic_leaves_no_temporary_file(tmp_path):
    record.write_atomic(tmp_path / "a" / "b.json", "{}\n")
    assert [p.name for p in (tmp_path / "a").iterdir()] == ["b.json"]
