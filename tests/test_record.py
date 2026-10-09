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


def _begin(s, tmp_path, now=WHEN, **overrides):
    kwargs = dict(dataset="tinystories", tokenizer="own", time_budget_s=300,
                  recipe_path=tmp_path / "train.py", project_root=tmp_path, now=now)
    kwargs.update(overrides)
    return record.begin_run(s, **kwargs)


def _finish(s, run, val_bpb=0.52, peak_vram_mb=3546.0):
    """What capture.py writes when a run finishes: the run file with the record's fields."""
    entry = {"schema": 1, "run_id": run["run_id"], "commit": run["version"], "version": run["version"],
             "parent": run["parent"], "status": None, "description": None, "created": run["started"],
             "dataset": "tinystories", "prompts": ["Once"], "snapshots": [{"t_s": 0.0, "samples": ["x"]}],
             "final": {"val_bpb": val_bpb, "peak_vram_mb": peak_vram_mb}}
    record._write_json(s.runs_dir / f"{run['run_id']}.json", entry)
    return entry


def _decide(s, tmp_path, keep, description="tried a thing", **overrides):
    kwargs = dict(recipe_path=tmp_path / "train.py", checkpoint=tmp_path / "checkpoint_pre_eval.pt",
                  checkpoints_dir=tmp_path / "checkpoints")
    kwargs.update(overrides)
    return record.decide(s, keep, description, **kwargs)


def test_begin_run_records_version_and_parent(tmp_path):
    s = _start(tmp_path)
    run = _begin(s, tmp_path)
    assert run["version"] == run["parent"] == record.version_id(CODE)
    assert run["run_id"] == f"20261009T200000-0700_{run['version']}"
    assert json.loads(s.current_path.read_text(encoding="utf-8")) == run


def test_begin_run_refuses_a_changed_fixed_file(tmp_path):
    s = _start(tmp_path)
    (tmp_path / "capture.py").write_text("# edited\n", encoding="utf-8")
    with pytest.raises(record.RecordError, match="capture.py"):
        _begin(s, tmp_path)


def test_begin_run_refuses_another_pair_or_run_length(tmp_path):
    s = _start(tmp_path)
    with pytest.raises(record.RecordError, match="folktales"):
        _begin(s, tmp_path, dataset="folktales")
    with pytest.raises(record.RecordError, match="600"):
        _begin(s, tmp_path, time_budget_s=600)


def test_the_entry_holds_the_code_that_trained_even_if_train_py_changes(tmp_path):
    s = _start(tmp_path)
    run = _begin(s, tmp_path)
    (tmp_path / "train.py").write_text("MATRIX_LR = 0.99\n", encoding="utf-8")   # edited mid-run
    _finish(s, run)
    assert record.read_version(s, record.entries(s)[0]["version"]) == CODE


def test_keep_moves_the_pointer_and_archives_the_model(tmp_path):
    s = _start(tmp_path)
    (tmp_path / "checkpoint_pre_eval.pt").write_bytes(b"weights")
    run = _begin(s, tmp_path)
    _finish(s, run, val_bpb=0.52)
    result = _decide(s, tmp_path, keep=True, description="baseline")
    assert result["status"] == "keep"
    assert record.read_best(s) == {"version": run["version"], "run": run["run_id"], "val_bpb": 0.52}
    entry = record.entries(s)[0]
    assert entry["status"] == "keep" and entry["description"] == "baseline"
    assert (tmp_path / "checkpoints" / f"{run['run_id']}.pt").read_bytes() == b"weights"
    assert entry["checkpoint_sha256"]


def test_undo_resets_train_py_to_the_best_version(tmp_path):
    s = _start(tmp_path)
    _finish(s, _begin(s, tmp_path)); _decide(s, tmp_path, keep=True, description="baseline")
    (tmp_path / "train.py").write_text("MATRIX_LR = 0.07\n", encoding="utf-8")
    run = _begin(s, tmp_path, now=WHEN + timedelta(minutes=10))
    assert run["parent"] == record.version_id(CODE) != run["version"]
    _finish(s, run, val_bpb=0.53)
    assert _decide(s, tmp_path, keep=False, description="matrix lr 0.07")["status"] == "discard"
    assert (tmp_path / "train.py").read_text(encoding="utf-8") == CODE
    assert record.read_best(s)["version"] == record.version_id(CODE)
    assert record.version_id((tmp_path / "train.py").read_text(encoding="utf-8")) == record.version_id(CODE)
    assert len(list(s.versions_dir.glob("*.py"))) == 2   # the tried version stays


def test_a_confirmation_rerun_shares_the_version_and_keep_averages(tmp_path):
    s = _start(tmp_path)
    (tmp_path / "checkpoint_pre_eval.pt").write_bytes(b"w")
    a = _begin(s, tmp_path); _finish(s, a, val_bpb=0.50)
    b = _begin(s, tmp_path, now=WHEN + timedelta(minutes=10)); _finish(s, b, val_bpb=0.52)
    assert a["version"] == b["version"] and a["run_id"] != b["run_id"]
    assert _decide(s, tmp_path, keep=True)["runs"] == [a["run_id"], b["run_id"]]
    assert record.read_best(s)["val_bpb"] == 0.51


def test_a_different_train_py_waits_for_a_decision(tmp_path):
    s = _start(tmp_path)
    _finish(s, _begin(s, tmp_path))
    (tmp_path / "train.py").write_text("MATRIX_LR = 0.07\n", encoding="utf-8")
    with pytest.raises(record.RecordError, match="keep"):
        _begin(s, tmp_path, now=WHEN + timedelta(minutes=10))


def test_decisions_refuse_clear_mistakes(tmp_path):
    s = _start(tmp_path)
    with pytest.raises(record.RecordError, match="no run has finished"):
        _decide(s, tmp_path, keep=True)
    _begin(s, tmp_path)                                   # starts, never finishes: a crash
    assert record.unfinished(s) is not None
    with pytest.raises(record.RecordError, match="did not finish"):
        _decide(s, tmp_path, keep=True)
    assert _decide(s, tmp_path, keep=False, description="double width (OOM)")["status"] == "crash"
    assert record.entries(s)[0]["status"] == "crash" and record.unfinished(s) is None
    with pytest.raises(record.RecordError, match="few words"):
        _decide(s, tmp_path, keep=False, description="  ")


def test_an_unfinished_run_becomes_a_crash_when_the_next_starts(tmp_path):
    s = _start(tmp_path)
    first = _begin(s, tmp_path)
    _begin(s, tmp_path, now=WHEN + timedelta(minutes=10))
    crashed = [e for e in record.entries(s) if e["run_id"] == first["run_id"]][0]
    assert crashed["status"] == "crash" and crashed["description"] == "did not finish"


def test_results_tsv_lists_decided_runs_in_six_columns(tmp_path):
    s = _start(tmp_path)
    _finish(s, _begin(s, tmp_path), val_bpb=0.52)
    _decide(s, tmp_path, keep=False, description="tabs\tand\nnewlines, café")
    lines = s.results_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "\t".join(record.RESULTS_HEADER)
    row = lines[1].split("\t")
    assert len(row) == 6 and row[1] == record.version_id(CODE) and row[2] == "0.520000"
    assert row[3] == "3.5" and row[4] == "discard" and row[5] == "tabs and newlines, café"


def test_history_marks_the_best_and_diff_shows_the_change(tmp_path):
    s = _start(tmp_path)
    _finish(s, _begin(s, tmp_path)); _decide(s, tmp_path, keep=True, description="baseline")
    (tmp_path / "train.py").write_text(CODE.replace("0.05", "0.07"), encoding="utf-8")
    assert [r["best"] for r in record.history(s)] == [True]
    text = record.diff(s, recipe_path=tmp_path / "train.py")
    assert "-MATRIX_LR = 0.05" in text and "+MATRIX_LR = 0.07" in text


def test_export_sqlite_and_check(tmp_path):
    import contextlib
    import sqlite3
    s = _start(tmp_path)
    _finish(s, _begin(s, tmp_path)); _decide(s, tmp_path, keep=True, description="baseline")
    path = record.export_sqlite(s, tmp_path / "record.sqlite")
    with contextlib.closing(sqlite3.connect(path)) as db:   # closed, so Windows can delete tmp_path
        assert db.execute("SELECT count(*) FROM versions").fetchone() == (1,)
        assert db.execute("SELECT status FROM runs").fetchone() == ("keep",)
    assert record.check(s, project_root=tmp_path) == []
    s.results_path.write_text("tampered\n", encoding="utf-8")
    assert any("results.tsv" in p for p in record.check(s, project_root=tmp_path))
