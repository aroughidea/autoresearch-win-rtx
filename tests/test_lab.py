"""lab.py: the record's command line, run in a temporary project folder."""
import json

import lab
import record

CODE = "MATRIX_LR = 0.05\n"


def _project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text(CODE, encoding="utf-8")
    for name in record.FIXED_FILES:
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")


def _run(capsys, *argv):
    code = lab.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_without_a_session_lab_says_how_to_start_one(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    code, _, err = _run(capsys, "status")
    assert code == 1 and "lab.py start" in err


def test_start_then_status(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    assert _run(capsys, "start", "s1", "--dataset", "tinystories", "--tokenizer", "own")[0] == 0
    code, out, _ = _run(capsys, "status")
    assert code == 0 and "s1" in out and "same as the best version" in out and "none yet" in out


def test_status_reports_a_run_in_progress(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    _run(capsys, "start", "s1", "--dataset", "tinystories", "--tokenizer", "own")
    record.begin_run(record.active_session(), dataset="tinystories", tokenizer="own", time_budget_s=300)
    code, out, _ = _run(capsys, "status")
    assert "still training, or crashed" in out


def test_keep_undo_history_diff_show(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    _run(capsys, "start", "s1", "--dataset", "tinystories", "--tokenizer", "own")
    s = record.active_session()
    run = record.begin_run(s, dataset="tinystories", tokenizer="own", time_budget_s=300)
    entry = {"schema": 1, "run_id": run["run_id"], "commit": run["version"], "version": run["version"],
             "parent": run["parent"], "status": None, "description": None, "created": run["started"],
             "final": {"val_bpb": 0.52, "peak_vram_mb": 3546.0}}
    (s.runs_dir).mkdir(parents=True, exist_ok=True)
    (s.runs_dir / f"{run['run_id']}.json").write_text(json.dumps(entry), encoding="utf-8")
    code, out, _ = _run(capsys, "keep", "baseline")
    assert code == 0 and "0.52" in out
    assert "model not archived" in out          # no checkpoint_pre_eval.pt from this run
    code, out, _ = _run(capsys, "history")
    assert code == 0 and "baseline" in out and "keep" in out and "*" in out
    (tmp_path / "train.py").write_text("MATRIX_LR = 0.07\n", encoding="utf-8")
    code, out, _ = _run(capsys, "diff")
    assert "+MATRIX_LR = 0.07" in out
    code, out, _ = _run(capsys, "show", run["version"])
    assert out == CODE
    code, out, _ = _run(capsys, "undo", "nothing ran")
    assert code == 0 and "nothing to decide" in out
    assert (tmp_path / "train.py").read_text(encoding="utf-8") == CODE


def test_export_and_check(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    _run(capsys, "start", "s1", "--dataset", "tinystories", "--tokenizer", "own")
    assert _run(capsys, "export", "--sqlite", "record.sqlite")[0] == 0
    assert (tmp_path / "record.sqlite").exists()
    code, out, _ = _run(capsys, "check")
    assert code == 0 and "consistent" in out


def test_history_prints_non_ascii_when_piped(tmp_path, monkeypatch, capsys):
    """Finding 3: piped output is cp1252 here unless lab.py asks for UTF-8."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    _project(tmp_path, monkeypatch)
    _run(capsys, "start", "s1", "--dataset", "tinystories", "--tokenizer", "own")
    s = record.active_session()
    run = record.begin_run(s, dataset="tinystories", tokenizer="own", time_budget_s=300)
    entry = {"schema": 1, "run_id": run["run_id"], "commit": run["version"], "version": run["version"],
             "parent": run["parent"], "status": None, "description": None, "created": run["started"],
             "final": {"val_bpb": 0.52, "peak_vram_mb": 3546.0}}
    s.runs_dir.mkdir(parents=True, exist_ok=True)
    (s.runs_dir / f"{run['run_id']}.json").write_text(json.dumps(entry), encoding="utf-8")
    assert _run(capsys, "keep", "LR 0.05 \u2192 0.07")[0] == 0
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    done = subprocess.run([sys.executable, str(Path(lab.__file__)), "history"], cwd=tmp_path,
                          capture_output=True, env=env)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert "\u2192" in done.stdout.decode("utf-8")


def test_history_columns_line_up_with_full_run_ids(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    _run(capsys, "start", "s1", "--dataset", "tinystories", "--tokenizer", "own")
    s = record.active_session()
    run = record.begin_run(s, dataset="tinystories", tokenizer="own", time_budget_s=300)
    entry = {"schema": 1, "run_id": run["run_id"], "commit": run["version"], "version": run["version"],
             "parent": run["parent"], "status": None, "description": None, "created": run["started"],
             "final": {"val_bpb": 0.52, "peak_vram_mb": 3546.0}}
    s.runs_dir.mkdir(parents=True, exist_ok=True)
    (s.runs_dir / f"{run['run_id']}.json").write_text(json.dumps(entry), encoding="utf-8")
    _run(capsys, "keep", "baseline")
    _, out, _ = _run(capsys, "history")
    header, row = out.splitlines()[:2]
    assert run["run_id"] in row
    assert header.index("version") == row.index(run["version"], len(run["run_id"]))
