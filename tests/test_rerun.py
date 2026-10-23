"""deploy/rerun.py: re-running a featured best with the moment switch, and filing what it made where
chat.py serves it. A fake version stands in for the training: it drives capture.py as train.py does."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rerun", ROOT / "deploy" / "rerun.py")
rerun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rerun)

BEST = "20261022T052816-0700_6c874e790ab4"

FAKE_VERSION = '''
import sys, torch
from capture import RunCapture, runs_dir_from_env
from prepare import TIME_BUDGET

class Tok:
    name = "own"
    def get_vocab_size(self): return 8

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2))

def main():
    assert sys.argv[1:] == ["--dataset", "folktales"], sys.argv
    cap = RunCapture(Tok(), dataset="folktales", runs_dir=runs_dir_from_env(False), device="cpu", snapshot_times=())
    model = Model()
    for t in (0, 10, 30, 60, 120, 300, TIME_BUDGET):
        with torch.no_grad():
            model.w.fill_(t)
        cap.on_step(model, float(t), t, 1.0)
    torch.save(model.state_dict(), "checkpoint_pre_eval.pt")   # the end, as train.py saves it; never filed
    cap.finish(val_bpb=1.21, peak_vram_mb=1.0, training_seconds=float(TIME_BUDGET), num_steps=1800,
               num_params=33_600_000, recipe={}, commit="abc1234", committed_at=None)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def _project(tmp_path, monkeypatch):
    """A copy of the deploy layout with one session, its best's run file and a fake version."""
    deploy = tmp_path / "deploy"
    session = deploy / "sessions" / "folktales-b-10min-2026-10-21"
    (session / "runs").mkdir(parents=True)
    (session / "versions").mkdir()
    (session / "runs" / f"{BEST}.json").write_text(json.dumps({
        "run_id": BEST, "version": "6c874e790ab4", "dataset": "folktales", "time_budget_s": 600,
        "tokenizer": {"name": "own", "vocab_size": 8192},
        "final": {"val_bpb": 1.207438, "num_steps": 1790}}), encoding="utf-8")
    (session / "versions" / "6c874e790ab4.py").write_text(FAKE_VERSION, encoding="utf-8")
    for name in ("capture.py", "prepare.py", "record.py"):
        (tmp_path / name).write_text((ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(rerun, "ROOT", tmp_path)
    monkeypatch.setattr(rerun, "DEPLOY", deploy)
    monkeypatch.chdir(tmp_path)
    for name in ("AUTORESEARCH_RUNS_DIR", "AR_SAVE_MOMENTS", "AR_RERUN_OF", "AUTORESEARCH_TIME_BUDGET"):
        monkeypatch.delenv(name, raising=False)
    return session


def test_the_plan_runs_the_versions_own_code_with_the_switch_on(tmp_path, monkeypatch):
    session = _project(tmp_path, monkeypatch)
    p = rerun.plan(BEST, tmp_path / "work")
    assert p["version"] == session / "versions" / "6c874e790ab4.py" and p["dataset"] == "folktales"
    assert p["env"]["AR_SAVE_MOMENTS"] == "10,60,300" and p["env"]["AR_RERUN_OF"] == BEST
    assert p["env"]["AUTORESEARCH_RUNS_DIR"] == str(tmp_path / "work")
    assert p["env"]["AUTORESEARCH_TIME_BUDGET"] == "600"
    # The run's own pair, never whatever this machine has active.
    assert (p["env"]["AUTORESEARCH_DATASET"], p["env"]["AUTORESEARCH_TOKENIZER"]) == ("folktales", "own")
    assert p["command"][0] == sys.executable and p["command"][-2:] == [str(p["version"]), "folktales"]
    with pytest.raises(SystemExit, match="no run"):
        rerun.plan("20200101T000000-0700_000000000000", tmp_path / "work")


def test_a_rerun_files_its_run_file_and_three_models_where_chat_py_serves_them(tmp_path, monkeypatch, capsys):
    session = _project(tmp_path, monkeypatch)
    assert rerun.main([BEST]) == 0
    reruns = list((session / "reruns").glob("*.json"))
    assert len(reruns) == 1
    record = json.loads(reruns[0].read_text(encoding="utf-8"))
    assert record["rerun_of"] == BEST
    assert [m["at_s"] for m in record["moments"]] == [10, 60, 300]
    files = sorted(p.name for p in (tmp_path / "deploy" / "checkpoints").iterdir())
    assert files == sorted(m["checkpoint"] for m in record["moments"])
    assert all(name.startswith(record["run_id"] + "@") for name in files)
    assert not (tmp_path / "deploy" / ".rerun-work").exists()
    out = capsys.readouterr().out
    assert "1800 steps" in out and "1790 steps" in out


def test_it_refuses_to_run_where_a_session_is_active(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "active.txt").write_text("s1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="session is active"):
        rerun.main([BEST])


def test_a_rerun_missing_a_moment_files_nothing(tmp_path, monkeypatch):
    session = _project(tmp_path, monkeypatch)
    work = tmp_path / "work"
    work.mkdir()
    (work / "x.json").write_text(json.dumps({"run_id": "x", "rerun_of": BEST, "moments": [{"at_s": 10}],
                                             "tokenizer": {"name": "own"},
                                             "moments_error": "OSError: disk full"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="disk full"):
        rerun.file_rerun(work, session, {"run_id": BEST, "time_budget_s": 600}, tmp_path / "deploy" / "checkpoints")
    assert not (session / "reruns").exists()


def test_a_rerun_on_another_vocabulary_files_nothing(tmp_path, monkeypatch):
    session = _project(tmp_path, monkeypatch)
    work = tmp_path / "work"
    work.mkdir()
    (work / "x.json").write_text(json.dumps({"run_id": "x", "rerun_of": BEST, "tokenizer": {"name": "phi3"},
                                             "moments": [{"at_s": a} for a in (10, 60, 300)]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="phi3 vocabulary"):
        rerun.file_rerun(work, session, {"run_id": BEST, "time_budget_s": 600, "tokenizer": {"name": "own"}},
                         tmp_path / "deploy" / "checkpoints")
    assert not (session / "reruns").exists()
