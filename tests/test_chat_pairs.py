import json

import torch

import chat


def test_commit_from_name():
    assert chat._commit_from_name("20260523T175831-0700_e9fffd9.pt") == "e9fffd9"
    assert chat._commit_from_name("checkpoint_pre_eval.pt") is None


def _write_run(tmp_path, stem, dataset="folktales", tokenizer="phi3", **extra):
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    record = {"dataset": dataset, "tokenizer": {"name": tokenizer, "vocab_size": 32015}, **extra}
    (runs / f"{stem}.json").write_text(json.dumps(record), encoding="utf-8")
    return runs


def test_run_file_found_by_checkpoint_stem(tmp_path):
    runs = _write_run(tmp_path, "20261007T010000-0700_abc1234")
    run = chat._run_for_checkpoint("checkpoints/20261007T010000-0700_abc1234.pt", runs_dir=str(runs))
    assert run["dataset"] == "folktales"
    assert chat._pair_for_checkpoint("checkpoints/20261007T010000-0700_abc1234.pt", run) == ("folktales", "phi3")


def test_malformed_run_file_is_ignored(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "x_abc1234.json").write_text("{not json", encoding="utf-8")
    assert chat._run_for_checkpoint("checkpoints/x_abc1234.pt", runs_dir=str(runs)) is None


def test_archived_checkpoint_without_run_uses_defaults():
    assert chat._pair_for_checkpoint("checkpoints/20260523T175831-0700_e9fffd9.pt", None) == ("tinystories", "own")


def test_pre_eval_checkpoint_uses_the_active_pair(monkeypatch):
    monkeypatch.setattr(chat, "_resolve_dataset_name", lambda name=None: "folktales")
    monkeypatch.setattr(chat, "_resolve_tokenizer_name", lambda name=None: "gpt2")
    assert chat._pair_for_checkpoint("checkpoint_pre_eval.pt", None) == ("folktales", "gpt2")


def test_checkpoint_vocab_rows(tmp_path):
    path = tmp_path / "m.pt"
    torch.save({"transformer.wte.weight": torch.zeros(37, 4)}, path)
    assert chat._checkpoint_vocab_rows(str(path)) == 37
    assert chat._checkpoint_vocab_rows(str(tmp_path / "missing.pt")) is None


def test_results_by_commit_tolerates_blank_scores(tmp_path):
    tsv = tmp_path / "results.tsv"
    tsv.write_text(
        "timestamp\tcommit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "2026-05-23T15:57:43-07:00\t75027e8\t0.520096\t6.6\tkeep\tbaseline\n"
        "2026-05-23T16:00:00-07:00\tdeadbee\tmissing\t0.0\tcrash\toom\n",
        encoding="utf-8",
    )
    rows = chat._results_by_commit(str(tsv))
    assert rows["75027e8"] == {"val_bpb": 0.520096, "status": "keep", "description": "baseline"}
    assert rows["deadbee"]["val_bpb"] is None


def test_growth_from_run():
    run = {"prompts": ["Once upon a time"], "snapshots": [
        {"t_s": 0.0, "samples": ["noise"]}, {"t_s": 300.4, "samples": ["a story"], "final": True}]}
    growth = chat._growth_from_run(run)
    assert growth["prompts"] == ["Once upon a time"]
    assert [s["t_s"] for s in growth["snapshots"]] == [0.0, 300.4]
    assert chat._growth_from_run({"prompts": [], "snapshots": []}) is None
    assert chat._growth_from_run(None) is None
