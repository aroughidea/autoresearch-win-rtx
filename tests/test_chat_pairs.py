import json

import torch

import capture
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


def _active_pair(monkeypatch, dataset, tokenizer):
    monkeypatch.setattr(chat, "_resolve_dataset_name", lambda name=None: dataset)
    monkeypatch.setattr(chat, "_resolve_tokenizer_name", lambda name=None: tokenizer)


def _pre_eval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.save({"transformer.wte.weight": torch.zeros(8, 2)}, "checkpoint_pre_eval.pt")
    return "checkpoint_pre_eval.pt"


def test_pre_eval_uses_the_pair_recorded_beside_it(tmp_path, monkeypatch):
    """Trained on Folktales, then TinyStories prepared for the next run: still Folktales."""
    path = _pre_eval(tmp_path, monkeypatch)
    capture.write_checkpoint_pair(path, "folktales", "phi3")
    _active_pair(monkeypatch, "tinystories", "own")
    assert chat._pair_for_checkpoint(path, None) == ("folktales", "phi3")


def test_pre_eval_record_is_ignored_once_the_checkpoint_changes(tmp_path, monkeypatch):
    """git checkout puts the May checkpoint back; the record beside it is now about another file."""
    path = _pre_eval(tmp_path, monkeypatch)
    capture.write_checkpoint_pair(path, "folktales", "own")
    torch.save({"transformer.wte.weight": torch.zeros(8, 3)}, path)
    _active_pair(monkeypatch, "tinystories", "own")
    assert chat._pair_for_checkpoint(path, None) == ("tinystories", "own")


def test_pre_eval_without_a_record_is_unknown_off_the_default_pair(tmp_path, monkeypatch):
    """The reviewer's case: Folktales prepared, the May TinyStories checkpoint still in place."""
    path = _pre_eval(tmp_path, monkeypatch)
    _active_pair(monkeypatch, "folktales", "own")
    assert chat._pair_for_checkpoint(path, None) is None
    archived = _entries(tmp_path, [("b_bbbbbbb.pt", 8)])[0]
    entries = [{"id": "m1", "path": path, "label": path}, {**archived, "id": "m2"}]
    store = chat.ModelStore("cpu", entries, path, tokenizer_loader=lambda dataset, tokenizer: _Tok(8),
                            runs_dir=str(tmp_path / "runs"), results_path=str(tmp_path / "none.tsv"))
    listed = {m["id"]: m for m in store.list_models()}
    assert listed["m1"]["available"] is False and "can't tell" in listed["m1"]["reason"]
    assert store.active_id == "m2"


def test_save_pre_eval_checkpoint_records_its_pair(tmp_path, monkeypatch):
    import train

    class _T:
        dataset, name = "folktales", "gpt2"

    monkeypatch.chdir(tmp_path)
    train._save_pre_eval_checkpoint(torch.nn.Linear(2, 2), _T())
    assert capture.read_checkpoint_pair("checkpoint_pre_eval.pt") == ("folktales", "gpt2")


def test_page_says_scores_compare_only_within_a_dataset():
    facts = chat._HTML[chat._HTML.index("function factsText"):]
    assert "lower is better" in facts[:facts.index("\n  }")]
    assert 'id="compare-note"' in chat._HTML


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


class _Tok:
    def __init__(self, n):
        self.n = n

    def get_vocab_size(self):
        return self.n


def _entries(tmp_path, rows):
    out = []
    for i, (name, vocab_rows) in enumerate(rows):
        path = tmp_path / name
        torch.save({"transformer.wte.weight": torch.zeros(vocab_rows, 2)}, path)
        out.append({"id": f"m{i + 1}", "path": str(path), "label": name})
    return out


def test_mismatched_vocabulary_marks_model_unavailable(tmp_path):
    entries = _entries(tmp_path, [("a_aaaaaaa.pt", 8), ("b_bbbbbbb.pt", 9)])
    store = chat.ModelStore("cpu", entries, entries[0]["path"],
                            tokenizer_loader=lambda dataset, tokenizer: _Tok(8),
                            runs_dir=str(tmp_path / "runs"), results_path=str(tmp_path / "none.tsv"))
    listed = {m["id"]: m for m in store.list_models()}
    assert listed["m1"]["available"] is True
    assert listed["m2"]["available"] is False and "vocabulary" in listed["m2"]["reason"]


def test_missing_tokenizer_marks_model_unavailable(tmp_path):
    """One model's pair was never prepared on this machine: it is listed, not fatal."""
    entries = _entries(tmp_path, [("a_aaaaaaa.pt", 8), ("b_bbbbbbb.pt", 8)])
    runs = _write_run(tmp_path, "b_bbbbbbb", dataset="folktales", tokenizer="phi3")

    def loader(dataset, tokenizer):
        if tokenizer == "phi3":
            raise FileNotFoundError("no tokenizer.pkl")
        return _Tok(8)

    store = chat.ModelStore("cpu", entries, entries[0]["path"], tokenizer_loader=loader,
                            runs_dir=str(runs), results_path=str(tmp_path / "none.tsv"))
    listed = {m["id"]: m for m in store.list_models()}
    assert listed["m1"]["available"] is True
    assert listed["m2"]["available"] is False and "prepare.py" in listed["m2"]["reason"]


def test_no_usable_model_stops_with_a_clear_message(tmp_path):
    entries = _entries(tmp_path, [("a_aaaaaaa.pt", 8)])
    try:
        chat.ModelStore("cpu", entries, entries[0]["path"], tokenizer_loader=lambda dataset, tokenizer: _Tok(9),
                        runs_dir=str(tmp_path / "runs"), results_path=str(tmp_path / "none.tsv"))
    except RuntimeError as exc:
        assert "prepare.py" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_facts_come_from_the_run_file_and_the_scoreboard(tmp_path):
    entries = _entries(tmp_path, [("20261007T010000-0700_abc1234.pt", 8)])
    runs = _write_run(tmp_path, "20261007T010000-0700_abc1234", final={"params_m": 18.9},
                      prompts=["Once"], snapshots=[{"t_s": 0, "samples": ["x"]}])
    tsv = tmp_path / "results.tsv"
    tsv.write_text("timestamp\tcommit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
                   "2026-10-07T01:00:00-07:00\tabc1234\t1.389\t6.6\tkeep\tfolktales baseline\n", encoding="utf-8")
    store = chat.ModelStore("cpu", entries, entries[0]["path"], tokenizer_loader=lambda dataset, tokenizer: _Tok(8),
                            runs_dir=str(runs), results_path=str(tsv))
    m = store.list_models()[0]
    assert (m["dataset"], m["tokenizer"], m["params_m"]) == ("folktales", "phi3", 18.9)
    assert (m["val_bpb"], m["description"], m["has_growth"]) == (1.389, "folktales baseline", True)
    assert store.growth_for_id("m1")["prompts"] == ["Once"]


def test_page_names_research_time_and_training_time():
    """The experiments chart is research time (a new model each run); Watch it learn is training time."""
    assert "Progress: " not in chat._HTML
    assert "research time" in chat._HTML
    assert "training time" in chat._HTML


def test_pre_eval_record_rejects_a_different_file_with_the_same_size_and_time(tmp_path, monkeypatch):
    """Two checkpoints written in one clock tick can share a size and a modified time; content decides."""
    import os
    path = _pre_eval(tmp_path, monkeypatch)
    capture.write_checkpoint_pair(path, "folktales", "own")
    before = os.stat(path)
    data = bytearray(open(path, "rb").read())
    data[-200] ^= 0xFF                      # same size, different content
    open(path, "wb").write(bytes(data))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    _active_pair(monkeypatch, "tinystories", "own")
    assert chat._pair_for_checkpoint(path, None) == ("tinystories", "own")
