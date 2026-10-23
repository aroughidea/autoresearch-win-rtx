"""chat.py as the model server LLM Explorables calls (its contract:
docs/superpowers/specs/2026-10-23-model-demo-contract.md in tj60647/llm-explorables). The endpoints,
field names and model ids that site depends on are pinned here, and so is how many models stay loaded."""
import json
import os

import torch
from fastapi.testclient import TestClient

import chat

RUN_A = "20261021T202736-0700_c9ba08e4f442"
RUN_B = "20261022T052816-0700_6c874e790ab4"
RUN_C = "20261021T000138-0700_c9ba08e4f442"


class _Tok:
    """Eight tokens: 0 is the start, 1 the end, the rest letters."""

    def get_vocab_size(self):
        return 8

    def get_bos_token_id(self):
        return 0

    def get_eos_token_id(self):
        return 1

    def encode(self, text, prepend=None):
        return ([prepend] if prepend is not None else []) + [2 + (ord(c) % 6) for c in text]

    def decode(self, ids):
        return "".join("abcdefgh"[i] for i in ids)


class _Model(torch.nn.Module):
    """Always writes token 3, so a generation runs to max_tokens."""

    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"sequence_len": 64})()

    def forward(self, idx):
        logits = torch.full((idx.size(0), idx.size(1), 8), -1e9)
        logits[..., 3] = 0.0
        return logits


def _checkpoints(tmp_path, stems):
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    for i, stem in enumerate(stems):
        path = folder / f"{stem}.pt"
        torch.save({"transformer.wte.weight": torch.zeros(8, 2)}, path)
        t = 1_760_000_000 + i
        os.utime(path, (t, t))
    return folder


def _store(tmp_path, monkeypatch, stems=(RUN_A, RUN_B, RUN_C), max_loaded=4, loads=None):
    _checkpoints(tmp_path, stems)
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    for stem in stems:
        (runs / f"{stem}.json").write_text(json.dumps({
            "dataset": "folktales", "tokenizer": {"name": "own", "vocab_size": 8},
            "status": "keep", "description": "a kept version", "final": {"val_bpb": 1.2, "params_m": 26.3},
        }), encoding="utf-8")

    def loader(path, device, run):
        if loads is not None:
            loads.append(path)
        return _Model()

    entries = chat._discover_checkpoints("checkpoint_pre_eval.pt")
    return chat.ModelStore("cpu", entries, f"checkpoints/{stems[0]}.pt", tokenizer_loader=lambda d, t: _Tok(),
                           runs_dir=str(runs), results_path=str(tmp_path / "none.tsv"),
                           max_loaded=max_loaded, model_loader=loader)


def _lines(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_a_models_id_is_its_run_id(tmp_path, monkeypatch):
    """Not m1, m2: those change whenever a model is added, and another site names models by id."""
    _checkpoints(tmp_path, [RUN_A, RUN_B])
    monkeypatch.chdir(tmp_path)
    ids = [e["id"] for e in chat._discover_checkpoints("checkpoint_pre_eval.pt")]
    assert ids == [RUN_A, RUN_B]


def test_two_files_with_one_stem_keep_separate_ids(tmp_path, monkeypatch):
    _checkpoints(tmp_path, [RUN_A])
    monkeypatch.chdir(tmp_path)
    torch.save({}, tmp_path / f"{RUN_A}.pt")
    ids = [e["id"] for e in chat._discover_checkpoints("checkpoint_pre_eval.pt")]
    assert sorted(ids) == [RUN_A, f"{RUN_A}-2"]


def test_models_lists_each_model_with_the_contracts_fields(tmp_path, monkeypatch):
    client = TestClient(chat.build_app(_store(tmp_path, monkeypatch)))
    models = client.get("/models").json()["models"]
    assert [m["id"] for m in models] == [RUN_A, RUN_B, RUN_C]
    for m in models:
        assert {"id", "dataset", "available", "val_bpb", "params_m", "vocab_size"} <= set(m)
    assert (models[0]["dataset"], models[0]["available"], models[0]["val_bpb"],
            models[0]["params_m"], models[0]["vocab_size"]) == ("folktales", True, 1.2, 26.3, 8)


def test_generate_streams_json_lines_from_the_model_named_by_its_run_id(tmp_path, monkeypatch):
    loads = []
    client = TestClient(chat.build_app(_store(tmp_path, monkeypatch, loads=loads)))
    r = client.post("/generate", json={"prompt": "Once", "max_tokens": 5, "temperature": 0.8,
                                       "top_k": 40, "model_id": RUN_B})
    assert r.status_code == 200
    lines = _lines(r)
    assert len(lines) == 5                                       # stops at max_tokens
    assert lines[-1]["t"] == "ddddd"                             # the continuation so far
    assert lines[-1]["toks"] == [{"id": 3, "text": "d"}] * 5
    assert loads == [f"checkpoints/{RUN_B}.pt"]


def test_generate_refuses_a_model_it_does_not_have(tmp_path, monkeypatch):
    """Answering with another model would pass its writing off as the one asked for."""
    client = TestClient(chat.build_app(_store(tmp_path, monkeypatch)))
    r = client.post("/generate", json={"prompt": "Once", "max_tokens": 5, "model_id": "m1"})
    assert r.status_code == 404 and r.json()["error"] == "unknown_model"


def test_public_mode_clamps_requests_silently(tmp_path, monkeypatch):
    seen = {}

    def fake_respond(model, tokenizer, device):
        def respond(prompt, max_tokens, temperature, top_k):
            seen.update(prompt=prompt, max_tokens=max_tokens, top_k=top_k)
            yield "x", [{"id": 3, "text": "x"}]
        return respond

    monkeypatch.setattr(chat, "_SPACE_MODE", True)
    monkeypatch.setattr(chat, "_make_respond", fake_respond)
    client = TestClient(chat.build_app(_store(tmp_path, monkeypatch)))
    r = client.post("/generate", json={"prompt": "a" * 5000, "max_tokens": 9000, "temperature": 0.8,
                                       "top_k": 9000, "model_id": RUN_A})
    assert r.status_code == 200
    assert (len(seen["prompt"]), seen["max_tokens"], seen["top_k"]) == (2000, 500, 200)


def test_at_most_max_loaded_models_stay_in_memory_least_recently_used_dropped(tmp_path, monkeypatch):
    loads = []
    store = _store(tmp_path, monkeypatch, max_loaded=2, loads=loads)
    store.get_bundle_by_id(RUN_A)
    store.get_bundle_by_id(RUN_B)
    store.get_bundle_by_id(RUN_A)                 # A is now the more recent
    store.get_bundle_by_id(RUN_C)                 # so B is dropped, not A
    assert store.loaded_ids() == [RUN_A, RUN_C]
    assert len(loads) == 3
    store.get_bundle_by_id(RUN_B)                 # B loads again; A, the least recent, goes
    assert store.loaded_ids() == [RUN_C, RUN_B]
    assert len(loads) == 4


def test_nothing_loads_until_asked_for(tmp_path, monkeypatch):
    loads = []
    store = _store(tmp_path, monkeypatch, loads=loads)
    assert store.loaded_ids() == [] and loads == []


# Within a run (the contract's "Within a run"): every kept version also answers untrained, as
# <run id>@0s, and a re-run of a best that saved moments answers as <best run id>@<seconds>s.

RERUN = "20261023T190005-0700_1a2b3c4"


def _moment_store(tmp_path, monkeypatch, loads, builds, moments=(10, 60, 300, 600)):
    _checkpoints(tmp_path, [RUN_A, RUN_B])
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "sessions" / "s1" / "runs"
    runs.mkdir(parents=True)
    for stem in (RUN_A, RUN_B):
        (runs / f"{stem}.json").write_text(json.dumps({
            "dataset": "folktales", "tokenizer": {"name": "own", "vocab_size": 8}, "version": stem[-12:],
            "status": "keep", "description": "a kept version", "final": {"val_bpb": 1.2, "params_m": 26.3},
            "prompts": ["Once"], "snapshots": [{"t_s": 0, "samples": ["the session's own"]}],
        }), encoding="utf-8")
    reruns = tmp_path / "sessions" / "s1" / "reruns"
    reruns.mkdir()
    listed = []
    for at in moments:
        name = f"{RERUN}@{at}s.pt"
        torch.save({"transformer.wte.weight": torch.zeros(8, 2)}, tmp_path / "checkpoints" / name)
        listed.append({"at_s": at, "t_s": at + 0.4, "step": at * 3, "checkpoint": name,
                       "checkpoint_sha256": "0" * 64, **({"final": True} if at == 600 else {})})
    (reruns / f"{RERUN}.json").write_text(json.dumps({
        "run_id": RERUN, "rerun_of": RUN_B, "dataset": "folktales",
        "tokenizer": {"name": "own", "vocab_size": 8}, "time_budget_s": 600,
        "final": {"val_bpb": 1.21, "params_m": 33.6}, "moments": listed,
        "prompts": ["Once"], "snapshots": [{"t_s": 10, "samples": ["the re-run's own"]}],
    }), encoding="utf-8")

    def loader(path, device, run):
        loads.append((path, run.get("version")))
        return _Model()

    def builder(path, device, run):
        builds.append((path, run.get("version")))
        return _Model()

    entries = chat._discover_checkpoints("checkpoint_pre_eval.pt")
    return chat.ModelStore("cpu", entries, f"checkpoints/{RUN_B}.pt", tokenizer_loader=lambda d, t: _Tok(),
                           runs_dir=str(runs), results_path=str(tmp_path / "none.tsv"),
                           model_loader=loader, untrained_builder=builder)


def test_models_lists_each_untrained_model_and_each_saved_moment(tmp_path, monkeypatch):
    store = _moment_store(tmp_path, monkeypatch, [], [])
    models = {m["id"]: m for m in TestClient(chat.build_app(store)).get("/models").json()["models"]}
    assert sorted(models) == sorted([RUN_A, RUN_B, f"{RUN_A}@0s", f"{RUN_B}@0s",
                                     f"{RUN_B}@10s", f"{RUN_B}@60s", f"{RUN_B}@300s", f"{RUN_B}@600s"])
    assert "moment_s" not in models[RUN_A] or models[RUN_A]["moment_s"] is None
    untrained = models[f"{RUN_A}@0s"]
    assert (untrained["moment_s"], untrained["available"], untrained["dataset"], untrained["vocab_size"]) == (0, True, "folktales", 8)
    assert untrained.get("rerun_of") is None and untrained["val_bpb"] is None
    ten = models[f"{RUN_B}@10s"]
    assert (ten["moment_s"], ten["rerun_of"], ten["rerun_id"], ten["available"]) == (10, RUN_B, RERUN, True)
    assert ten["val_bpb"] is None and models[f"{RUN_B}@600s"]["val_bpb"] == 1.21
    assert store.active_id == RUN_B                      # the page still opens on a session's own model
    # The re-run's files are not listed under their own names.
    assert not any(i.startswith(RERUN) for i in models)


def test_an_untrained_model_is_built_from_its_versions_code_never_loaded(tmp_path, monkeypatch):
    loads, builds = [], []
    client = TestClient(chat.build_app(_moment_store(tmp_path, monkeypatch, loads, builds)))
    r = client.post("/generate", json={"prompt": "Once", "max_tokens": 3, "model_id": f"{RUN_A}@0s"})
    assert r.status_code == 200 and len(_lines(r)) == 3
    assert builds == [(f"checkpoints/{RUN_A}.pt", RUN_A[-12:])] and loads == []


def test_a_saved_moment_loads_its_file_with_the_code_of_the_run_it_repeats(tmp_path, monkeypatch):
    loads, builds = [], []
    store = _moment_store(tmp_path, monkeypatch, loads, builds)
    client = TestClient(chat.build_app(store))
    r = client.post("/generate", json={"prompt": "Once", "max_tokens": 2, "model_id": f"{RUN_B}@60s"})
    assert r.status_code == 200
    assert loads == [(f"checkpoints/{RERUN}@60s.pt", RUN_B[-12:])] and builds == []
    # Its growth card is the re-run's own samples, not the session run's.
    assert client.get("/growth", params={"model_id": f"{RUN_B}@60s"}).json()["snapshots"][0]["samples"] == ["the re-run's own"]
    assert client.get("/growth", params={"model_id": RUN_B}).json()["snapshots"][0]["samples"] == ["the session's own"]


def test_a_moment_whose_file_is_missing_is_not_listed(tmp_path, monkeypatch):
    store = _moment_store(tmp_path, monkeypatch, [], [])
    (tmp_path / "checkpoints" / f"{RERUN}@300s.pt").unlink()
    ids = [m["id"] for m in chat.ModelStore("cpu", chat._discover_checkpoints("x.pt"), f"checkpoints/{RUN_B}.pt",
                                            tokenizer_loader=lambda d, t: _Tok(),
                                            runs_dir=str(tmp_path / "sessions" / "s1" / "runs"),
                                            results_path=str(tmp_path / "none.tsv")).list_models()]
    assert f"{RUN_B}@300s" not in ids and f"{RUN_B}@10s" in ids and store is not None


def test_chat_pys_own_page_lists_only_the_sessions_models():
    """The pickers would otherwise hold 24 models; the page keeps to the runs' own."""
    assert "m.moment_s == null" in chat._HTML


class _Tiny(torch.nn.Module):
    """A version's network in miniature: built on meta, given storage, then init_weights."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lin = torch.nn.Linear(4, 4)

    def init_weights(self, embed_dtype=torch.bfloat16):
        torch.nn.init.normal_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)


def test_building_an_untrained_model_uses_seed_42_and_leaves_the_servers_randomness_alone(tmp_path, monkeypatch):
    import types
    code = types.SimpleNamespace(GPT=_Tiny)
    monkeypatch.setattr(chat, "_code_and_config", lambda state, run: (code, types.SimpleNamespace(sequence_len=8)))
    path = tmp_path / "m.pt"
    torch.save({"transformer.wte.weight": torch.zeros(8, 2)}, path)
    torch.manual_seed(7)
    before = torch.rand(1)
    torch.manual_seed(7)
    one = chat._untrained_model(str(path), "cpu", {"version": "v"})
    after = torch.rand(1)
    two = chat._untrained_model(str(path), "cpu", {"version": "v"})
    assert torch.equal(before, after), "the server's own sampling randomness is untouched"
    assert torch.equal(one.lin.weight, two.lin.weight), "the same untrained model every time"
    torch.manual_seed(42)
    expected = torch.nn.init.normal_(torch.empty(4, 4))
    assert torch.equal(one.lin.weight, expected), "seed 42, as train.py"
    assert not one.training
