import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _local_imports(module, seen):
    """Repo-root modules that `module` imports, followed transitively."""
    if module in seen:
        return seen
    seen.add(module)
    tree = ast.parse((ROOT / f"{module}.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module.split(".")[0]]
        for name in names:
            if (ROOT / f"{name}.py").exists():
                _local_imports(name, seen)
    return seen


def test_demo_image_copies_every_module_chat_py_needs():
    """The public demo crashed on 2026-10-07: train.py began importing capture.py and the
    Dockerfile's COPY list did not include it."""
    dockerfile = (ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    copied = set(re.findall(r"\b([a-z_]+)\.py\b", " ".join(l for l in dockerfile.splitlines() if l.startswith("COPY"))))
    needed = _local_imports("chat", set())
    assert needed <= copied, f"deploy/Dockerfile COPY is missing: {sorted(needed - copied)}"


def _content_sha256(path):
    """A model's SHA-256: the file's own, or, for a Git LFS pointer not fetched, the oid it records
    (LFS names a file by the SHA-256 of its content)."""
    import hashlib
    head = path.read_bytes()[:200]
    if head.startswith(b"version https://git-lfs"):
        return re.search(rb"oid sha256:([0-9a-f]{64})", path.read_bytes()).group(1).decode()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_runs():
    import json
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "deploy" / "sessions").glob("*/runs/*.json"))]


def test_every_model_the_demo_serves_is_the_one_its_run_saved():
    """deploy/checkpoints/ holds exactly the models deploy/sessions/' run files name, each with the
    SHA-256 its run recorded, so the page never pairs a model with another run's record."""
    runs = _bundle_runs()
    assert runs, "deploy/sessions/ has no run files"
    named = {pathlib.Path(r["checkpoint"]).name: r["checkpoint_sha256"] for r in runs}
    present = {p.name: p for p in (ROOT / "deploy" / "checkpoints").glob("*.pt")}
    missing = sorted(set(named) - set(present))
    assert not missing, f"copy these models into deploy/checkpoints/ (see deploy/README.md): {missing}"
    assert sorted(present) == sorted(named), f"deploy/checkpoints/ holds models no run file names: {sorted(set(present) - set(named))}"
    for name, sha in named.items():
        assert _content_sha256(present[name]) == sha, f"{name} is not the model its run saved"


def test_each_served_model_has_the_code_that_trained_it():
    for run in _bundle_runs():
        version = run["version"]
        assert list((ROOT / "deploy" / "sessions").glob(f"*/versions/{version}.py")), f"no versions/{version}.py for {run['run_id']}"


def test_the_image_opens_on_a_model_it_serves_from_the_active_session():
    import json
    dockerfile = (ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    opens = re.search(r'"--checkpoint", "checkpoints/([^"]+)"', dockerfile).group(1)
    active = (ROOT / "deploy" / "sessions" / "active.txt").read_text(encoding="utf-8").strip()
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "deploy" / "sessions" / active / "runs").glob("*.json")]
    assert opens in {pathlib.Path(r["checkpoint"]).name for r in runs}


# The eight models LLM Explorables' model demo names (its contract, and shared/DemoModels.js there):
# per session the start, two kept along the way, and the best. A model's id is its run id.
CONTRACT_MODELS = {
    "folktales-b-10min-2026-10-21": [
        "20261021T202736-0700_c9ba08e4f442", "20261021T203936-0700_ae5c68b35cb6",
        "20261021T213750-0700_7d99f54b4193", "20261022T052816-0700_6c874e790ab4",
    ],
    "tinystories-b-10min-2026-10-20": [
        "20261021T000138-0700_c9ba08e4f442", "20261021T003646-0700_6db69b64b97b",
        "20261021T024603-0700_b351add2f9da", "20261021T130839-0700_05602a6a5706",
    ],
}


def test_the_bundle_serves_the_eight_models_the_explorables_name():
    import json
    sessions = ROOT / "deploy" / "sessions"
    bundled = {s.name: sorted(p.stem for p in (s / "runs").glob("*.json")) for s in sessions.iterdir() if s.is_dir()}
    assert bundled == CONTRACT_MODELS
    for session, ids in CONTRACT_MODELS.items():
        for run_id in ids:
            run = json.loads((sessions / session / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))
            assert run["status"] == "keep" and run["checkpoint"] == f"checkpoints/{run_id}.pt"
