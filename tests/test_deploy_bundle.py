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
