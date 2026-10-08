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
