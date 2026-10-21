import ast
import subprocess
import sys
from pathlib import Path

import record

ROOT = Path(__file__).resolve().parent.parent
RECIPE_B = ROOT / "library" / "recipes" / "recipe-b.py"
STARTER = "6ad8ddd"  # train.py with the starter kit's recipe (library/README.md)


def settings(source):
    """train.py's settings: the module-level assignments to upper-case names, as values."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():
                try:
                    out[name] = ast.literal_eval(node.value)
                except ValueError:
                    out[name] = ast.unparse(node.value)
    return out


def starter_train_py():
    return subprocess.run(["git", "show", f"{STARTER}:train.py"], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout


def test_recipe_b_is_the_version_measured():
    # c9ba08e4f442 is the version the 13 October bench trained (ar-bench-attention-b-2026-10-13).
    assert record.version_id(RECIPE_B.read_text(encoding="utf-8")) == "c9ba08e4f442"


def test_recipe_b_changes_only_the_attention_settings():
    a, b = settings(starter_train_py()), settings(RECIPE_B.read_text(encoding="utf-8"))
    changed = {k: (a.get(k), b.get(k)) for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert changed == {"N_KV_HEAD": (1, 3), "WINDOW_PATTERN": ("SSSL", "L")}


def test_make_recipe_b_remakes_it_from_the_starter(tmp_path):
    out = tmp_path / "recipe-b.py"
    subprocess.run([sys.executable, str(ROOT / "library" / "recipes" / "make_recipe_b.py"), str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    assert out.read_bytes() == RECIPE_B.read_bytes()
