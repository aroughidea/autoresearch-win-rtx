# chat.py
# Author: Thomas J McLeish
# License: MIT
#
# Launch a local browser UI for the trained model.
# The model continues whatever text you type, token by token.
#
# Usage:
#   uv run chat.py
#   uv run chat.py --checkpoint checkpoint_pre_eval.pt --port 8000
#
# Prerequisites:
#   uv run prepare.py   (one-time setup)
#   uv run train.py     (produces checkpoint_pre_eval.pt)
#
# Public hosting: setting SPACE_MODE=1 switches this same file into public-demo
# mode — binds 0.0.0.0:$PORT, clamps generation requests, and adds a banner
# saying which models it serves, with links to How a Model Gets Made and the README. Unset (the default), nothing below changes.
# See deploy/ for the container that runs it.

import argparse
import csv
import json
import math
import re
import os
import socket
import sys
import threading
import time
import types
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import importlib.util
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from generate import _config_from_state_dict, _sample_top_k
from capture import read_checkpoint_pair
from record import RecordError, active_session, compact_timestamp
from prepare import (
    DEFAULT_DATASET,
    DEFAULT_TOKENIZER,
    TOKENIZER_SOURCES,
    Tokenizer,
    _resolve_dataset_name,
    _resolve_tokenizer_name,
)
import train
from train import GPT


# ---------------------------------------------------------------------------
# Load model (startup + on-demand switch)
# ---------------------------------------------------------------------------

_VERSION_MODULES: dict[str, types.ModuleType] = {}


def _version_code(run: dict | None) -> Path | None:
    """The train.py that trained a run, as the record kept it: sessions/<name>/versions/<version>.py."""
    version = (run or {}).get("version")
    if not version:
        return None
    found = sorted(Path("sessions").glob(f"*/versions/{version}.py"))
    return found[0] if found else None


def _import_version(path: Path):
    """Import a version's train.py under its own name (train.py only defines things on import; it trains
    under `if __name__ == "__main__"`), once per file."""
    key = os.path.normcase(os.path.abspath(path))
    if key not in _VERSION_MODULES:
        name = f"_version_{Path(path).stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load the version's code from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module      # dataclasses (and pickle) look a module up by name while it loads
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
        _VERSION_MODULES[key] = module
    return _VERSION_MODULES[key]


def _code_and_config(state_dict: dict, run: dict | None):
    """The code to rebuild a saved model with, and its config. A model is only its weights: how they are
    wired (attention, windows, the cap on the output scores) is in the code that trained it. So: that
    version's train.py when the record kept it; else today's train.py with the attention windows its run
    file recorded; else today's train.py as it stands."""
    path = _version_code(run)
    if path is not None:
        code = _import_version(path)
        pattern = getattr(code, "WINDOW_PATTERN", None)
    else:
        code = train
        pattern = ((run or {}).get("recipe") or {}).get("WINDOW_PATTERN") or getattr(train, "WINDOW_PATTERN", None)
    return code, _config_from_state_dict(state_dict, config_cls=code.GPTConfig, window_pattern=pattern,
                                         head_dim=getattr(code, "HEAD_DIM", None))


def _load_model_from_checkpoint(checkpoint_path: str, device: str, run: dict | None = None):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    code, config = _code_and_config(state_dict, run)
    model = code.GPT(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _discover_checkpoints(primary_checkpoint: str) -> list[dict]:
    candidates: list[Path] = []
    primary = Path(primary_checkpoint)
    if primary.exists():
        candidates.append(primary)

    root = Path(".")
    checkpoints_dir = root / "checkpoints"
    if checkpoints_dir.exists():
        candidates.extend(sorted(checkpoints_dir.glob("*.pt")))
    candidates.extend(sorted(root.glob("*.pt")))

    seen: set[Path] = set()
    entries: list[dict] = []
    for path in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        stat = path.stat()
        entries.append(
            {
                "path": str(path).replace("\\", "/"),
                "mtime_ts": stat.st_mtime,
                "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "size_mb": stat.st_size / (1024 * 1024),
                "label": path.name,
            }
        )

    entries.sort(key=lambda e: e["mtime_ts"])
    for i, entry in enumerate(entries):
        entry["id"] = f"m{i + 1}"
    return entries


# ---------------------------------------------------------------------------
# Which dataset and tokenizer each model was trained with
# ---------------------------------------------------------------------------

DATASET_BLURBS = {
    "tinystories": "short children’s stories",
    "folktales": "folk and fairy tales",
}


def _commit_from_name(name: str) -> str | None:
    match = re.search(r"_([0-9a-f]{7,40})\.pt$", name)
    return match.group(1) if match else None


def _run_for_checkpoint(path: str, runs_dir: str = "runs") -> dict | None:
    """The run file saved with a checkpoint: same <timestamp>_<commit> stem. Looked for in runs_dir
    first, then in the root runs/ and every session's runs/: a model kept in an earlier session has
    its run file there, and without it the model would be decoded with the default pair's tokenizer."""
    name = Path(path).stem + ".json"
    seen = set()
    for folder in [Path(runs_dir), Path("runs"), *sorted(Path("sessions").glob("*/runs"))]:
        key = os.path.normcase(os.path.abspath(folder))
        if key in seen:
            continue
        seen.add(key)
        try:
            return json.loads((folder / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def _session_paths() -> tuple[str, str]:
    """Where the page reads runs and the scoreboard: the active session's folder, else the repo root.
    A broken sessions/active.txt is reported, not fatal: the page still opens on the root files."""
    try:
        session = active_session()
    except RecordError as exc:
        print(f"{exc}; showing the root runs/ and results.tsv instead")
        return "runs", "results.tsv"
    return (str(session.runs_dir), str(session.results_path)) if session else ("runs", "results.tsv")


def _pair_for_checkpoint(path: str, run: dict | None) -> tuple[str, str] | None:
    """(dataset, tokenizer) a checkpoint was trained with, or None when that can't be told.

    From its run file when there is one. checkpoint_pre_eval.pt has none, so train.py records
    its pair beside it; without that record it is only assumed default while the default pair
    is active (a guess on any other pair would decode it with the wrong tokenizer). Archived
    checkpoints without a run file predate the choice.
    """
    if run:
        tokenizer = (run.get("tokenizer") or {}).get("name") or DEFAULT_TOKENIZER
        return run.get("dataset") or DEFAULT_DATASET, tokenizer
    if Path(path).name == "checkpoint_pre_eval.pt":
        recorded = read_checkpoint_pair(path)
        if recorded:
            return recorded
        active = (_resolve_dataset_name(None), _resolve_tokenizer_name(None))
        return active if active == (DEFAULT_DATASET, DEFAULT_TOKENIZER) else None
    return DEFAULT_DATASET, DEFAULT_TOKENIZER


def _checkpoint_vocab_rows(path: str) -> int | None:
    """Rows of the embedding table, read without loading the whole checkpoint."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        return int(state["transformer.wte.weight"].shape[0])
    except Exception:
        return None


def _results_by_commit(path: str = "results.tsv") -> dict[str, dict]:
    rows: dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        return rows
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            commit = (r.get("commit") or "").strip()
            if not commit:
                continue
            try:
                score = float((r.get("val_bpb") or "").strip())
                score = score if math.isfinite(score) and score > 0 else None
            except ValueError:
                score = None
            rows[commit] = {
                "val_bpb": score,
                "status": (r.get("status") or "").strip(),
                "description": (r.get("description") or "").strip(),
            }
    return rows


def _growth_from_run(run: dict | None) -> dict | None:
    if not run or not run.get("prompts") or not run.get("snapshots"):
        return None
    return {
        "prompts": run["prompts"],
        "snapshots": [{"t_s": s.get("t_s"), "samples": s.get("samples", [])} for s in run["snapshots"]],
    }


class ModelStore:
    """The models on disk, each with the tokenizer of the pair it was trained with."""

    def __init__(self, device: str, entries: list[dict], active_path: str,
                 tokenizer_loader=None, runs_dir: str = "runs", results_path: str = "results.tsv"):
        self.results_path = results_path
        self._runs_dir = runs_dir
        self._device = device
        self._loader = tokenizer_loader or (lambda dataset, tokenizer: Tokenizer.from_directory(dataset=dataset, tokenizer=tokenizer))
        self._tokenizers: dict[tuple[str, str], object] = {}
        self._cache: dict[str, GPT] = {}
        self._lock = threading.Lock()
        self._runs: dict[str, dict | None] = {}
        scoreboard = _results_by_commit(results_path)

        self._entries = []
        for entry in entries:
            run = _run_for_checkpoint(entry["path"], runs_dir)
            pair = _pair_for_checkpoint(entry["path"], run)
            dataset, tokenizer_name = pair or (None, None)
            commit = _commit_from_name(entry["label"])
            row = scoreboard.get(commit or "", {})
            final = (run or {}).get("final") or {}
            own = run if run and "status" in run else None   # a record entry carries its own decision
            facts = {
                **entry,
                "commit": commit,
                "dataset": dataset,
                "dataset_blurb": DATASET_BLURBS.get(dataset, dataset),
                "tokenizer": tokenizer_name,
                "tokenizer_source": TOKENIZER_SOURCES.get(tokenizer_name, ""),
                "params_m": final.get("params_m"),
                "val_bpb": final.get("val_bpb") if own else row.get("val_bpb", final.get("val_bpb")),
                "status": (own.get("status") or "") if own else row.get("status", ""),
                "description": (own.get("description") or "") if own else row.get("description", ""),
                "has_growth": _growth_from_run(run) is not None,
            }
            if pair:
                facts["available"], facts["reason"], facts["vocab_size"] = self._check(facts)
            else:
                facts["available"], facts["vocab_size"] = False, None
                facts["reason"] = ("can't tell which dataset and tokenizer trained it: no record was saved beside it, "
                                   "and the active pair isn't the default")
            self._runs[entry["id"]] = run
            self._entries.append(facts)
        self._by_id = {e["id"]: e for e in self._entries}

        normalized_active = str(Path(active_path)).replace("\\", "/")
        usable = [e for e in self._entries if e["available"]]
        if not usable:
            raise RuntimeError("No usable checkpoint found. Run 'uv run prepare.py' and 'uv run train.py' first.")
        active = next((e for e in usable if e["path"] == normalized_active), usable[-1])
        self._active_id = active["id"]

    def _tokenizer(self, dataset: str, tokenizer_name: str):
        key = (dataset, tokenizer_name)
        if key not in self._tokenizers:
            try:
                self._tokenizers[key] = self._loader(dataset, tokenizer_name)
            except (FileNotFoundError, OSError, ValueError, KeyError):
                self._tokenizers[key] = None
        return self._tokenizers[key]

    def _check(self, facts: dict) -> tuple[bool, str, int | None]:
        tok = self._tokenizer(facts["dataset"], facts["tokenizer"])
        if tok is None:
            return False, (f"its tokenizer ({facts['dataset']}, {facts['tokenizer']}) is not prepared on this machine: "
                           f"uv run prepare.py --dataset {facts['dataset']} --tokenizer {facts['tokenizer']}"), None
        size = tok.get_vocab_size()
        rows = _checkpoint_vocab_rows(facts["path"])
        if rows is not None and rows != size:
            return False, f"its vocabulary has {rows:,} tokens but the {facts['tokenizer']} tokenizer has {size:,}", size
        return True, "", size

    @property
    def device(self) -> str:
        return self._device

    @property
    def active_id(self) -> str:
        return self._active_id

    def list_models(self) -> list[dict]:
        return [{**e, "active": e["id"] == self._active_id} for e in self._entries]

    def model_id_for_row(self, timestamp: str, commit: str) -> str | None:
        """The model a scoreboard row's run saved. A record run is named by its start time and version,
        so a row finds its own model even when two sessions share a version (every session starts from
        the same code). A row whose run kept no model (the first two of three baseline runs) and rows
        from before the record match by commit, preferring a model of this scoreboard's own session."""
        try:
            run_id = f"{compact_timestamp(datetime.fromisoformat(timestamp))}_{commit}"
        except ValueError:
            run_id = None
        same_commit = []
        for e in self._entries:
            if Path(e["label"]).stem == run_id:
                return e["id"]
            if commit and e["commit"] == commit:
                same_commit.append(e)
        own = [e for e in same_commit if (Path(self._runs_dir) / f"{Path(e['label']).stem}.json").exists()]
        return (own or same_commit or [{"id": None}])[0]["id"]

    def tokenizer_for_id(self, model_id: str | None):
        entry = self._by_id.get(model_id or self._active_id) or self._by_id[self._active_id]
        if not entry["available"]:
            entry = self._by_id[self._active_id]
        return self._tokenizer(entry["dataset"], entry["tokenizer"])

    def growth_for_id(self, model_id: str) -> dict | None:
        return _growth_from_run(self._runs.get(model_id))

    def get_bundle_by_id(self, model_id: str | None):
        with self._lock:
            entry = self._by_id.get(model_id or self._active_id)
            if entry is None:
                raise KeyError(f"Unknown model id: {model_id}")
            if not entry["available"]:
                raise ValueError(f"This model can't be loaded: {entry['reason']}.")
            model = self._cache.get(entry["path"])
            if model is None:
                model = _load_model_from_checkpoint(entry["path"], self._device, self._runs.get(entry["id"]))
                self._cache[entry["path"]] = model
            return model, self._tokenizer(entry["dataset"], entry["tokenizer"]), self._device, entry

    def get_active_bundle(self):
        return self.get_bundle_by_id(self._active_id)


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------

def _make_respond(model: GPT, tokenizer: Tokenizer, device: str):
    """Return a function that streams accumulated text and per-token strings."""
    eos_id = tokenizer.get_eos_token_id()

    @torch.no_grad()
    def respond(prompt: str, max_tokens: int, temperature: float, top_k: int):
        token_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
        idx = torch.tensor([token_ids], dtype=torch.long, device=device)
        prompt_len = len(token_ids)
        top_k_val = top_k if top_k > 0 else None
        tok_dicts: list[dict] = []  # running list of {id, text} per generated token

        amp_enabled = device == "cuda"
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp_enabled):
            for _ in range(max_tokens):
                ctx = (
                    idx
                    if idx.size(1) <= model.config.sequence_len
                    else idx[:, -model.config.sequence_len:]
                )
                logits = model(ctx)[:, -1, :]
                next_token = _sample_top_k(logits, top_k_val, temperature)
                idx = torch.cat((idx, next_token), dim=1)
                if next_token.item() == eos_id:
                    break
                tok_dicts.append({"id": next_token.item(), "text": tokenizer.decode([next_token.item()])})
                yield tokenizer.decode(idx[0, prompt_len:].tolist()), list(tok_dicts)

    return respond


# ---------------------------------------------------------------------------
# HTML page: self-contained but for Inter from Google Fonts (offline, the system font stands in).
# It wears the look of LLM Explorables' "How a Model Gets Made" page, whose "Use a model" step links
# here: the same colour tokens, light and dark, the same type and controls. A raw string, so the
# script's backslashes reach the browser as written.
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Use a model · autoresearch</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<script>
  // The theme before the page paints: the reader's own choice when they made one, else the system's.
  try {
    const t = localStorage.getItem('autoresearch.theme');
    if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
  } catch (e) { /* storage may be off */ }
</script>
<!-- Inter, as on the LLM Explorables pages; offline, the system font stands in. -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<style>
  /* The look of LLM Explorables' "How a Model Gets Made" page (llm-explorables training/style.css):
     its colour tokens, light and dark, Inter, hairline dividers, segmented controls. */
  :root {
    --bg: #ffffff; --bg-alt: #f6f6f7; --bg-soft: #f3f4f6;
    --text-1: #3c3c43; --text-2: #67676c; --text-3: #929295;
    --divider: #e2e2e3; --ink: #000000;
    --kept: #1a7f37; --worse: #b45309;
    --tok-1: #e8eefc; --tok-2: #fbe9f1; --tok-3: #e6f4ea; --tok-4: #fdf3d8;
    --tip-bg: rgba(20, 20, 22, 0.94); --tip-text: #f5f5f7;
    --switch-bg: rgba(142, 150, 170, 0.14); --switch-border: #c2c2c4; --knob: #ffffff;
    --font: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    --mono: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    color-scheme: light;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #1b1b1f; --bg-alt: #161618; --bg-soft: #202127;
      --text-1: #dfdfd6; --text-2: #98989f; --text-3: #6a6a71;
      --divider: #2e2e32; --ink: #ffffff;
      --kept: #3fb950; --worse: #f0a95a;
      --tok-1: #23293a; --tok-2: #33242c; --tok-3: #213028; --tok-4: #342d1d;
      --tip-bg: rgba(245, 245, 247, 0.96); --tip-text: #1b1b1f;
      --switch-bg: rgba(101, 117, 133, 0.16); --switch-border: #3c3f44; --knob: #000000;
      color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --bg: #1b1b1f; --bg-alt: #161618; --bg-soft: #202127;
    --text-1: #dfdfd6; --text-2: #98989f; --text-3: #6a6a71;
    --divider: #2e2e32; --ink: #ffffff;
    --kept: #3fb950; --worse: #f0a95a;
    --tok-1: #23293a; --tok-2: #33242c; --tok-3: #213028; --tok-4: #342d1d;
    --tip-bg: rgba(245, 245, 247, 0.96); --tip-text: #1b1b1f;
    --switch-bg: rgba(101, 117, 133, 0.16); --switch-border: #3c3f44; --knob: #000000;
    color-scheme: dark;
  }

  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body { margin: 0; font: 16px/1.6 var(--font); color: var(--text-1); background: var(--bg); -webkit-font-smoothing: antialiased; }
  a { color: inherit; }
  button, select, input, textarea { font: inherit; color: inherit; }
  :focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
  .num { font-variant-numeric: tabular-nums; }

  /* The top bar */
  .topbar { display: flex; align-items: center; gap: 16px; height: 64px; padding: 0 20px; border-bottom: 1px solid var(--divider); background: var(--bg); }
  .brand { display: flex; align-items: center; gap: 8px; min-width: 0; font-size: 16px; font-weight: 600; text-decoration: none; white-space: nowrap; }
  .brand .mark { flex: none; width: 24px; height: 24px; }
  .brand .mark .paper { fill: var(--bg); stroke: var(--ink); }
  .brand .mark .ink { fill: var(--ink); }
  .brand .sep { color: var(--text-3); font-weight: 400; }
  .brand .here { overflow: hidden; text-overflow: ellipsis; font-weight: 500; color: var(--text-2); }
  .bar { display: flex; align-items: center; gap: 4px; margin-left: auto; }
  .bar a { padding: 0 12px; font-size: 14px; font-weight: 500; line-height: 64px; text-decoration: none; white-space: nowrap; }
  .bar a:hover { color: var(--text-2); }
  .ext { width: 11px; height: 11px; margin-left: 2px; vertical-align: -1px; fill: none; stroke: var(--text-3); stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }
  .switch { position: relative; flex: none; width: 40px; height: 22px; margin-left: 8px; padding: 0; border: 1px solid var(--switch-border); border-radius: 11px; background: var(--switch-bg); cursor: pointer; }
  .knob { position: absolute; top: 1px; left: 1px; width: 18px; height: 18px; border-radius: 50%; background: var(--knob); box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08); transition: transform 0.25s; }
  .switch[aria-checked="true"] .knob { transform: translateX(18px); }
  @media (max-width: 719px) { .bar .wide { display: none; } .brand .home { display: none; } .brand .home + .sep { display: none; } }

  /* The page column */
  .page { max-width: 1240px; margin: 0 auto; padding: 36px 20px 72px; }
  .intro h1 { margin: 0; font-size: 28px; font-weight: 600; line-height: 1.3; letter-spacing: -0.02em; color: var(--ink); }
  .intro p { max-width: 70ch; margin: 10px 0 0; font-size: 17px; line-height: 28px; color: var(--text-2); }
  .intro p strong { font-weight: 600; color: var(--text-1); }
  .intro .public { font-size: 15px; line-height: 24px; }
  .intro a { color: var(--text-1); text-underline-offset: 3px; text-decoration-thickness: 1px; }

  .tabs { display: flex; gap: 24px; margin-top: 28px; overflow-x: auto; border-bottom: 1px solid var(--divider); scrollbar-width: none; }
  .tabs::-webkit-scrollbar { display: none; }
  .tab { flex: none; margin-bottom: -1px; padding: 10px 0 12px; border: 0; border-bottom: 2px solid transparent; background: none; font-size: 15px; font-weight: 500; color: var(--text-2); white-space: nowrap; cursor: pointer; }
  .tab:hover { color: var(--text-1); }
  .tab[aria-selected="true"] { border-bottom-color: var(--ink); color: var(--ink); }

  .panel { padding-top: 28px; }
  .question { max-width: 36ch; margin: 0; font-size: 22px; font-weight: 600; line-height: 1.3; letter-spacing: -0.02em; color: var(--ink); }
  .lede { max-width: 70ch; margin: 10px 0 0; line-height: 28px; }
  .hint { margin: 4px 0 0; font-size: 14px; line-height: 22px; color: var(--text-2); }
  .note { max-width: 72ch; margin: 14px 0 0; font-size: 14px; line-height: 22px; color: var(--text-2); }
  .eyebrow { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--text-2); }

  /* Controls, as on the explorables */
  .btn { padding: 6px 12px; border: 1px solid var(--divider); border-radius: 8px; background: var(--bg); font-size: 14px; font-weight: 500; cursor: pointer; }
  .btn:hover:not(:disabled) { border-color: var(--text-3); }
  .btn:disabled { color: var(--text-3); cursor: default; }
  .btn.primary { padding: 8px 18px; border-color: var(--ink); background: var(--ink); color: var(--bg); font-weight: 600; }
  .btn.primary:disabled { border-color: var(--divider); background: var(--bg-soft); color: var(--text-3); }
  .linkish { padding: 0; border: 0; background: none; font-size: 14px; font-weight: 500; color: var(--text-2); text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px; cursor: pointer; }
  .linkish:hover { color: var(--text-1); }
  .seg { display: inline-flex; flex-wrap: wrap; max-width: 100%; border: 1px solid var(--divider); border-radius: 8px; overflow: hidden; }
  .seg button { padding: 5px 12px; border: 0; border-left: 1px solid var(--divider); background: var(--bg); font-size: 14px; font-weight: 500; cursor: pointer; }
  .seg button:first-child { border-left: 0; }
  .seg button:hover:not([aria-pressed="true"]) { background: var(--bg-soft); }
  .seg button[aria-pressed="true"] { background: var(--ink); color: var(--bg); }
  .select-wrap { position: relative; display: inline-flex; min-width: 0; max-width: 100%; }
  .select-wrap select { width: 100%; max-width: 100%; padding: 7px 36px 7px 12px; border: 1px solid var(--divider); border-radius: 8px; background: var(--bg); font-size: 14px; font-weight: 500; color: var(--text-1); appearance: none; cursor: pointer; text-overflow: ellipsis; }
  .select-wrap select:hover { border-color: var(--text-3); }
  .select-wrap::after { content: ""; position: absolute; top: 50%; right: 14px; width: 7px; height: 7px; margin-top: -5px; border-right: 1.5px solid var(--text-2); border-bottom: 1.5px solid var(--text-2); transform: rotate(45deg); pointer-events: none; }
  input[type=range] { width: 100%; accent-color: var(--ink); cursor: pointer; }

  /* Continue text: the opening and the settings */
  .compose { padding: 18px 20px; border: 1px solid var(--divider); border-radius: 12px; background: var(--bg); }
  .compose label.eyebrow { display: block; margin-bottom: 8px; }
  .openings { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
  .openings .btn { padding: 4px 10px; font-size: 13px; }
  textarea { display: block; width: 100%; min-height: 84px; padding: 10px 12px; border: 1px solid var(--divider); border-radius: 8px; background: var(--bg); font-size: 16px; line-height: 1.55; resize: vertical; }
  textarea:hover { border-color: var(--text-3); }
  .compose-foot { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 14px 28px; margin-top: 14px; }
  .settings { display: grid; grid-template-columns: repeat(3, minmax(150px, 220px)); gap: 10px 28px; }
  .setting .top { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
  .setting .top label { font-size: 13px; font-weight: 600; color: var(--text-1); }
  .setting .value { font-size: 13px; color: var(--text-2); }
  .setting .range-hint { font-size: 12px; line-height: 18px; color: var(--text-3); }
  .go { display: flex; align-items: center; gap: 12px; }
  .go .hint { margin: 0; font-size: 13px; color: var(--text-3); }
  @media (max-width: 819px) { .settings { grid-template-columns: minmax(0, 1fr); width: 100%; } .go { width: 100%; justify-content: space-between; } }

  /* The two models, side by side */
  .panes { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; margin-top: 20px; }
  @media (max-width: 819px) { .panes { grid-template-columns: minmax(0, 1fr); } }
  .pane { display: flex; flex-direction: column; min-width: 0; padding: 16px 18px 18px; border: 1px solid var(--divider); border-radius: 12px; background: var(--bg); }
  .pane-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; }
  .pane-head .eyebrow { flex: 1 0 100%; }
  .pane-head .select-wrap { flex: 1 1 220px; }
  .pane-facts { margin: 8px 0 12px; font-size: 13px; line-height: 20px; color: var(--text-2); }
  .pane-facts b { font-weight: 600; color: var(--ink); }
  .output-box { flex: 1; min-height: 190px; padding-top: 12px; border-top: 1px solid var(--divider); font-size: 16px; line-height: 28px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .output-box .opening { font-weight: 600; color: var(--ink); }
  .output-box.empty { color: var(--text-3); font-size: 14px; }
  @keyframes blink { 50% { opacity: 0; } }
  .cursor { display: inline-block; width: 2px; height: 1.1em; margin-left: 1px; border-radius: 1px; background: var(--ink); vertical-align: text-bottom; animation: blink .65s step-end infinite; }
  @media (prefers-reduced-motion: reduce) { .cursor { animation: none; } .knob { transition: none; } }
  .token-box { flex: 1; min-height: 190px; display: flex; flex-wrap: wrap; align-content: flex-start; gap: 2px 0; padding-top: 12px; border-top: 1px solid var(--divider); }
  .tok { display: inline-flex; flex-direction: column; align-items: center; padding: 2px 1px; border-radius: 3px; }
  .tok:nth-child(4n+1) { background: var(--tok-1); } .tok:nth-child(4n+2) { background: var(--tok-2); }
  .tok:nth-child(4n+3) { background: var(--tok-3); } .tok:nth-child(4n+4) { background: var(--tok-4); }
  .tok-id { font: 600 10px/1.2 var(--mono); color: var(--text-2); }
  .tok-text { font: 13px/1.35 var(--mono); white-space: pre; color: var(--text-1); }

  /* Watch it learn */
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; margin-top: 20px; }
  .reading { max-width: 72ch; margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--divider); font-size: 17px; line-height: 30px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .reading .opening { font-weight: 600; color: var(--ink); }

  /* The session's chart, drawn as One session's on the explorables */
  .legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 18px 0 6px; padding: 0; list-style: none; font-size: 13px; color: var(--text-2); }
  .legend li { display: inline-flex; align-items: center; gap: 6px; }
  .legend svg { width: 22px; height: 16px; overflow: visible; }
  #chart { position: relative; min-width: 0; }
  #chart-svg { display: block; width: 100%; height: auto; overflow: visible; }
  .grid { stroke: var(--divider); stroke-width: 1; }
  .axis { stroke: var(--text-3); stroke-width: 1; }
  .tick { font-size: 12px; fill: var(--text-2); }
  .ytitle { font-size: 12px; font-weight: 500; fill: var(--text-2); }
  .stair { fill: none; stroke: var(--kept); stroke-width: 2; }
  .kept { fill: var(--kept); stroke: var(--kept); stroke-width: 1.5; }
  .aside { fill: var(--bg); stroke: var(--text-3); stroke-width: 2; }
  .clipped { fill: var(--text-3); }
  .ring-on { fill: none; stroke: var(--ink); stroke-width: 2; }
  .pt { cursor: default; }
  .pt .hit { fill: transparent; }
  .chart-tooltip { position: fixed; z-index: 1000; max-width: 260px; padding: 8px 11px; border-radius: 8px; background: var(--tip-bg); color: var(--tip-text); font-size: 13px; line-height: 20px; white-space: pre-line; pointer-events: none; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2); }

  /* The vocabulary */
  .vocab-search { display: block; width: 100%; max-width: 420px; margin-top: 18px; padding: 8px 12px; border: 1px solid var(--divider); border-radius: 8px; background: var(--bg); font-size: 15px; }
  .vocab-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(112px, 1fr)); gap: 4px; max-height: 600px; margin-top: 14px; overflow-y: auto; }
  .vocab-entry { padding: 5px 8px; border-radius: 6px; background: var(--bg-soft); overflow: hidden; }
  .vocab-entry-id { font: 600 11px var(--mono); color: var(--text-2); }
  .vocab-entry-text { display: block; overflow: hidden; font: 13px var(--mono); white-space: pre; text-overflow: ellipsis; color: var(--text-1); }
  .vocab-entry-text .empty { color: var(--text-3); }

  footer { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; max-width: 1240px; margin: 0 auto; padding: 18px 20px 28px; border-top: 1px solid var(--divider); font-size: 13px; color: var(--text-2); }
  footer a { text-underline-offset: 3px; }

  @media (max-width: 719px) {
    .page { padding: 24px 16px 56px; }
    .intro h1 { font-size: 24px; }
    .intro p { font-size: 16px; line-height: 26px; }
    .compose, .pane { padding: 14px; }
  }
</style>
</head>
<body>
<header class="topbar">
  <a class="brand" href="/">
    <svg class="mark" viewBox="0 0 64 64" aria-hidden="true"><rect class="paper" x="3" y="3" width="58" height="58" stroke-width="4"/><path class="ink" d="M14 48L32 12L50 48H42L32 28L22 48H14Z"/><rect class="ink" x="28" y="31" width="8" height="17"/></svg>
    <span class="home">autoresearch</span><span class="sep">/</span><span class="here">Use a model</span>
  </a>
  <div class="bar">
    <a class="wide" href="https://llmexplorables.aroughidea.com/training/">How a Model Gets Made <svg class="ext" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M9 7h8v8"/></svg></a>
    <a href="https://github.com/aroughidea/autoresearch-win-rtx">GitHub <svg class="ext" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M9 7h8v8"/></svg></a>
    <button class="switch" id="theme-switch" type="button" role="switch" aria-checked="false" aria-label="Dark theme"><span class="knob"></span></button>
  </div>
</header>

<div class="page">
  <section class="intro">
    <h1>Use a model</h1>
    <p>Give two small language models the same opening and read how each one continues it. A model continues text in the style of the text it learned from; these learned from <span id="dataset-blurb">short children’s stories</span>. Their writing appears one token at a time.</p>
    <!--PUBLIC-->
  </section>

  <nav class="tabs" role="tablist" aria-label="Views">
    <button class="tab" type="button" role="tab" id="tab-gen" aria-controls="page-gen" aria-selected="true">Continue text</button>
    <button class="tab" type="button" role="tab" id="tab-growth" aria-controls="page-growth" aria-selected="false" hidden>Watch it learn</button>
    <button class="tab" type="button" role="tab" id="tab-session" aria-controls="page-session" aria-selected="false">The session</button>
    <button class="tab" type="button" role="tab" id="tab-vocab" aria-controls="page-vocab" aria-selected="false">The vocabulary</button>
  </nav>

  <section class="panel" id="page-gen" role="tabpanel" aria-labelledby="tab-gen">
    <div class="compose">
      <label class="eyebrow" for="prompt">Opening</label>
      <div class="openings" id="openings"></div>
      <textarea id="prompt" rows="3" placeholder="Once upon a time"></textarea>
      <div class="compose-foot">
        <div class="settings">
          <div class="setting">
            <div class="top"><label for="max-tokens">Length</label><span class="value num"><span id="max-tokens-val">300</span> tokens</span></div>
            <input type="range" id="max-tokens" min="20" max="500" step="10" value="300">
            <div class="range-hint">Tokens are the pieces of text a model reads and writes</div>
          </div>
          <div class="setting">
            <div class="top"><label for="temperature">Temperature</label><span class="value num" id="temperature-val">0.80</span></div>
            <input type="range" id="temperature" min="0" max="2" step="0.05" value="0.8">
            <div class="range-hint">0 always takes the likeliest token; higher takes more chances</div>
          </div>
          <div class="setting">
            <div class="top"><label for="top-k">Top-k</label><span class="value num" id="top-k-val">40</span></div>
            <input type="range" id="top-k" min="0" max="200" step="5" value="40">
            <div class="range-hint">How many of the likeliest tokens it chooses among; 0 means all</div>
          </div>
        </div>
        <div class="go"><span class="hint">Ctrl + Enter</span><button class="btn primary" id="submit-btn" type="button">Generate</button></div>
      </div>
    </div>

    <div class="panes">
      <article class="pane" id="card-baseline">
        <div class="pane-head">
          <span class="eyebrow" id="role-baseline">Left</span>
          <div class="select-wrap"><select class="pane-pick" id="pick-baseline" aria-label="Left model"></select></div>
          <div class="seg" role="group" aria-label="Show the left model's output as">
            <button type="button" id="btn-text-baseline" aria-pressed="true">Text</button><button type="button" id="btn-tokens-baseline" aria-pressed="false">Tokens</button>
          </div>
          <button class="linkish" type="button" id="clear-baseline">Clear</button>
        </div>
        <div class="pane-facts" id="facts-baseline"></div>
        <div id="output-text-baseline" class="output-box empty" aria-live="polite">Its continuation appears here.</div>
        <div id="output-tokens-baseline" class="token-box" hidden></div>
      </article>
      <article class="pane" id="card-best">
        <div class="pane-head">
          <span class="eyebrow" id="role-best">Right</span>
          <div class="select-wrap"><select class="pane-pick" id="pick-best" aria-label="Right model"></select></div>
          <div class="seg" role="group" aria-label="Show the right model's output as">
            <button type="button" id="btn-text-best" aria-pressed="true">Text</button><button type="button" id="btn-tokens-best" aria-pressed="false">Tokens</button>
          </div>
          <button class="linkish" type="button" id="clear-best">Clear</button>
        </div>
        <div class="pane-facts" id="facts-best"></div>
        <div id="output-text-best" class="output-box empty" aria-live="polite">Its continuation appears here.</div>
        <div id="output-tokens-best" class="token-box" hidden></div>
      </article>
    </div>
    <p class="note" id="compare-note" hidden>These two learned from different datasets, so their scores don’t compare: each score measures how well a model predicts its own dataset. Compare them by what they write.</p>
  </section>

  <section class="panel" id="page-growth" role="tabpanel" aria-labelledby="tab-growth" hidden>
    <h2 class="question">What did a model write while it learned?</h2>
    <p class="lede">This is training time: one model, from its first step to the end of its run. Its run file kept what it wrote for the same openings at each moment, so you can read it learning.</p>
    <div class="controls">
      <div class="select-wrap"><select id="growth-model" aria-label="Model to watch"></select></div>
      <div class="select-wrap"><select id="growth-prompt" aria-label="Opening"></select></div>
      <div class="seg" id="growth-times" role="group" aria-label="Moment in training"></div>
    </div>
    <div id="growth-text" class="reading"></div>
    <p class="note">Every model wrote with the same openings and sampling settings, so the differences come from the model.</p>
  </section>

  <section class="panel" id="page-session" role="tabpanel" aria-labelledby="tab-session" hidden>
    <h2 class="question" id="chart-title">One night of an AI agent’s experiments</h2>
    <p class="lede">This is research time: each dot is a new model. The agent changes one thing in the training script, trains a model for the same few minutes and scores it on text kept out of training. A better score keeps the change; any other sets it aside. Lower is better.</p>
    <ul class="legend" aria-hidden="true">
      <li><svg viewBox="0 0 22 16"><circle class="kept" cx="11" cy="8" r="5"/></svg>Kept</li>
      <li><svg viewBox="0 0 22 16"><circle class="aside" cx="11" cy="8" r="5"/></svg>Set aside</li>
      <li><svg viewBox="0 0 22 16"><path class="stair" d="M1 11H11V5H21"/></svg>Best so far</li>
      <li><svg viewBox="0 0 22 16"><circle class="ring-on" cx="11" cy="8" r="6.5"/></svg>On show under Continue text</li>
      <li><svg viewBox="0 0 22 16"><path class="clipped" d="M11 2L17 13H5Z"/></svg>Off the scale</li>
    </ul>
    <div id="chart"><svg id="chart-svg" viewBox="0 0 900 320" role="img" aria-label="Score of each experiment"></svg></div>
    <p class="hint" id="chart-status">Loading the session…</p>
  </section>

  <section class="panel" id="page-vocab" role="tabpanel" aria-labelledby="tab-vocab" hidden>
    <h2 class="question">The <span id="vocab-count"></span> tokens the left model reads and writes</h2>
    <p class="lede">A model’s vocabulary is the list of pieces of text it can take in and produce, each with a number, its token id. Spaces are part of a token; ↵ is a new line.</p>
    <input class="vocab-search" id="vocab-search" type="search" placeholder="Find a token by its text or id" hidden>
    <div id="vocab-grid" class="vocab-grid"></div>
  </section>
</div>

<footer>
  <span>Scores are bits per byte on text kept out of training. Lower is better.</span>
  <a href="https://github.com/aroughidea/autoresearch-win-rtx/blob/master/TRAINING-DECISIONS.md">What the training decisions are</a>
</footer>
<div id="chart-tooltip" class="chart-tooltip" hidden></div>

<script>
  const $ = id => document.getElementById(id);

  let _models = [];
  let _results = [];
  let _baselineModelId = null;
  let _bestModelId     = null;
  const OPENINGS = ['Once upon a time', 'The old king said', 'In the dark forest', 'The little girl found a'];

  // ---- The theme switch ----
  function isDark() {
    const t = document.documentElement.dataset.theme;
    return t ? t === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function syncThemeSwitch() { $('theme-switch').setAttribute('aria-checked', isDark() ? 'true' : 'false'); }
  $('theme-switch').addEventListener('click', () => {
    const next = isDark() ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('autoresearch.theme', next); } catch (e) { /* storage may be off */ }
    syncThemeSwitch();
  });
  syncThemeSwitch();

  // ---- Tabs ----
  const PAGES = ['gen', 'growth', 'session', 'vocab'];
  function switchPage(p) {
    PAGES.forEach(q => {
      $('page-' + q).hidden = q !== p;
      $('tab-' + q).setAttribute('aria-selected', q === p ? 'true' : 'false');
    });
    if (p === 'vocab') loadVocab();
    if (p === 'session') drawChart();
  }
  PAGES.forEach(p => $('tab-' + p).addEventListener('click', () => switchPage(p)));

  // ---- Helpers ----
  function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function displayText(s) {
    return s.replace(/\n/g, '↵').replace(/\r/g, '↵').replace(/\t/g, '→');
  }
  const DATASET_NAMES = { tinystories: 'TinyStories', folktales: 'Folktales' };
  const datasetName = d => DATASET_NAMES[d] || d || '';
  const f4 = v => v.toFixed(4);

  // ---- The two panes ----
  function clearOutput(which) {
    const out = $('output-text-' + which);
    out.className = 'output-box empty';
    out.textContent = 'Its continuation appears here.';
    $('output-tokens-' + which).innerHTML = '';
  }
  function setView(v, which) {
    $('output-text-' + which).hidden = v !== 'text';
    $('output-tokens-' + which).hidden = v !== 'tokens';
    $('btn-text-' + which).setAttribute('aria-pressed', v === 'text' ? 'true' : 'false');
    $('btn-tokens-' + which).setAttribute('aria-pressed', v === 'tokens' ? 'true' : 'false');
  }
  ['baseline', 'best'].forEach(w => {
    $('btn-text-' + w).addEventListener('click', () => setView('text', w));
    $('btn-tokens-' + w).addEventListener('click', () => setView('tokens', w));
    $('clear-' + w).addEventListener('click', () => clearOutput(w));
  });
  function updateTokens(toks, which) {
    const box = $('output-tokens-' + which);
    box.innerHTML = '';
    toks.forEach(({ id, text }, i) => {
      const el = document.createElement('div');
      el.className = 'tok';
      el.innerHTML = '<span class="tok-id">' + id + '</span><span class="tok-text">' + escapeHtml(displayText(text)) + '</span>';
      el.title = 'token ' + (i + 1) + ' · id ' + id + ' · ' + JSON.stringify(text);
      box.appendChild(el);
    });
  }

  // ---- The opening and the settings ----
  $('max-tokens').oninput  = e => $('max-tokens-val').textContent  = e.target.value;
  $('temperature').oninput = e => $('temperature-val').textContent = parseFloat(e.target.value).toFixed(2);
  $('top-k').oninput       = e => $('top-k-val').textContent       = e.target.value;
  function setPrompt(text) { $('prompt').value = text; $('prompt').focus(); }
  function fillOpenings(list) {
    const box = $('openings');
    box.innerHTML = '';
    list.forEach(text => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'btn';
      b.textContent = text;
      b.onclick = () => setPrompt(text);
      box.appendChild(b);
    });
  }
  fillOpenings(OPENINGS);
  $('prompt').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); }
  });
  $('submit-btn').addEventListener('click', () => generate());

  async function loadModels() {
    const resp = await fetch('/models');
    if (!resp.ok) throw new Error('Could not load models');
    const payload = await resp.json();
    _models = payload.models || [];
  }

  // ---- The session's chart (research time) ----
  function chartRange(vals) {
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.06 || 0.01;
    const step = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1].find(st => (hi - lo + 2 * pad) / st <= 6) || 1;
    const y0 = Math.floor((lo - pad) / step) * step;
    const y1 = Math.ceil((hi + pad) / step) * step;
    const ticks = [];
    for (let v = y0; v < y1 + step / 2; v += step) ticks.push(+v.toFixed(6));
    return { y0, y1, ticks, step };
  }

  function drawChart() {
    const rows = _results;
    const svgEl = $('chart-svg');
    const status = $('chart-status');
    if (!rows.length) {
      svgEl.style.display = 'none';
      status.hidden = false;
      status.textContent = 'No experiments recorded yet. Start a session and train (see the README); each run appears here once it is kept or set aside.';
      return;
    }
    svgEl.style.display = 'block';
    const parsed = rows.map((r, i) => ({ ...r, n: i + 1 }));
    // The whole session stays in view, the start included. Only a score far above anything kept (a run
    // that failed) would flatten it: that is drawn at the top edge instead.
    const keptScores = parsed.filter(r => r.status === 'keep').map(r => r.val_bpb);
    const top = keptScores.length ? Math.max(...keptScores) : Math.max(...parsed.map(r => r.val_bpb));
    const clipHi = top * 1.5;
    const inView = parsed.filter(r => r.val_bpb <= clipHi);
    const { y0, y1, ticks, step } = chartRange(inView.map(r => r.val_bpb));

    const W = Math.max(320, Math.round($('chart').clientWidth || 900));
    const H = W < 560 ? 280 : 320;
    const m = { l: 48, r: 10, t: 22, b: 40 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const N = parsed.length;
    const x = n => m.l + ((n - 0.5) / N) * pw;
    const y = v => m.t + ((y1 - Math.min(y1, Math.max(y0, v))) / (y1 - y0)) * ph;
    const dec = Math.max(2, -Math.floor(Math.log10(step) + 1e-9));
    const dotR = N > 20 ? +Math.max(2.5, Math.min(6, (pw / N) * 0.42)).toFixed(2) : 6;
    // A model on show is ringed at the row that saved it: the last row naming it (the start's first two
    // runs kept no model of their own, so they name the third's).
    const shown = new Set([$('pick-baseline').value, $('pick-best').value]);
    const ringAt = new Set();
    const lastRow = {};
    for (const r of parsed) if (r.model_id) lastRow[r.model_id] = r.n;
    for (const id of shown) if (lastRow[id]) ringAt.add(lastRow[id]);

    let s = `<text class="ytitle" x="0" y="12">Score</text>`;
    for (const t of ticks) {
      s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}"/>`;
      s += `<text class="tick num" x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end">${t.toFixed(dec)}</text>`;
    }
    s += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${m.t + ph}" y2="${m.t + ph}"/>`;
    const every = N > 20 ? 5 : 1;
    for (const r of parsed) {
      if (r.n !== 1 && r.n % every) continue;
      s += `<text class="tick num" x="${x(r.n)}" y="${m.t + ph + 18}" text-anchor="middle">${r.n}</text>`;
    }
    s += `<text class="tick" x="${m.l + pw / 2}" y="${H - 2}" text-anchor="middle">Experiment</text>`;

    // The best kept score so far, as a stair.
    let best = Infinity;
    const stair = [];
    for (const r of parsed) {
      if (r.status === 'keep' && r.val_bpb < best) { best = r.val_bpb; stair.push([x(r.n), y(best)]); }
    }
    if (stair.length) {
      let d = `M${stair[0][0]},${stair[0][1]}`;
      for (let i = 1; i < stair.length; i++) d += ` H${stair[i][0]} V${stair[i][1]}`;
      s += `<path class="stair" d="${d} H${W - m.r}"/>`;
    }
    for (const r of parsed) {
      const cx = x(r.n);
      s += `<g class="pt" data-i="${r.n - 1}">`;
      if (r.val_bpb > clipHi) {
        s += `<path class="clipped" d="M${cx},${m.t - 2} l${dotR + 1},${(dotR + 1) * 1.8} h${-2 * (dotR + 1)}Z"/>`;
      } else {
        const cy = y(r.val_bpb);
        if (ringAt.has(r.n)) s += `<circle class="ring-on" cx="${cx}" cy="${cy}" r="${dotR + 4.5}"/>`;
        s += `<circle class="${r.status === 'keep' ? 'kept' : 'aside'}" cx="${cx}" cy="${cy}" r="${dotR}"/>`;
        s += `<circle class="hit" cx="${cx}" cy="${cy}" r="${Math.max(8, dotR + 4)}"/>`;
      }
      s += `</g>`;
    }
    svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svgEl.setAttribute('aria-label', `Score of each of the ${N} experiments, with the best score so far as a line`);
    svgEl.innerHTML = s;
    const keptN = parsed.filter(r => r.status === 'keep').length;
    status.hidden = false;
    status.textContent = `${N} experiments, ${keptN} kept. The best scored ${f4(best)}.`;

    svgEl.querySelectorAll('.pt').forEach(el => {
      const show = e => {
        const r = parsed[parseInt(el.dataset.i, 10)];
        const tip = $('chart-tooltip');
        const when = new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        tip.textContent = `Experiment ${r.n} · ${r.status === 'keep' ? 'kept' : 'set aside'}\n${r.description}\nScore ${r.val_bpb.toFixed(6)} · ${when}`;
        tip.hidden = false;
        const px = (e.touches ? e.touches[0] : e).clientX, py = (e.touches ? e.touches[0] : e).clientY;
        tip.style.left = Math.min(px + 14, window.innerWidth - 270) + 'px';
        tip.style.top = (py - 12) + 'px';
      };
      el.addEventListener('mousemove', show);
      el.addEventListener('touchstart', show, { passive: true });
      el.addEventListener('mouseleave', () => { $('chart-tooltip').hidden = true; });
    });
  }
  window.addEventListener('resize', () => { if (!$('page-session').hidden) drawChart(); });
  document.addEventListener('touchstart', e => { if (!e.target.closest('.pt')) $('chart-tooltip').hidden = true; }, { passive: true });

  async function loadResults() {
    try {
      const resp = await fetch('/results');
      if (!resp.ok) throw new Error(resp.statusText);
      const { rows, skipped } = await resp.json();
      _results = rows;
      if (skipped) $('chart-status').textContent = skipped + ' row(s) of results.tsv could not be drawn and were left out.';

      // The start (the session's first kept row) and its best (lowest kept score from the same dataset).
      const kept = rows.filter(r => r.status === 'keep');
      if (kept.length) {
        $('card-best').hidden = false;
        const first = kept[0];
        // Scores compare only within one dataset, so "best" comes from the start's.
        // Each row names its own model: by commit alone, two sessions that started from the same code
        // would share one.
        const datasetOf = r => (_models.find(m => m.id === r.model_id) || {}).dataset;
        const sameData = kept.filter(r => datasetOf(r) && datasetOf(r) === datasetOf(first));
        const best  = (sameData.length ? sameData : kept).reduce((a, b) => a.val_bpb <= b.val_bpb ? a : b);
        _baselineModelId = first.model_id || null;
        _bestModelId     = best.model_id  || null;
        setupPickers(_baselineModelId, _bestModelId);
      } else {
        _baselineModelId = null;
        _bestModelId     = null;
        setupPickers(null, null);
        $('card-best').hidden = true;
      }
      const ds = datasetName(((_models.find(m => m.id === _baselineModelId)) || {}).dataset);
      if (ds) $('chart-title').textContent = `One night of an AI agent’s experiments on ${ds}`;
    } catch (err) {
      $('chart-status').textContent = 'Could not load the session: ' + err.message;
    }
  }

  function factsText(m) {
    const parts = [];
    if (m.val_bpb != null) parts.push('Score <b>' + f4(m.val_bpb) + '</b> (lower is better)');
    parts.push(datasetName(m.dataset));
    if (m.vocab_size) parts.push((m.tokenizer === 'own' ? 'its own vocabulary' : m.tokenizer + ' vocabulary') + ', ' + m.vocab_size.toLocaleString() + ' tokens');
    if (m.params_m) parts.push(m.params_m + ' M parameters');
    return parts.join(' · ');
  }
  function roleOf(m) {
    if (!m) return '';
    if (m.id === _baselineModelId) return 'The start';
    if (m.id === _bestModelId) return 'The best';
    return (m.description || '').startsWith('baseline') ? 'A start' : 'An experiment';
  }
  function modelName(m) {
    const role = m.id === _baselineModelId ? 'the start' : m.id === _bestModelId ? 'the best'
      : (m.description || '').startsWith('baseline') ? 'a start' : (m.description || m.label);
    return datasetName(m.dataset) + ' · ' + role;
  }
  function fillPick(sel, chosen) {
    sel.innerHTML = '';
    _models.forEach(m => {
      const o = document.createElement('option');
      o.value = m.id;
      o.textContent = modelName(m) + (m.available ? '' : ' (can’t load)');
      o.disabled = !m.available;
      o.title = m.available ? factsText(m).replace(/<\/?b>/g, '') : m.reason;
      sel.appendChild(o);
    });
    if (chosen) sel.value = chosen;
  }
  function showFacts(which) {
    const m = _models.find(x => x.id === $('pick-' + which).value);
    $('facts-' + which).innerHTML = m ? factsText(m) : '';
    $('role-' + which).textContent = roleOf(m) || (which === 'baseline' ? 'Left' : 'Right');
    if (which === 'baseline' && m) { $('dataset-blurb').textContent = m.dataset_blurb; _vocabData = null; }
    updateCompareNote();
  }
  function updateCompareNote() {
    const a = _models.find(x => x.id === $('pick-baseline').value);
    const b = _models.find(x => x.id === $('pick-best').value);
    $('compare-note').hidden = !(a && b && a.dataset !== b.dataset && !$('card-best').hidden);
  }
  function setupPickers(leftId, rightId) {
    const usable = _models.filter(m => m.available);
    const fallback = (usable[usable.length - 1] || {}).id;
    fillPick($('pick-baseline'), leftId || fallback);
    fillPick($('pick-best'), rightId || fallback);
    ['baseline', 'best'].forEach(w => {
      $('pick-' + w).onchange = () => { showFacts(w); if (!$('page-session').hidden) drawChart(); };
      showFacts(w);
    });
    setupGrowth();
  }

  // ---- Watch it learn (training time) ----
  let _growth = null;
  function momentLabel(t) {
    if (t < 60) return Math.round(t) + ' s';
    const min = t / 60;
    return (Math.abs(min - Math.round(min)) < 0.05 ? Math.round(min) : min.toFixed(1)) + ' min';
  }
  async function setupGrowth() {
    const withGrowth = _models.filter(m => m.has_growth && m.available);
    $('tab-growth').hidden = !withGrowth.length;
    if (!withGrowth.length) return;
    const sel = $('growth-model');
    sel.innerHTML = '';
    withGrowth.forEach(m => { const o = document.createElement('option'); o.value = m.id; o.textContent = modelName(m); sel.appendChild(o); });
    if (withGrowth.some(m => m.id === _bestModelId)) sel.value = _bestModelId;
    sel.onchange = loadGrowth;
    $('growth-prompt').onchange = () => showGrowth(_growthAt);
    await loadGrowth();
  }
  let _growthAt = 0;
  async function loadGrowth() {
    _growth = await (await fetch('/growth?model_id=' + encodeURIComponent($('growth-model').value))).json();
    if (_growth.prompts.length) fillOpenings(_growth.prompts);
    const p = $('growth-prompt');
    const keep = p.value;
    p.innerHTML = '';
    _growth.prompts.forEach((text, i) => { const o = document.createElement('option'); o.value = i; o.textContent = '“' + text + '”'; p.appendChild(o); });
    if (keep && +keep < _growth.prompts.length) p.value = keep;
    const times = $('growth-times');
    times.innerHTML = '';
    _growth.snapshots.forEach((sn, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = momentLabel(sn.t_s);
      b.onclick = () => showGrowth(i);
      times.appendChild(b);
    });
    showGrowth(_growth.snapshots.length - 1);
  }
  function showGrowth(i) {
    if (!_growth || !_growth.snapshots.length) return;
    _growthAt = i;
    const prompt = parseInt($('growth-prompt').value || '0', 10);
    Array.from($('growth-times').children).forEach((b, j) => b.setAttribute('aria-pressed', j === i ? 'true' : 'false'));
    const box = $('growth-text');
    box.innerHTML = '<span class="opening">' + escapeHtml(_growth.prompts[prompt]) + '</span>' + escapeHtml(_growth.snapshots[i].samples[prompt] || '');
  }

  // ---- The vocabulary ----
  let _vocabData = null;
  async function loadVocab() {
    if (_vocabData) return;
    const grid = $('vocab-grid');
    grid.innerHTML = '<p class="hint">Loading…</p>';
    const { entries } = await (await fetch('/vocab?model_id=' + encodeURIComponent($('pick-baseline').value))).json();
    _vocabData = entries;
    $('vocab-count').textContent = entries.length.toLocaleString();
    $('vocab-search').hidden = false;
    renderVocab(entries);
  }
  function renderVocab(entries) {
    const frag = document.createDocumentFragment();
    entries.forEach(({ id, text }) => {
      const el = document.createElement('div');
      el.className = 'vocab-entry';
      el.title = 'id ' + id + ' · ' + JSON.stringify(text);
      const disp = escapeHtml(displayText(text)) || '<span class="empty">∅</span>';
      el.innerHTML = '<div class="vocab-entry-id">' + id + '</div><span class="vocab-entry-text">' + disp + '</span>';
      frag.appendChild(el);
    });
    const grid = $('vocab-grid');
    grid.innerHTML = '';
    grid.appendChild(frag);
  }
  function filterVocab() {
    if (!_vocabData) return;
    const q = $('vocab-search').value.toLowerCase();
    if (!q) { renderVocab(_vocabData); return; }
    renderVocab(_vocabData.filter(({ id, text }) => String(id).includes(q) || text.toLowerCase().includes(q)));
  }
  $('vocab-search').addEventListener('input', filterVocab);

  // ---- Generating ----
  async function runGenerate(prompt, modelId, which) {
    const output = $('output-text-' + which);
    output.className = 'output-box';
    output.textContent = '';
    $('output-tokens-' + which).innerHTML = '';
    const opening = document.createElement('span');
    opening.className = 'opening';
    opening.textContent = prompt;
    const written = document.createTextNode('');
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    output.append(opening, written, cursor);

    try {
      const body = {
        prompt,
        max_tokens:  parseInt($('max-tokens').value, 10),
        temperature: parseFloat($('temperature').value),
        top_k:       parseInt($('top-k').value, 10),
      };
      if (modelId) body.model_id = modelId;
      const resp = await fetch('/generate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) { output.textContent = 'The server could not continue it: ' + resp.statusText; return; }
      const reader  = resp.body.getReader();
      const decoder = new TextDecoder();
      let carry = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        carry += decoder.decode(value, { stream: true });
        const lines = carry.split('\n');
        carry = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const { t, toks } = JSON.parse(line);
            written.data = t;
            updateTokens(toks, which);
          } catch (_) { /* a partial line */ }
        }
      }
    } finally {
      cursor.remove();
    }
  }

  async function generate() {
    const prompt = $('prompt').value.trim();
    if (!prompt) { $('prompt').focus(); return; }
    const btn = $('submit-btn');
    btn.disabled = true;
    btn.textContent = 'Writing…';
    try {
      const runs = [runGenerate(prompt, $('pick-baseline').value, 'baseline')];
      if (!$('card-best').hidden) runs.push(runGenerate(prompt, $('pick-best').value, 'best'));
      await Promise.all(runs);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Generate';
    }
  }

  // ---- Start ----
  loadModels()
    .then(() => loadResults())
    .catch(err => { $('chart-status').textContent = 'The page could not start: ' + err; });
</script>
</body>
</html>
"""

# Public-demo mode (see the note at the top of this file). Everything gated on
# this flag is inert when the env var is unset, which is the local default.
_SPACE_MODE = os.environ.get("SPACE_MODE") == "1"
_SPACE_BANNER = (
    '<p class="public">A public demo of four small language models, each trained for 10 minutes on one '
    "laptop graphics card. For two datasets it serves the starter script and the best version an AI agent "
    "found in a night of experiments. On text kept out of training the best scored 63% better on Folktales "
    "(3.287 to 1.207 bits per byte, from the average of its three start runs) and 2.6% better on TinyStories "
    "(0.4366 to 0.4251); lower is better. Compare a start with its best, or the two starts (the same code, "
    "trained on different text) to see what the dataset decides. "
    '<a href="https://llmexplorables.aroughidea.com/training/">How a Model Gets Made</a> walks through how '
    'they were made, and <a href="https://github.com/aroughidea/autoresearch-win-rtx/blob/master/README.md#start-here-non-technical">'
    "the README</a> explains the project.</p>"
)

# The A Rough Idea mark, as LLM Explorables' pages carry it, served at /favicon.svg; /favicon.ico
# answers the same, so a browser that asks for it by habit gets no 404.
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64" fill="none">'
    '<rect x="3" y="3" width="58" height="58" fill="white" stroke="black" stroke-width="4"/>'
    '<path d="M14 48L32 12L50 48H42L32 28L22 48H14Z" fill="black"/>'
    '<rect x="28" y="31" width="8" height="17" fill="black"/></svg>'
)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(model_store: ModelStore) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def root():
        # On a public URL the banner says which models these are; locally it is left out.
        return _HTML.replace("<!--PUBLIC-->", _SPACE_BANNER if _SPACE_MODE else "", 1)

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 500
        temperature: float = 0.0
        top_k: int = 0
        model_id: str | None = None

    @app.post("/generate")
    def generate(req: GenerateRequest):
        if _SPACE_MODE:
            # Public-site guards: silently clamp, never error.
            req.prompt = req.prompt[:2000]
            req.max_tokens = min(req.max_tokens, 500)
            req.top_k = min(req.top_k, 200)
        try:
            model, tokenizer, device, _ = model_store.get_bundle_by_id(req.model_id)
        except KeyError:
            model, tokenizer, device, _ = model_store.get_active_bundle()
        except ValueError as exc:
            message = str(exc)
            return StreamingResponse(iter([json.dumps({"t": message, "toks": []}) + "\n"]), media_type="text/plain")
        respond = _make_respond(model, tokenizer, device)

        def stream():
            for text, toks in respond(req.prompt, req.max_tokens, req.temperature, req.top_k):
                yield json.dumps({"t": text, "toks": toks}) + "\n"
        return StreamingResponse(stream(), media_type="text/plain")

    @app.get("/vocab")
    def vocab(model_id: str | None = None):
        tokenizer = model_store.tokenizer_for_id(model_id)
        n = tokenizer.get_vocab_size()
        return {"entries": [{"id": i, "text": tokenizer.decode([i])} for i in range(n)]}

    @app.get("/growth")
    def growth(model_id: str):
        return model_store.growth_for_id(model_id) or {"prompts": [], "snapshots": []}

    @app.get("/models")
    def models():
        return {"models": model_store.list_models()}

    @app.get("/favicon.svg", include_in_schema=False)
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(content=_FAVICON_SVG, media_type="image/svg+xml")

    @app.get("/results")
    def results_data():
        path = Path(model_store.results_path)
        if not path.exists():
            return {"rows": [], "skipped": 0}
        rows = []
        skipped = 0
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                timestamp = (r.get("timestamp") or "").strip()
                raw_bpb = (r.get("val_bpb") or "").strip()
                try:
                    val_bpb = float(raw_bpb)
                    datetime.fromisoformat(timestamp)
                except ValueError:
                    skipped += 1
                    continue
                if not math.isfinite(val_bpb) or val_bpb <= 0:
                    skipped += 1
                    continue
                try:
                    memory_gb = float((r.get("memory_gb") or "").strip())
                except ValueError:
                    memory_gb = None
                commit = (r.get("commit") or "").strip()
                rows.append({
                    "timestamp": timestamp,
                    "commit": commit,
                    "model_id": model_store.model_id_for_row(timestamp, commit),
                    "val_bpb": val_bpb,
                    "memory_gb": memory_gb,
                    "status": (r.get("status") or "unknown").strip(),
                    "description": (r.get("description") or "").strip(),
                })
        return {"rows": rows, "skipped": skipped}

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_free_port(start: int) -> int:
    """Return the first free TCP port >= start."""
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {start}–{start + 19}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local browser UI for the trained model.")
    parser.add_argument("--checkpoint", default="checkpoint_pre_eval.pt", help="Path to .pt checkpoint")
    parser.add_argument("--port",       type=int, default=8000,           help="Local port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true",              help="Do not open a browser tab automatically")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    entries = _discover_checkpoints(args.checkpoint)
    runs_dir, results_path = _session_paths()  # an active session's runs and scoreboard live in its folder
    try:
        model_store = ModelStore(device=device, entries=entries, active_path=args.checkpoint,
                                 runs_dir=runs_dir, results_path=results_path)
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)
    for m in model_store.list_models():
        if not m["available"]:
            print(f"Skipping {m['label']}: {m['reason']}.")
    model_store.get_active_bundle()  # load the default model before the page opens
    print("Model ready.\n")
    app = build_app(model_store)

    if _SPACE_MODE:
        host = "0.0.0.0"
        port = int(os.environ.get("PORT", "7860"))
    else:
        host = "127.0.0.1"
        port = _find_free_port(args.port)
        if port != args.port:
            print(f"Port {args.port} in use, using {port} instead.")

    if not args.no_browser:
        def _open():
            time.sleep(1.2)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
