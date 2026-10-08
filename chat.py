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
# pointing at TRAINING-DECISIONS.md and the walkthrough. Unset (the default), nothing below changes.
# See deploy/ for the container that runs it.

import argparse
import base64
import csv
import json
import math
import re
import os
import socket
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from generate import _config_from_state_dict, _sample_top_k
from capture import read_checkpoint_pair
from prepare import (
    DEFAULT_DATASET,
    DEFAULT_TOKENIZER,
    TOKENIZER_SOURCES,
    Tokenizer,
    _resolve_dataset_name,
    _resolve_tokenizer_name,
)
from train import GPT


# ---------------------------------------------------------------------------
# Load model (startup + on-demand switch)
# ---------------------------------------------------------------------------

def _load_model_from_checkpoint(checkpoint_path: str, device: str) -> GPT:
  state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
  config = _config_from_state_dict(state_dict)
  model = GPT(config)
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
    """The run file saved with a checkpoint: same <timestamp>_<commit> stem."""
    run_path = Path(runs_dir) / (Path(path).stem + ".json")
    try:
        return json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


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
            facts = {
                **entry,
                "commit": commit,
                "dataset": dataset,
                "dataset_blurb": DATASET_BLURBS.get(dataset, dataset),
                "tokenizer": tokenizer_name,
                "tokenizer_source": TOKENIZER_SOURCES.get(tokenizer_name, ""),
                "params_m": final.get("params_m"),
                "val_bpb": row.get("val_bpb", final.get("val_bpb")),
                "status": row.get("status", ""),
                "description": row.get("description", ""),
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
                model = _load_model_from_checkpoint(entry["path"], self._device)
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
# HTML page (self-contained, no external dependencies)
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>autoresearch \u2014 local text generation</title>
<style>
  :root {
    --bg: #f3f4f6;
    --surface: #ffffff;
    --text: #111827;
    --muted: #6b7280;
    --muted-strong: #4b5563;
    --border: #d1d5db;
    --border-strong: #9ca3af;
    --accent: #111827;
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.06), 0 0 0 1px rgba(17, 24, 39, 0.04);
    --focus-ring: 0 0 0 3px rgba(37, 99, 235, 0.2);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 18px 18px 28px;
  }

  .container { width: 100%; max-width: 1320px; margin: 0 auto; }

  /* ---- Header ---- */
  .header { margin-bottom: 14px; }
  .header h1 { font-size: 1.35rem; font-weight: 700; margin-bottom: 8px; }
  .header p  { font-size: 0.9rem; color: var(--muted); line-height: 1.55; max-width: 1000px; }
  .header strong { color: var(--text); }

  /* ---- Cards ---- */
  .card {
    background: var(--surface);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 10px;
    box-shadow: var(--shadow);
  }
  .card-label {
    font-size: 0.73rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--muted);
    margin-bottom: 12px;
  }

  /* ---- Model timeline ---- */
  .model-active {
    font-size: 0.82rem;
    color: var(--muted-strong);
    margin-bottom: 10px;
    line-height: 1.4;
  }
  .timeline-wrap {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 4px;
  }
  .timeline-node {
    min-width: 190px;
    border: 1.5px solid var(--border);
    background: #fafafa;
    border-radius: 8px;
    padding: 8px 10px;
    text-align: left;
    color: var(--text);
  }
  .timeline-node:hover:not(:disabled) {
    border-color: var(--border-strong);
    background: #f3f4f6;
  }
  .timeline-node.active {
    border-color: #059669;
    background: #ecfdf5;
    box-shadow: inset 0 0 0 1px rgba(5, 150, 105, 0.25);
  }
  .timeline-node:disabled {
    opacity: 0.65;
    cursor: wait;
  }
  .timeline-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }
  .timeline-idx {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--muted);
  }
  .timeline-time {
    font-size: 0.72rem;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .timeline-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* ---- Parameters ---- */
  .params { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  .param-row {
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 0.86rem; margin-bottom: 4px;
  }
  .param-row .name  { font-weight: 600; }
  .param-row .value {
    font-size: 0.82rem; font-variant-numeric: tabular-nums;
    background: #f3f4f6; padding: 2px 7px; border-radius: 4px;
  }
  input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }
  .param-hint { font-size: 0.76rem; color: var(--muted); margin-top: 5px; line-height: 1.45; }

  /* ---- Prompt ---- */
  textarea {
    width: 100%; min-height: 100px;
    font-size: 0.96rem; font-family: inherit; line-height: 1.55;
    padding: 10px 12px;
    border: 1.5px solid var(--border); border-radius: 8px;
    resize: vertical; outline: none;
    transition: border-color .15s, box-shadow .15s;
  }
  textarea:focus { border-color: var(--border-strong); box-shadow: var(--focus-ring); }
  .prompt-footer {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 10px;
  }
  .prompt-hint { font-size: 0.78rem; color: var(--muted); }

  button {
    padding: 10px 20px;
    background: var(--accent); color: #fff;
    border: none; border-radius: 6px;
    font-size: 0.9rem; font-weight: 600;
    cursor: pointer; transition: background .15s, transform .05s;
  }
  button:hover:not(:disabled) { background: #374151; }
  button:active:not(:disabled) { transform: translateY(1px); }
  button:disabled { background: #c0c0c0; cursor: not-allowed; }
  button:focus-visible,
  input:focus-visible {
    outline: none;
    box-shadow: var(--focus-ring);
  }

  /* ---- Output ---- */
  .output-box {
    min-height: 120px;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.04rem; line-height: 1.8;
    white-space: pre-wrap;
  }
  .output-box.empty {
    font-family: inherit; font-size: 0.87rem;
    color: var(--muted); font-style: italic;
  }
  @keyframes blink { 50% { opacity: 0; } }
  .cursor {
    display: inline-block; width: 2px; height: 1.1em;
    background: #1a1a1a; border-radius: 1px;
    vertical-align: text-bottom; margin-left: 1px;
    animation: blink .65s step-end infinite;
  }

  /* ---- Output view toggle ---- */
  .card-label-row {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px;
  }
  .card-label-row .card-label { margin-bottom: 0; }
  .view-toggle { display: flex; gap: 4px; }
  .toggle-btn {
    padding: 3px 12px; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .05em;
    background: transparent; color: var(--muted);
    border: 1.5px solid var(--border); border-radius: 5px;
    cursor: pointer; transition: all .15s;
  }
  .toggle-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .clear-btn {
    padding: 3px 12px; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .05em;
    background: transparent; color: var(--muted);
    border: 1.5px solid var(--border); border-radius: 5px;
    cursor: pointer; transition: all .15s;
  }
  .clear-btn:hover { color: #b91c1c; border-color: #b91c1c; }

  /* ---- Token chips ---- */
  .token-box {
    min-height: 100px;
    display: flex; flex-wrap: wrap; gap: 0; align-content: flex-start;
  }
  .tok {
    display: inline-flex; flex-direction: column; align-items: center;
    padding: 2px 0; border-radius: 3px; cursor: default; min-width: 0;
  }
  .tok-id {
    font-family: 'Courier New', monospace; font-size: 0.58rem;
    font-weight: 700; color: var(--muted-strong); line-height: 1.2;
  }
  .tok-text {
    font-family: 'Courier New', monospace; font-size: 0.78rem;
    white-space: pre; line-height: 1.3; color: #1a1a1a;
  }

  /* ---- Page navigation ---- */
  .page-nav {
    display: flex; margin-bottom: 12px;
    border-bottom: 2px solid var(--border);
  }
  .page-tab {
    padding: 10px 22px; background: transparent; border: none;
    font-size: 0.88rem; font-weight: 600; color: var(--muted);
    cursor: pointer; border-bottom: 2px solid transparent;
    margin-bottom: -2px; transition: color .15s, border-color .15s;
  }
  .page-tab.active { color: var(--text); border-bottom-color: var(--accent); }
  .page-tab:hover:not(.active) { color: var(--muted-strong); }

  /* ---- Generator layout ---- */
  #page-gen { display: flex; flex-direction: column; gap: 12px; }
  .gen-bottom {
    display: grid;
    grid-template-columns: minmax(320px, 380px) minmax(0, 1fr);
    gap: 12px;
    align-items: start;
  }
  .gen-left { display: flex; flex-direction: column; gap: 10px; }
  .gen-left .card { margin-bottom: 0; }
  .gen-outputs { display: flex; flex-direction: column; gap: 10px; }
  .card-output { display: flex; flex-direction: column; margin-bottom: 0; }
  .card-output .card-label-row { flex-shrink: 0; }
  .card-output .output-box { flex: 1; min-height: 180px; }
  .card-output .token-box  { flex: 1; min-height: 180px; }

  /* ---- Progress chart ---- */
  .chart-svg { width: 100%; display: block; overflow: visible; }
  .chart-tooltip {
    position: fixed; pointer-events: none;
    background: rgba(17,24,39,0.93); color: #f9fafb;
    font-size: 0.78rem; line-height: 1.55;
    padding: 8px 11px; border-radius: 7px;
    max-width: 240px; z-index: 1000;
    display: none; white-space: pre-line;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
  }

  /* ---- Vocabulary browser ---- */
  .vocab-search {
    display: block; width: 100%; padding: 8px 12px;
    font-size: 0.9rem; font-family: inherit;
    border: 1.5px solid var(--border); border-radius: 8px; margin-bottom: 16px;
    outline: none; transition: border-color .15s;
  }
  .vocab-search:focus { border-color: var(--border-strong); box-shadow: var(--focus-ring); }
  .vocab-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
    gap: 5px; max-height: 580px; overflow-y: auto;
  }
  .vocab-entry {
    background: #f8f8f8; border-radius: 4px; padding: 5px 8px; overflow: hidden;
  }
  .vocab-entry-id {
    font-family: 'Courier New', monospace; font-size: 0.7rem;
    font-weight: 700; color: var(--muted);
  }
  .vocab-entry-text {
    font-family: 'Courier New', monospace; font-size: 0.82rem;
    white-space: pre; overflow: hidden; text-overflow: ellipsis;
    display: block; color: #333;
  }

  @media (max-width: 1080px) {
    .gen-bottom { grid-template-columns: 1fr; }
    .card-output .output-box,
    .card-output .token-box {
      min-height: 200px;
    }
  }

  @media (max-width: 760px) {
    body { padding: 14px 12px 20px; }
    .card { padding: 14px; }
    .params { grid-template-columns: 1fr; gap: 12px; }
    .page-tab { padding: 10px 12px; }
    .prompt-footer { gap: 12px; align-items: flex-start; flex-direction: column; }
  }
  .pane-pick { font: inherit; font-size: 0.85rem; font-weight: 600; max-width: 100%; padding: 4px 6px;
               border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--text); }
  .pane-facts { font-size: 0.78rem; color: var(--muted); margin: -4px 0 8px; }
  .compare-note { font-size: 0.82rem; color: var(--muted); padding: 0 4px; }
  .growth-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }
  .growth-times { display: flex; flex-wrap: wrap; gap: 4px; }
  .growth-note { font-size: 0.78rem; color: var(--muted); margin: 8px 0 0; }
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>autoresearch &mdash; local text generation</h1>
    <p>
      This is a <strong>text completion model</strong> trained on <span id="dataset-blurb">short children&rsquo;s stories</span>. It is not a chat assistant &mdash; it continues whatever you type in the same
      writing style. Type the start of a sentence or story and press
      <strong>Generate</strong>. Output appears token&nbsp;by&nbsp;token in real time.
    </p>
  </div>

  <nav class="page-nav" role="tablist" aria-label="Main views">
    <button class="page-tab active" id="tab-gen" role="tab" aria-selected="true" aria-controls="page-gen" onclick="switchPage('gen')">Generator</button>
    <button class="page-tab" id="tab-vocab" role="tab" aria-selected="false" aria-controls="page-vocab" onclick="switchPage('vocab')">Vocabulary</button>
  </nav>

  <div id="page-gen" role="tabpanel" aria-labelledby="tab-gen">

  <div class="card">
    <div id="chart-title" class="card-label">The agent's experiments (research time: a new model each run)</div>
    <svg id="chart-svg" class="chart-svg" viewBox="0 0 900 240" preserveAspectRatio="xMidYMid meet"></svg>
    <div id="chart-status" style="font-size:0.78rem;color:var(--muted);margin-top:4px">Loading results\u2026</div>
  </div>

  <div class="gen-bottom">

  <div class="gen-left">

  <div class="card">
    <div class="card-label">Model parameters</div>
    <div class="params">

      <div>
        <div class="param-row">
          <span class="name">Max tokens</span>
          <span class="value" id="max-tokens-val">500</span>
        </div>
        <input type="range" id="max-tokens" min="20" max="500" step="10" value="500">
        <div class="param-hint">How many tokens (word pieces) to generate</div>
      </div>

      <div>
        <div class="param-row">
          <span class="name">Temperature</span>
          <span class="value" id="temperature-val">0.00</span>
        </div>
        <input type="range" id="temperature" min="0" max="2" step="0.01" value="0">
        <div class="param-hint">0 = predictable &nbsp;&middot;&nbsp; 2 = chaotic</div>
      </div>

      <div>
        <div class="param-row">
          <span class="name">Top-k</span>
          <span class="value" id="top-k-val">0</span>
        </div>
        <input type="range" id="top-k" min="0" max="200" step="5" value="0">
        <div class="param-hint">Next-token choices considered &mdash; 0 means all</div>
      </div>

    </div>
  </div>

  <div class="card">
    <div class="card-label">Prompt</div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
      <button class="clear-btn" style="font-size:0.8rem;padding:4px 10px;text-transform:none;letter-spacing:normal" onclick="setPrompt('Once upon a time,')">Once upon a time,</button>
      <button class="clear-btn" style="font-size:0.8rem;padding:4px 10px;text-transform:none;letter-spacing:normal" onclick="setPrompt('The little fox looked up and said,')">The little fox looked up and said,</button>
      <button class="clear-btn" style="font-size:0.8rem;padding:4px 10px;text-transform:none;letter-spacing:normal" onclick="setPrompt('Deep in the forest there lived a')">Deep in the forest there lived a</button>
    </div>
    <textarea id="prompt" placeholder="Once upon a time\u2026"></textarea>
    <div class="prompt-footer">
      <span class="prompt-hint">Ctrl&thinsp;+&thinsp;Enter to generate</span>
      <button id="submit-btn" onclick="generate()">Generate</button>
    </div>
  </div>

  </div><!-- /gen-left -->

  <div class="gen-outputs">

  <div class="card card-output">
    <div class="card-label-row">
      <select class="pane-pick" id="pick-baseline" aria-label="Left model"></select>
      <div style="display:flex;gap:12px;align-items:center">
        <div class="view-toggle">
          <button class="toggle-btn active" id="btn-text-baseline" aria-pressed="true" onclick="setView('text','baseline')">Text</button>
          <button class="toggle-btn" id="btn-tokens-baseline" aria-pressed="false" onclick="setView('tokens','baseline')">Tokens</button>
        </div>
        <button class="clear-btn" onclick="clearOutput('baseline')">Clear</button>
      </div>
    </div>
    <div class="pane-facts" id="facts-baseline"></div>
    <div id="output-text-baseline" class="output-box empty">Output will appear here&hellip;</div>
    <div id="output-tokens-baseline" class="token-box" style="display:none"></div>
  </div>

  <div class="card card-output" id="card-best">
    <div class="card-label-row">
      <select class="pane-pick" id="pick-best" aria-label="Right model"></select>
      <div style="display:flex;gap:12px;align-items:center">
        <div class="view-toggle">
          <button class="toggle-btn active" id="btn-text-best" aria-pressed="true" onclick="setView('text','best')">Text</button>
          <button class="toggle-btn" id="btn-tokens-best" aria-pressed="false" onclick="setView('tokens','best')">Tokens</button>
        </div>
        <button class="clear-btn" onclick="clearOutput('best')">Clear</button>
      </div>
    </div>
    <div class="pane-facts" id="facts-best"></div>
    <div id="output-text-best" class="output-box empty">Output will appear here&hellip;</div>
    <div id="output-tokens-best" class="token-box" style="display:none"></div>
  </div>

  <div class="compare-note" id="compare-note" style="display:none">These two learned from different datasets, so their scores don&rsquo;t compare: each score measures how well a model predicts its own dataset. Compare them by what they write.</div>

  </div><!-- /gen-outputs -->

  <div class="card" id="card-growth" style="display:none">
    <div class="card-label">Watch it learn (training time: one model, 0 s to 5 min)</div>
    <div class="growth-row">
      <select class="pane-pick" id="growth-model" aria-label="Model to watch"></select>
      <select class="pane-pick" id="growth-prompt" aria-label="Prompt"></select>
      <div class="growth-times" id="growth-times" role="group" aria-label="Moment in training"></div>
    </div>
    <div id="growth-text" class="output-box"></div>
    <p class="growth-note">What this model wrote at each moment of its 5 minutes of training, saved in its run file. Every model uses the same prompts and sampling settings, so the differences come from the model.</p>
  </div>

  </div><!-- /gen-bottom -->

  </div><!-- /page-gen -->

  <div id="page-vocab" role="tabpanel" aria-labelledby="tab-vocab" style="display:none">
    <div class="card">
      <div class="card-label">Vocabulary &mdash; <span id="vocab-count"></span> tokens</div>
      <input class="vocab-search" id="vocab-search" type="text"
        placeholder="Filter by token ID or text…" oninput="filterVocab()" style="display:none">
      <div id="vocab-grid" class="vocab-grid"></div>
    </div>
  </div><!-- /page-vocab -->

</div>
<div id="chart-tooltip" class="chart-tooltip"></div>
<script>
  const $ = id => document.getElementById(id);

  let _models = [];
  let _results = [];
  let _commitToModelId = {};
  let _baselineModelId = null;
  let _bestModelId     = null;

  // ---- Page navigation ----
  function switchPage(p) {
    $('page-gen').style.display   = p === 'gen'   ? 'flex'  : 'none';
    $('page-vocab').style.display = p === 'vocab' ? 'block' : 'none';
    $('tab-gen').className   = 'page-tab' + (p === 'gen'   ? ' active' : '');
    $('tab-vocab').className = 'page-tab' + (p === 'vocab' ? ' active' : '');
    $('tab-gen').setAttribute('aria-selected', p === 'gen' ? 'true' : 'false');
    $('tab-vocab').setAttribute('aria-selected', p === 'vocab' ? 'true' : 'false');
    if (p === 'vocab') loadVocab();
  }

  // ---- Helpers ----
  const TOK_COLORS = ['#dbeafe','#fce7f3','#d1fae5','#fef9c3','#ede9fe','#ffedd5'];

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function displayText(s) {
    return s.replace(/\\n/g, '\\u21b5').replace(/\\r/g, '\\u21b5').replace(/\\t/g, '\\u2192');
  }

  // ---- Output helpers ----
  function clearOutput(which) {
    const out = $('output-text-' + which);
    if (!out) return;
    out.className = 'output-box empty';
    out.textContent = 'Output will appear here\u2026';
    $('output-tokens-' + which).innerHTML = '';
  }

  function setView(v, which) {
    $('output-text-'   + which).style.display = v === 'text'   ? 'block' : 'none';
    $('output-tokens-' + which).style.display = v === 'tokens' ? 'flex'  : 'none';
    $('btn-text-'   + which).className = 'toggle-btn' + (v === 'text'   ? ' active' : '');
    $('btn-tokens-' + which).className = 'toggle-btn' + (v === 'tokens' ? ' active' : '');
    $('btn-text-'   + which).setAttribute('aria-pressed', v === 'text'   ? 'true' : 'false');
    $('btn-tokens-' + which).setAttribute('aria-pressed', v === 'tokens' ? 'true' : 'false');
  }

  function updateTokens(toks, which) {
    const box = $('output-tokens-' + which);
    if (!box) return;
    box.innerHTML = '';
    toks.forEach(({ id, text }, i) => {
      const el = document.createElement('div');
      el.className = 'tok';
      el.style.background = TOK_COLORS[i % TOK_COLORS.length];
      el.innerHTML =
        '<span class="tok-id">' + id + '</span>' +
        '<span class="tok-text">' + escapeHtml(displayText(text)) + '</span>';
      el.title = '#' + (i + 1) + '  id=' + id + '  ' + JSON.stringify(text);
      box.appendChild(el);
    });
  }

  // ---- Sliders ----
  $('max-tokens').oninput  = e => $('max-tokens-val').textContent  = e.target.value;
  $('temperature').oninput = e => $('temperature-val').textContent = parseFloat(e.target.value).toFixed(2);
  $('top-k').oninput       = e => $('top-k-val').textContent       = e.target.value;

  function setPrompt(text) { $('prompt').value = text; $('prompt').focus(); }

  $('prompt').addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); generate(); }
  });

  // ---- Model loading (kept for model-ID resolution) ----
  async function loadModels() {
    const resp = await fetch('/models');
    if (!resp.ok) throw new Error('Could not load models');
    const payload = await resp.json();
    _models = payload.models || [];
    _commitToModelId = {};
    _models.forEach(m => { if (m.commit) _commitToModelId[m.commit] = m.id; });
  }

  // ---- Progress chart ----
  function niceStep(range) {
    if (range <= 0) return 0.001;
    const exp  = Math.floor(Math.log10(range));
    const frac = range / Math.pow(10, exp);
    const s    = frac < 1.5 ? 1 : frac < 3 ? 2 : frac < 7 ? 5 : 10;
    return s * Math.pow(10, exp);
  }

  function drawChart() {
    const rows = _results;
    if (!rows.length) {
      $('chart-svg').style.display = 'none';
      const s = $('chart-status');
      s.style.display = 'block';
      s.innerHTML = 'No experiments logged yet — this is a fresh workspace. Run <code>uv run train.py</code> to train your first model; each run appends a row to <code>results.tsv</code> and the progress chart appears here on reload.';
      return;
    }
    $('chart-svg').style.display = 'block';

    const parsed = rows.map((r, i) => ({ ...r, idx: i + 1 }));

    // IQR-based outlier clipping for y-axis
    const vals  = [...parsed.map(r => r.val_bpb)].sort((a, b) => a - b);
    const q1    = vals[Math.floor(vals.length * 0.25)];
    const q3    = vals[Math.floor(vals.length * 0.75)];
    const iqr   = q3 - q1 || 0.001;
    const clipHi = q3 + 1.5 * iqr;

    const normal  = parsed.filter(r => r.val_bpb <= clipHi);
    const yVals   = normal.map(r => r.val_bpb);
    const yLo_raw = Math.min(...yVals);
    const yHi_raw = Math.max(...yVals);
    const yPad    = (yHi_raw - yLo_raw) * 0.3 || 0.001;
    const yLo     = yLo_raw - yPad * 0.4;
    const yHi     = yHi_raw + yPad;

    // SVG coordinate space
    const W = 900, H = 240;
    const ML = 68, MR = 24, MT = 28, MB = 44;
    const pw = W - ML - MR, ph = H - MT - MB;

    const n      = parsed.length;
    const xRange = Math.max(1, n - 1);

    const xS = i  => n === 1 ? ML + pw / 2 : ML + ((i - 1) / xRange) * pw;
    const yS = v  => { const c = Math.max(yLo, Math.min(yHi, v)); return MT + (1 - (c - yLo) / (yHi - yLo)) * ph; };

    let svg = '';

    // Y grid + tick labels
    const yStep  = niceStep((yHi - yLo) / 5);
    const yStart = Math.ceil(yLo / yStep) * yStep;
    for (let y = yStart; y <= yHi + yStep * 0.01; y = +(y + yStep).toFixed(10)) {
      const cy = yS(y).toFixed(1);
      svg += `<line x1="${ML}" y1="${cy}" x2="${W - MR}" y2="${cy}" stroke="#e5e7eb" stroke-width="1"/>`;
      svg += `<text x="${ML - 5}" y="${+cy + 3.5}" text-anchor="end" font-size="10" fill="#9ca3af">${y.toFixed(4)}</text>`;
    }

    // X grid + tick labels (experiment index)
    const xTickStep = Math.max(1, Math.ceil(n / 9));
    for (let t = 1; t <= n; t += xTickStep) {
      const cx = xS(t).toFixed(1);
      svg += `<line x1="${cx}" y1="${MT}" x2="${cx}" y2="${MT + ph}" stroke="#e5e7eb" stroke-width="1"/>`;
      svg += `<text x="${cx}" y="${MT + ph + 14}" text-anchor="middle" font-size="10" fill="#9ca3af">${t}</text>`;
    }

    // Axes
    svg += `<line x1="${ML}" y1="${MT}" x2="${ML}" y2="${MT + ph}" stroke="#d1d5db" stroke-width="1.5"/>`;
    svg += `<line x1="${ML}" y1="${MT + ph}" x2="${W - MR}" y2="${MT + ph}" stroke="#d1d5db" stroke-width="1.5"/>`;

    // Axis labels
    svg += `<text transform="rotate(-90)" x="${-(MT + ph / 2)}" y="13" text-anchor="middle" font-size="10" fill="#6b7280">Validation BPB (lower is better)</text>`;
    svg += `<text x="${ML + pw / 2}" y="${H - 4}" text-anchor="middle" font-size="10" fill="#6b7280">Experiment #</text>`;

    // Running-best step line (connects only models that improved on prior best)
    const keptImproving = [];
    let runBest = Infinity;
    parsed.forEach(r => {
      if (r.status === 'keep' && r.val_bpb < runBest) {
        runBest = r.val_bpb;
        keptImproving.push({ ...r, runBest });
      }
    });
    if (keptImproving.length) {
      let stepPath = '';
      keptImproving.forEach((r, i) => {
        const cx = xS(r.idx).toFixed(1);
        const cy = yS(r.runBest).toFixed(1);
        if (i === 0) { stepPath = `M${cx},${cy}`; }
        else         { stepPath += ` H${cx} V${cy}`; }
      });
      stepPath += ` H${W - MR}`;
      svg += `<path d="${stepPath}" fill="none" stroke="#10b981" stroke-width="1.5" stroke-linejoin="round"/>`;
    }

    // Data points: discarded first (behind), then kept on top
    const discarded = parsed.filter(r => r.val_bpb <= clipHi && r.status !== 'keep');
    const kept      = parsed.filter(r => r.val_bpb <= clipHi && r.status === 'keep');
    const clipped   = parsed.filter(r => r.val_bpb > clipHi);

    discarded.forEach(r => {
      const cx = xS(r.idx).toFixed(1), cy = yS(r.val_bpb).toFixed(1);
      svg += `<circle cx="${cx}" cy="${cy}" r="10" fill="#f3f4f6" stroke="#9ca3af" stroke-width="1.5" class="chart-pt" data-idx="${r.idx - 1}"/>`;
      svg += `<text x="${cx}" y="${+cy + 3.5}" text-anchor="middle" font-size="9" fill="#6b7280" pointer-events="none">${r.idx}</text>`;
    });

    kept.forEach(r => {
      const cx = xS(r.idx).toFixed(1), cy = yS(r.val_bpb).toFixed(1);
      svg += `<circle cx="${cx}" cy="${cy}" r="13" fill="#10b981" stroke="#059669" stroke-width="2" class="chart-pt" data-idx="${r.idx - 1}" style="cursor:pointer"/>`;
      svg += `<text x="${cx}" y="${+cy + 4}" text-anchor="middle" font-size="10" fill="white" font-weight="700" pointer-events="none">${r.idx}</text>`;
      svg += `<text x="${cx}" y="${+cy - 17}" text-anchor="start" font-size="9" fill="#059669" transform="rotate(-35,${cx},${+cy - 17})" pointer-events="none">${escapeHtml(r.description)}</text>`;
    });

    clipped.forEach(r => {
      const cx = xS(r.idx).toFixed(1), ty = MT + 12;
      svg += `<polygon points="${cx},${ty - 11} ${+cx - 9},${ty + 5} ${+cx + 9},${ty + 5}" fill="#9ca3af" stroke="#6b7280" stroke-width="1.5" class="chart-pt" data-idx="${r.idx - 1}"/>`;
      svg += `<text x="${cx}" y="${ty + 20}" text-anchor="middle" font-size="9" fill="#d97706">${r.val_bpb.toFixed(4)}</text>`;
      svg += `<text x="${cx}" y="${ty + 33}" text-anchor="middle" font-size="9" fill="#6b7280" font-weight="700">${r.idx}</text>`;
    });

    // Legend (top-right)
    const lx = W - MR - 128, ly = MT + 6;
    svg += `<rect x="${lx - 4}" y="${ly - 4}" width="134" height="74" rx="5" fill="white" stroke="#e5e7eb" stroke-width="1"/>`;
    svg += `<circle cx="${lx + 7}" cy="${ly + 9}"  r="6"  fill="#f3f4f6" stroke="#9ca3af" stroke-width="1.5"/>`;
    svg += `<text x="${lx + 18}" y="${ly + 13}" font-size="10" fill="#374151">Discarded</text>`;
    svg += `<circle cx="${lx + 7}" cy="${ly + 28}" r="7"  fill="#10b981" stroke="#059669" stroke-width="1.5"/>`;
    svg += `<text x="${lx + 18}" y="${ly + 32}" font-size="10" fill="#374151">Kept</text>`;
    svg += `<line x1="${lx + 2}" y1="${ly + 46}" x2="${lx + 14}" y2="${ly + 46}" stroke="#10b981" stroke-width="1.5"/>`;
    svg += `<text x="${lx + 18}" y="${ly + 50}" font-size="10" fill="#374151">Running best</text>`;
    svg += `<polygon points="${lx + 7},${ly + 58} ${lx + 0},${ly + 68} ${lx + 14},${ly + 68}" fill="#9ca3af" stroke="#6b7280" stroke-width="1"/>`;
    svg += `<text x="${lx + 18}" y="${ly + 67}" font-size="10" fill="#374151">Outlier (clipped)</text>`;

    // Title
    const keptN = parsed.filter(r => r.status === 'keep').length;
    svg += `<text x="${ML + pw / 2}" y="18" text-anchor="middle" font-size="12" font-weight="700" fill="#111827">${parsed.length} experiments, ${keptN} kept</text>`;

    $('chart-svg').innerHTML = svg;
    $('chart-status').style.display = 'none';

    // Hover tooltips
    $('chart-svg').querySelectorAll('.chart-pt').forEach(el => {
      el.addEventListener('mousemove', e => {
        const r   = parsed[parseInt(el.dataset.idx)];
        const tip = $('chart-tooltip');
        tip.innerHTML =
          `<strong>#${r.idx}  ${escapeHtml(r.description)}</strong>\\n` +
          `BPB: ${r.val_bpb.toFixed(6)}\\n` +
          `Status: ${r.status}\\n` +
          `Time: ${new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}\\n` +
          `Memory: ${r.memory_gb == null ? '—' : r.memory_gb + ' GB'}`;
        tip.style.display = 'block';
        tip.style.left = (e.clientX + 14) + 'px';
        tip.style.top  = (e.clientY - 12) + 'px';
      });
      el.addEventListener('mouseleave', () => { $('chart-tooltip').style.display = 'none'; });
    });
  }

  async function loadResults() {
    try {
      const resp = await fetch('/results');
      if (!resp.ok) throw new Error(resp.statusText);
      const { rows, skipped } = await resp.json();
      _results = rows;
      drawChart();
      if (skipped) {
        const s = $('chart-status');
        s.style.display = 'block';
        s.textContent = skipped + ' unplottable row(s) in results.tsv were skipped.';
      }

      // Identify baseline (first kept) and best (lowest val_bpb kept)
      const kept = rows.filter(r => r.status === 'keep');
      if (kept.length) {
        $('card-best').style.display = '';
        const first = kept[0];
        // Scores compare only within one dataset, so "best" comes from the baseline's.
        const datasetOf = r => (_models.find(m => m.commit === r.commit) || {}).dataset;
        const sameData = kept.filter(r => datasetOf(r) && datasetOf(r) === datasetOf(first));
        const best  = (sameData.length ? sameData : kept).reduce((a, b) => a.val_bpb <= b.val_bpb ? a : b);
        _baselineModelId = _commitToModelId[first.commit] || null;
        _bestModelId     = _commitToModelId[best.commit]  || null;
        setupPickers(_baselineModelId, _bestModelId);
      } else {
        _baselineModelId = null;
        _bestModelId     = null;
        setupPickers(null, null);
        $('card-best').style.display = 'none';
      }
    } catch (err) {
      $('chart-status').textContent = 'Could not load results: ' + err.message;
    }
  }

  function factsText(m) {
    const parts = [];
    if (m.val_bpb != null) parts.push('score ' + m.val_bpb.toFixed(6) + ' (lower is better)');
    parts.push(m.dataset === 'folktales' ? 'Folktales' : 'TinyStories');
    if (m.vocab_size) parts.push(m.tokenizer + ' tokenizer (' + m.vocab_size.toLocaleString() + ' tokens)');
    if (m.params_m) parts.push(m.params_m + ' M parameters');
    return parts.join(' \\u00b7 ');
  }
  function modelName(m) { return (m.description || m.label) + (m.commit ? ' \\u2014 ' + m.commit : ''); }
  function fillPick(sel, chosen) {
    sel.innerHTML = '';
    _models.forEach(m => {
      const o = document.createElement('option');
      o.value = m.id;
      o.textContent = modelName(m) + (m.available ? '' : ' (can\\u2019t load)');
      o.disabled = !m.available;
      o.title = m.available ? factsText(m) : m.reason;
      sel.appendChild(o);
    });
    if (chosen) sel.value = chosen;
  }
  function showFacts(which) {
    const m = _models.find(x => x.id === $('pick-' + which).value);
    $('facts-' + which).textContent = m ? factsText(m) : '';
    if (which === 'baseline' && m) { $('dataset-blurb').textContent = m.dataset_blurb; _vocabData = null; }
    updateCompareNote();
  }
  function updateCompareNote() {
    const a = _models.find(x => x.id === $('pick-baseline').value);
    const b = _models.find(x => x.id === $('pick-best').value);
    const differ = a && b && a.dataset !== b.dataset && $('card-best').style.display !== 'none';
    $('compare-note').style.display = differ ? '' : 'none';
  }
  function setupPickers(leftId, rightId) {
    const usable = _models.filter(m => m.available);
    const fallback = (usable[usable.length - 1] || {}).id;
    fillPick($('pick-baseline'), leftId || fallback);
    fillPick($('pick-best'), rightId || fallback);
    ['baseline', 'best'].forEach(w => { $('pick-' + w).onchange = () => showFacts(w); showFacts(w); });
    setupGrowth();
  }

  // ---- Watch it learn ----
  let _growth = null;
  async function setupGrowth() {
    const withGrowth = _models.filter(m => m.has_growth && m.available);
    if (!withGrowth.length) { $('card-growth').style.display = 'none'; return; }
    $('card-growth').style.display = '';
    const sel = $('growth-model');
    sel.innerHTML = '';
    withGrowth.forEach(m => { const o = document.createElement('option'); o.value = m.id; o.textContent = modelName(m); sel.appendChild(o); });
    sel.onchange = loadGrowth;
    $('growth-prompt').onchange = () => showGrowth(0);
    await loadGrowth();
  }
  async function loadGrowth() {
    _growth = await (await fetch('/growth?model_id=' + encodeURIComponent($('growth-model').value))).json();
    const p = $('growth-prompt');
    p.innerHTML = '';
    _growth.prompts.forEach((text, i) => { const o = document.createElement('option'); o.value = i; o.textContent = text; p.appendChild(o); });
    const times = $('growth-times');
    times.innerHTML = '';
    _growth.snapshots.forEach((s, i) => {
      const b = document.createElement('button');
      b.className = 'toggle-btn';
      b.textContent = s.t_s < 60 ? Math.round(s.t_s) + ' s' : (s.t_s / 60).toFixed(s.t_s % 60 ? 1 : 0) + ' min';
      b.onclick = () => showGrowth(i);
      times.appendChild(b);
    });
    showGrowth(_growth.snapshots.length - 1);
  }
  function showGrowth(i) {
    if (!_growth || !_growth.snapshots.length) return;
    const prompt = parseInt($('growth-prompt').value || '0');
    Array.from($('growth-times').children).forEach((b, j) => {
      b.classList.toggle('active', j === i); b.setAttribute('aria-pressed', j === i ? 'true' : 'false');
    });
    $('growth-text').textContent = _growth.prompts[prompt] + (_growth.snapshots[i].samples[prompt] || '');
  }

  // ---- Vocabulary browser ----
  let _vocabData = null;

  async function loadVocab() {
    if (_vocabData) return;
    const grid = $('vocab-grid');
    grid.innerHTML = '<div style="padding:20px;color:#aaa">Loading\u2026</div>';
    const { entries } = await (await fetch('/vocab?model_id=' + encodeURIComponent($('pick-baseline').value))).json();
    _vocabData = entries;
    $('vocab-count').textContent = entries.length.toLocaleString();
    renderVocab(entries);
  }

  function renderVocab(entries) {
    const frag = document.createDocumentFragment();
    entries.forEach(({ id, text }) => {
      const el = document.createElement('div');
      el.className = 'vocab-entry';
      el.title = 'id=' + id + '  ' + JSON.stringify(text);
      const disp = escapeHtml(displayText(text)) || '<span style="color:#ddd">\u2205</span>';
      el.innerHTML =
        '<div class="vocab-entry-id">' + id + '</div>' +
        '<span class="vocab-entry-text">' + disp + '</span>';
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
    renderVocab(_vocabData.filter(({ id, text }) =>
      String(id).includes(q) || text.toLowerCase().includes(q)
    ));
  }

  // ---- Generator ----
  async function runGenerate(prompt, modelId, which) {
    const output = $('output-text-' + which);
    output.className   = 'output-box';
    output.textContent = '';
    $('output-tokens-' + which).innerHTML = '';

    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    output.appendChild(cursor);

    try {
      const body = {
        prompt,
        max_tokens:  parseInt($('max-tokens').value),
        temperature: parseFloat($('temperature').value),
        top_k:       parseInt($('top-k').value),
      };
      if (modelId) body.model_id = modelId;

      const resp = await fetch('/generate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!resp.ok) { output.textContent = 'Server error: ' + resp.statusText; return; }

      const reader  = resp.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split('\\n')) {
          if (!line.trim()) continue;
          try {
            const { t, toks } = JSON.parse(line);
            output.textContent = '';
            output.appendChild(document.createTextNode(t));
            output.appendChild(cursor);
            updateTokens(toks, which);
          } catch (_) {}
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
    btn.disabled    = true;
    btn.textContent = 'Generating\u2026';

    try {
      await Promise.all([
        runGenerate(prompt, $('pick-baseline').value, 'baseline'),
        runGenerate(prompt, $('pick-best').value, 'best'),
      ]);
    } finally {
      btn.disabled    = false;
      btn.textContent = 'Generate';
    }
  }

  // ---- Init ----
  loadModels()
    .then(() => loadResults())
    .catch(err => { $('chart-status').textContent = 'Init failed: ' + err; });
</script>
</body>
</html>
"""

# Public-demo mode (see the note at the top of this file). Everything gated on
# this flag is inert when the env var is unset, which is the local default.
_SPACE_MODE = os.environ.get("SPACE_MODE") == "1"
_SPACE_BANNER = (
    '<p style="margin-top:6px;font-size:0.82rem;">'
    "A public demo of models trained by an autonomous research loop. The two write almost alike: "
    "the session's gain was smaller than run-to-run noise, which is why the agent needs a score. "
    '<a href="https://github.com/aroughidea/autoresearch-win-rtx/blob/master/TRAINING-DECISIONS.md" '
    'target="_blank" rel="noopener">What this demonstrates</a> &middot; '
    '<a href="https://github.com/aroughidea/autoresearch-win-rtx/blob/master/WALKTHROUGH.md" '
    'target="_blank" rel="noopener">how these were made</a>.</p>'
)

# 16x16 PNG favicon (dark slate background, emerald square) served at /favicon.ico
# so the browser stops logging a 404 on every page load.
_FAVICON_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAIElEQVR42mMQlFD/TwlmGIYG"
    "COxsxItHDRgZBozAvAAAO0qNkB8R+7AAAAAASUVORK5CYII="
)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(model_store: ModelStore) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def root():
        if _SPACE_MODE:
            # "local" is wrong on a public URL; retitle, then add the banner
            # after the header paragraph.
            page = _HTML.replace("local text generation", "a model trained by an AI researcher")
            return page.replace("</p>", "</p>" + _SPACE_BANNER, 1)
        return _HTML

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

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(content=_FAVICON_PNG, media_type="image/png")

    @app.get("/results")
    def results_data():
        path = Path("results.tsv")
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
                rows.append({
                    "timestamp": timestamp,
                    "commit": (r.get("commit") or "").strip(),
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
    try:
        model_store = ModelStore(device=device, entries=entries, active_path=args.checkpoint)
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
