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

import argparse
import json
import socket
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from generate import _config_from_state_dict, _sample_top_k
from prepare import Tokenizer
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


def _load(checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        tokenizer = Tokenizer.from_directory()
    except (FileNotFoundError, OSError):
        print("Tokenizer not found. Run 'uv run prepare.py' first.")
        sys.exit(1)

    try:
        model = _load_model_from_checkpoint(checkpoint_path, device)
    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run 'uv run train.py' first to produce a checkpoint.")
        sys.exit(1)

    print("Model ready.\n")
    return model, tokenizer, device


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


class ModelStore:
    def __init__(self, initial_model: GPT, tokenizer: Tokenizer, device: str, entries: list[dict], active_path: str):
        self._tokenizer = tokenizer
        self._device = device
        self._entries = entries
        self._entries_by_id = {e["id"]: e for e in entries}
        self._cache: dict[str, GPT] = {}
        self._lock = threading.Lock()

        normalized_active = str(Path(active_path)).replace("\\", "/")
        active_entry = next((e for e in entries if e["path"] == normalized_active), None)
        if active_entry is None and entries:
            active_entry = entries[-1]

        if active_entry is None:
            raise RuntimeError("No checkpoint files found. Provide --checkpoint or add .pt files.")

        self._active_id = active_entry["id"]
        self._cache[active_entry["path"]] = initial_model

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def device(self) -> str:
        return self._device

    def list_models(self) -> list[dict]:
        with self._lock:
            return [
                {
                    **entry,
                    "active": entry["id"] == self._active_id,
                }
                for entry in self._entries
            ]

    def get_active_bundle(self) -> tuple[GPT, Tokenizer, str, dict]:
        with self._lock:
            entry = self._entries_by_id[self._active_id]
            model = self._cache.get(entry["path"])
            if model is None:
                model = _load_model_from_checkpoint(entry["path"], self._device)
                self._cache[entry["path"]] = model
            return model, self._tokenizer, self._device, entry

    def select(self, model_id: str) -> dict:
        with self._lock:
            if model_id not in self._entries_by_id:
                raise KeyError(f"Unknown model id: {model_id}")
            entry = self._entries_by_id[model_id]
            if entry["path"] not in self._cache:
                self._cache[entry["path"]] = _load_model_from_checkpoint(entry["path"], self._device)
            self._active_id = model_id
            return entry


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

  /* ---- Generator two-column layout ---- */
  #page-gen {
    display: grid;
    grid-template-columns: minmax(320px, 380px) minmax(0, 1fr);
    gap: 12px;
    align-items: start;
  }
  .gen-left { display: flex; flex-direction: column; gap: 10px; }
  .gen-left .card { margin-bottom: 0; }
  .card-output { display: flex; flex-direction: column; margin-bottom: 0; }
  .card-output .card-label-row { flex-shrink: 0; }
  .card-output .output-box { flex: 1; min-height: 340px; }
  .card-output .token-box  { flex: 1; min-height: 340px; }

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
    #page-gen {
      grid-template-columns: 1fr;
    }
    .card-output .output-box,
    .card-output .token-box {
      min-height: 260px;
    }
  }

  @media (max-width: 760px) {
    body { padding: 14px 12px 20px; }
    .card { padding: 14px; }
    .params { grid-template-columns: 1fr; gap: 12px; }
    .page-tab { padding: 10px 12px; }
    .prompt-footer { gap: 12px; align-items: flex-start; flex-direction: column; }
  }
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>autoresearch &mdash; local text generation</h1>
    <p>
      This is a <strong>text completion model</strong> trained on short children&rsquo;s
      stories. It is not a chat assistant &mdash; it continues whatever you type in the same
      writing style. Type the start of a sentence or story and press
      <strong>Generate</strong>. Output appears word&nbsp;by&nbsp;word in real time.
    </p>
  </div>

  <nav class="page-nav" role="tablist" aria-label="Main views">
    <button class="page-tab active" id="tab-gen" role="tab" aria-selected="true" aria-controls="page-gen" onclick="switchPage('gen')">Generator</button>
    <button class="page-tab" id="tab-vocab" role="tab" aria-selected="false" aria-controls="page-vocab" onclick="switchPage('vocab')">Vocabulary</button>
  </nav>

  <div id="page-gen" role="tabpanel" aria-labelledby="tab-gen">

  <div class="gen-left">

  <div class="card">
    <div class="card-label">Model timeline</div>
    <div id="model-active" class="model-active">Loading checkpoints…</div>
    <div id="model-timeline" class="timeline-wrap"></div>
  </div>

  <div class="card">
    <div class="card-label">Model parameters</div>
    <div class="params">

      <div>
        <div class="param-row">
          <span class="name">Max tokens</span>
          <span class="value" id="max-tokens-val">150</span>
        </div>
        <input type="range" id="max-tokens" min="20" max="500" step="10" value="150">
        <div class="param-hint">How many words to generate</div>
      </div>

      <div>
        <div class="param-row">
          <span class="name">Temperature</span>
          <span class="value" id="temperature-val">0.90</span>
        </div>
        <input type="range" id="temperature" min="0" max="2" step="0.01" value="0.9">
        <div class="param-hint">0 = predictable &nbsp;&middot;&nbsp; 2 = chaotic</div>
      </div>

      <div>
        <div class="param-row">
          <span class="name">Top-k</span>
          <span class="value" id="top-k-val">50</span>
        </div>
        <input type="range" id="top-k" min="0" max="200" step="5" value="50">
        <div class="param-hint">Word choices considered &mdash; 0 means all</div>
      </div>

    </div>
  </div>

  <div class="card">
    <div class="card-label">Prompt</div>
    <textarea id="prompt" placeholder="Once upon a time\u2026"></textarea>
    <div class="prompt-footer">
      <span class="prompt-hint">Ctrl&thinsp;+&thinsp;Enter to generate</span>
      <button id="submit-btn" onclick="generate()">Generate</button>
    </div>
  </div>

  </div><!-- /gen-left -->

  <div class="card card-output">
    <div class="card-label-row">
      <div class="card-label">Output</div>
      <div style="display:flex;gap:12px;align-items:center">
        <div class="view-toggle">
          <button class="toggle-btn active" id="btn-text" aria-pressed="true" onclick="setView('text')">Text</button>
          <button class="toggle-btn" id="btn-tokens" aria-pressed="false" onclick="setView('tokens')">Tokens</button>
        </div>
        <button class="clear-btn" onclick="clearOutput()">Clear</button>
      </div>
    </div>
    <div id="output-text" class="output-box empty">Output will appear here&hellip;</div>
    <div id="output-tokens" class="token-box" style="display:none"></div>
  </div>

  </div><!-- /page-gen -->

  <div id="page-vocab" role="tabpanel" aria-labelledby="tab-vocab" style="display:none">
    <div class="card">
      <div class="card-label">Vocabulary &mdash; <span id="vocab-count"></span> tokens</div>
      <input class="vocab-search" id="vocab-search" type="text"
        placeholder="Filter by token ID or text…" oninput="filterVocab()">
      <div id="vocab-grid" class="vocab-grid"></div>
    </div>
  </div><!-- /page-vocab -->

</div>
<script>
  const $ = id => document.getElementById(id);

  let _models = [];
  let _switchingModel = false;

  function _fmtTime(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString([], { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
  }

  function _renderModelTimeline(models) {
    const active = models.find(m => m.active);
    const activeLabel = active
      ? `${active.label} (${active.path})`
      : 'No active checkpoint';
    $('model-active').textContent = `Active: ${activeLabel}`;

    const wrap = $('model-timeline');
    wrap.innerHTML = '';
    models.forEach((m, idx) => {
      const btn = document.createElement('button');
      btn.className = 'timeline-node' + (m.active ? ' active' : '');
      btn.disabled = _switchingModel;
      btn.title = `${m.path} | ${m.mtime_iso} | ${m.size_mb.toFixed(1)} MB`;
      btn.onclick = () => selectModel(m.id);

      const top = document.createElement('div');
      top.className = 'timeline-top';

      const num = document.createElement('span');
      num.className = 'timeline-idx';
      num.textContent = `Model ${idx + 1}`;

      const stamp = document.createElement('span');
      stamp.className = 'timeline-time';
      stamp.textContent = _fmtTime(m.mtime_iso);

      const name = document.createElement('div');
      name.className = 'timeline-name';
      name.textContent = m.label;

      top.appendChild(num);
      top.appendChild(stamp);
      btn.appendChild(top);
      btn.appendChild(name);
      wrap.appendChild(btn);
    });
  }

  async function loadModels() {
    const resp = await fetch('/models');
    if (!resp.ok) {
      throw new Error('Could not load model timeline');
    }
    const payload = await resp.json();
    _models = payload.models || [];
    _renderModelTimeline(_models);
  }

  async function selectModel(modelId) {
    if (_switchingModel) return;
    _switchingModel = true;
    _renderModelTimeline(_models);
    try {
      const resp = await fetch('/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId }),
      });
      if (!resp.ok) {
        throw new Error('Switch failed: ' + resp.statusText);
      }
      await loadModels();
      clearOutput();
    } catch (err) {
      $('model-active').textContent = String(err);
    } finally {
      _switchingModel = false;
      _renderModelTimeline(_models);
    }
  }

  $('max-tokens').oninput  = e => $('max-tokens-val').textContent  = e.target.value;
  $('temperature').oninput = e => $('temperature-val').textContent = parseFloat(e.target.value).toFixed(2);
  $('top-k').oninput       = e => $('top-k-val').textContent       = e.target.value;

  $('prompt').addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); generate(); }
  });

  // ---- Page navigation (Generator / Vocabulary) ----
  function switchPage(p) {
    $('page-gen').style.display   = p === 'gen'   ? 'grid' : 'none';
    $('page-vocab').style.display = p === 'vocab' ? 'block' : 'none';
    $('tab-gen').className   = 'page-tab' + (p === 'gen'   ? ' active' : '');
    $('tab-vocab').className = 'page-tab' + (p === 'vocab' ? ' active' : '');
    $('tab-gen').setAttribute('aria-selected', p === 'gen' ? 'true' : 'false');
    $('tab-vocab').setAttribute('aria-selected', p === 'vocab' ? 'true' : 'false');
    if (p === 'vocab') loadVocab();
  }

  // ---- Clear output ----
  function clearOutput() {
    const out = $('output-text');
    out.className = 'output-box empty';
    out.textContent = 'Output will appear here\u2026';
    $('output-tokens').innerHTML = '';
  }

  // ---- View toggle (Text / Tokens) ----
  function setView(v) {
    $('output-text').style.display   = v === 'text'   ? 'block' : 'none';
    $('output-tokens').style.display = v === 'tokens' ? 'flex'  : 'none';
    $('btn-text').className   = 'toggle-btn' + (v === 'text'   ? ' active' : '');
    $('btn-tokens').className = 'toggle-btn' + (v === 'tokens' ? ' active' : '');
    $('btn-text').setAttribute('aria-pressed', v === 'text' ? 'true' : 'false');
    $('btn-tokens').setAttribute('aria-pressed', v === 'tokens' ? 'true' : 'false');
  }

  // ---- Token chip helpers ----
  const TOK_COLORS = ['#dbeafe','#fce7f3','#d1fae5','#fef9c3','#ede9fe','#ffedd5'];

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function displayText(s) {
    // Replace invisible whitespace chars with visible markers
    return s.replace(/\\n/g, '\u21b5').replace(/\\r/g, '\u21b5').replace(/\\t/g, '\u2192');
  }

  function updateTokens(toks) {
    const box = $('output-tokens');
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

  // ---- Vocabulary browser ----
  let _vocabData = null;

  async function loadVocab() {
    if (_vocabData) return;
    const grid = $('vocab-grid');
    grid.innerHTML = '<div style="padding:20px;color:#aaa">Loading\u2026</div>';
    const { entries } = await (await fetch('/vocab')).json();
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
  async function generate() {
    const prompt = $('prompt').value.trim();
    if (!prompt) { $('prompt').focus(); return; }

    const btn    = $('submit-btn');
    const output = $('output-text');

    btn.disabled       = true;
    btn.textContent    = 'Generating\u2026';
    output.className   = 'output-box';
    output.textContent = '';
    $('output-tokens').innerHTML = '';

    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    output.appendChild(cursor);

    try {
      const resp = await fetch('/generate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_tokens:  parseInt($('max-tokens').value),
          temperature: parseFloat($('temperature').value),
          top_k:       parseInt($('top-k').value),
        }),
      });

      if (!resp.ok) {
        output.textContent = 'Server error: ' + resp.statusText;
        return;
      }

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
            updateTokens(toks);
          } catch (_) {}
        }
      }
    } finally {
      cursor.remove();
      btn.disabled    = false;
      btn.textContent = 'Generate';
    }
  }

  loadModels().catch(err => {
    $('model-active').textContent = 'Failed to load checkpoints: ' + err;
  });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(model_store: ModelStore) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def root():
        return _HTML

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 150
        temperature: float = 0.9
        top_k: int = 50

    class SelectModelRequest(BaseModel):
        model_id: str

    @app.post("/generate")
    def generate(req: GenerateRequest):
        model, tokenizer, device, _ = model_store.get_active_bundle()
        respond = _make_respond(model, tokenizer, device)

        def stream():
            for text, toks in respond(req.prompt, req.max_tokens, req.temperature, req.top_k):
                yield json.dumps({"t": text, "toks": toks}) + "\n"
        return StreamingResponse(stream(), media_type="text/plain")

    @app.get("/vocab")
    def vocab():
        tokenizer = model_store.tokenizer
        n = tokenizer.get_vocab_size()
        entries = [{"id": i, "text": tokenizer.decode([i])} for i in range(n)]
        return {"entries": entries}

    @app.get("/models")
    def models():
        return {"models": model_store.list_models()}

    @app.post("/models/select")
    def select_model(req: SelectModelRequest):
        try:
            entry = model_store.select(req.model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True, "active": entry}

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

    model, tokenizer, device = _load(args.checkpoint)
    entries = _discover_checkpoints(args.checkpoint)
    model_store = ModelStore(
      initial_model=model,
      tokenizer=tokenizer,
      device=device,
      entries=entries,
      active_path=args.checkpoint,
    )
    app = build_app(model_store)

    port = _find_free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} in use, using {port} instead.")

    if not args.no_browser:
        def _open():
            time.sleep(1.2)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
