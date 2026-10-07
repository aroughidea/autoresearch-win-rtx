"""
Run capture for the Training Decisions explorer.

Records what the model writes at fixed moments of training, on fixed prompts with
fixed decoding, and saves one small JSON run file per experiment in runs/.

A captured run trains exactly like an uncaptured one: train.py calls this module
before each step's timer starts (so sampling never counts toward the 5-minute
budget), and sampling uses its own random generator, never the global one.

The agent must not modify this file (see program.md).
"""

import contextlib
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch


def compact_timestamp(moment):
    """2026-05-23 15:57:43 -07:00 -> '20260523T155743-0700'. Colon-free, safe in Windows filenames."""
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.strftime("%Y%m%dT%H%M%S%z")


def make_run_id(commit, moment):
    return f"{compact_timestamp(moment)}_{commit or 'nogit'}"


def git_commit_info(cwd=None):
    """Short hash and commit time of HEAD, or (None, None) outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h%n%cI"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,  # Windows can stall a process launch for seconds
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or len(lines) != 2:
        return None, None
    try:
        return lines[0], datetime.fromisoformat(lines[1])
    except ValueError:
        return lines[0], None


@torch.no_grad()
def sample_continuation(model, tokenizer, prompt, *, seed, temperature, top_k, min_chars,
                        max_tokens, device, autocast_ctx=None):
    """Continue `prompt` until EOS, `min_chars` characters, or `max_tokens` tokens.

    Uses a private torch.Generator so the global RNG (and so training) is untouched.
    Returns only the continuation, not the prompt.
    """
    was_training = model.training
    model.eval()
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        ids = [tokenizer.get_bos_token_id()] + list(tokenizer.encode(prompt))
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        start = idx.size(1)
        eos = tokenizer.get_eos_token_id()
        window = model.config.sequence_len
        ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
        text = ""
        for _ in range(max_tokens):
            with ctx:
                logits = model(idx[:, -window:])
            logits = logits[:, -1, :].float() / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
            if next_id.item() == eos:
                break
            idx = torch.cat((idx, next_id), dim=1)
            text = tokenizer.decode(idx[0, start:].tolist())
            if len(text) >= min_chars:
                break
        return text
    finally:
        model.train(was_training)


SCHEMA = 1
SNAPSHOT_TIMES = (0, 10, 30, 60, 120)
PROMPTS = ("Once upon a time", "The old king said", "In the dark forest", "The little girl found a")
DECODING = {"seed": 1234, "temperature": 0.8, "top_k": 40, "min_chars": 320, "max_tokens": 300}
RECIPE_KEYS = (
    "DEPTH", "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN", "N_KV_HEAD", "TOTAL_BATCH_SIZE",
    "EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR", "WEIGHT_DECAY", "ADAM_BETAS",
    "WARMUP_RATIO", "WARMDOWN_RATIO", "FINAL_LR_FRAC",
)
CONFIG_KEYS = ("n_layer", "n_embd", "n_head", "n_kv_head", "vocab_size", "sequence_len")


def _finite(value, digits=4):
    """A rounded float, or None for missing, NaN or infinite values (JSON has no NaN)."""
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _jsonable(value):
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return _finite(value, 8)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def recipe_from(namespace, config=None):
    """The recipe as far as it can be read. Keys the agent renamed or removed are simply absent."""
    recipe = {k: _jsonable(namespace[k]) for k in RECIPE_KEYS if k in namespace}
    if config is not None:
        recipe.update({k: _jsonable(getattr(config, k)) for k in CONFIG_KEYS if hasattr(config, k)})
    return recipe


def runs_dir_from_env(smoke_test, env=None):
    """AUTORESEARCH_RUNS_DIR wins ('off' disables). Otherwise runs/ for real runs, nothing for smoke tests."""
    env = os.environ if env is None else env
    value = env.get("AUTORESEARCH_RUNS_DIR", "").strip()
    if value.lower() == "off":
        return None
    if value:
        return value
    return None if smoke_test else "runs"


class RunCapture:
    """Takes writing snapshots on a training-time schedule and writes one run file."""

    def __init__(self, tokenizer, *, dataset, runs_dir, device, tokenizer_name="own",
                 snapshot_times=SNAPSHOT_TIMES, prompts=PROMPTS, decoding=None, log=print):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.runs_dir = Path(runs_dir) if runs_dir is not None else None
        self.device = device
        self.tokenizer_name = tokenizer_name
        self.snapshot_times = tuple(sorted(float(t) for t in snapshot_times))
        self.prompts = tuple(prompts)
        self.decoding = dict(DECODING if decoding is None else decoding)
        self.log = log
        self.error = None
        self.begin_attempt()

    @property
    def enabled(self):
        return self.runs_dir is not None and self.error is None

    def begin_attempt(self, autocast_ctx=None):
        """Start (or restart, after an OOM retry) a training attempt."""
        self.autocast_ctx = autocast_ctx
        self.pending = list(self.snapshot_times)
        self.snapshots = []
        self.sampling_seconds = 0.0
        self.error = None

    def _say(self, message):
        """Log without ever raising: a cp1252 console cannot print every character."""
        try:
            self.log(message)
        except Exception:
            try:
                self.log(ascii(message))
            except Exception:
                pass

    def _fail(self, exc):
        self.error = f"{type(exc).__name__}: {exc}"
        self._say(f"capture: disabled for this run after an error: {self.error}")

    def on_step(self, model, training_seconds, step, train_loss):
        try:
            while self.enabled and self.pending and training_seconds >= self.pending[0]:
                self._take(model, self.pending.pop(0), step, train_loss)
        except Exception as exc:  # capture must never break training
            self._fail(exc)

    def on_train_end(self, model, training_seconds, step, train_loss):
        self.pending = []
        try:
            if self.enabled:
                self._take(model, training_seconds, step, train_loss, final=True)
        except Exception as exc:  # capture must never break training
            self._fail(exc)

    def _take(self, model, t_s, step, train_loss, final=False):
        started = time.time()
        d = self.decoding
        try:
            samples = [
                sample_continuation(
                    model, self.tokenizer, prompt,
                    seed=d["seed"] + i, temperature=d["temperature"], top_k=d["top_k"],
                    min_chars=d["min_chars"], max_tokens=d["max_tokens"],
                    device=self.device, autocast_ctx=self.autocast_ctx,
                )
                for i, prompt in enumerate(self.prompts)
            ]
        except Exception as exc:  # capture must never break training
            self._fail(exc)
            return
        took = time.time() - started
        self.sampling_seconds += took
        entry = {"t_s": round(float(t_s), 1), "step": int(step), "train_loss": _finite(train_loss), "samples": samples}
        if final:
            entry["final"] = True
        self.snapshots.append(entry)
        self._say(f"capture: t={entry['t_s']}s step={step} sampled in {took:.1f}s")

    def finish(self, *, val_bpb, peak_vram_mb, training_seconds, num_steps, num_params, recipe,
               commit=None, committed_at=None):
        """Write runs/<run_id>.json. Returns its path, or None when capture is off or the write failed."""
        if self.runs_dir is None:
            return None
        try:
            return self._write(val_bpb, peak_vram_mb, training_seconds, num_steps, num_params, recipe,
                               commit, committed_at)
        except Exception as exc:  # the score is already printed; never end the run with a traceback
            self._say(f"capture: could not write the run file: {type(exc).__name__}: {exc}")
            return None

    def _write(self, val_bpb, peak_vram_mb, training_seconds, num_steps, num_params, recipe,
               commit, committed_at):
        if commit is None and committed_at is None:
            commit, committed_at = git_commit_info()
        run_id = make_run_id(commit, committed_at or datetime.now().astimezone())
        record = {
            "schema": SCHEMA,
            "run_id": run_id,
            "commit": commit,
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": self.dataset,
            "tokenizer": {
                "name": getattr(self.tokenizer, "name", self.tokenizer_name),
                "source": getattr(self.tokenizer, "source", None),
                "vocab_size": int(self.tokenizer.get_vocab_size()),
            },
            "recipe": recipe,
            "decoding": self.decoding,
            "prompts": list(self.prompts),
            "snapshot_times": list(self.snapshot_times),
            "snapshots": self.snapshots,
            "final": {
                "val_bpb": _finite(val_bpb, 6),
                "training_s": _finite(training_seconds, 1),
                "peak_vram_mb": _finite(peak_vram_mb, 1),
                "num_steps": int(num_steps),
                "params_m": round(num_params / 1e6, 1),
                "sampling_s": round(self.sampling_seconds, 1),
            },
        }
        if self.error:
            record["capture_error"] = self.error
        path = self.runs_dir / f"{run_id}.json"
        tmp = path.with_name(path.name + ".tmp")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)  # a half-written file must not be committed by `git add runs/`
            raise
        self._say(f"capture: wrote {path}")
        return path
