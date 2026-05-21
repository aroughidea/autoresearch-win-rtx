# generate.py
# Author: Thomas J McLeish
# License: MIT
#
# Load a trained checkpoint and generate text from a prompt.
#
# Quick start:
#   uv run generate.py "Once upon a time"
#   uv run generate.py "The little dog" --max-tokens 200
#   uv run generate.py "Once upon a time" --temperature 1.2 --top-k 100
#
# Prerequisites:
#   uv run prepare.py      (one-time data + tokenizer setup)
#   uv run train.py        (produces checkpoint_pre_eval.pt)
#
# The checkpoint does not store the model config explicitly, but the config can
# be fully inferred from the shapes of the saved weight tensors. This script does
# that automatically. Use --n-head to override only if auto-detection fails.

import argparse
import sys

import torch

from prepare import Tokenizer
from train import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Config detection
# ---------------------------------------------------------------------------

def _config_from_state_dict(state_dict: dict, n_head_override: int | None = None) -> GPTConfig:
    """Infer GPTConfig from checkpoint weight shapes.

    Most fields are unambiguous from shapes. n_head requires a hint if the
    agent changed it and the ve_gate is absent (rare). A command-line override
    is available as a fallback.
    """
    vocab_size, n_embd = state_dict["transformer.wte.weight"].shape
    n_layer = int(state_dict["resid_lambdas"].shape[0])

    # n_kv_head from ve_gate.weight (shape: n_kv_head × ve_gate_channels=32)
    ve_gate_keys = [k for k in state_dict if k.endswith("ve_gate.weight")]
    if ve_gate_keys:
        n_kv_head = int(state_dict[ve_gate_keys[0]].shape[0])
    else:
        # No ve_gate layers — assume GQA ratio 1 as a safe fallback
        n_kv_head = None

    # n_head from c_k weight (shape: n_kv_head * head_dim × n_embd)
    # head_dim = c_k_out_dim / n_kv_head, n_head = n_embd / head_dim
    if n_kv_head is not None:
        c_k_out_dim = int(state_dict["transformer.h.0.attn.c_k.weight"].shape[0])
        head_dim = c_k_out_dim // n_kv_head
        n_head = n_embd // head_dim
    elif n_head_override is not None:
        n_head = n_head_override
        n_kv_head = n_head
    else:
        # Fall back to GPTConfig defaults if we can't determine n_head
        default = GPTConfig()
        n_head = default.n_head
        n_kv_head = default.n_kv_head
        print(
            f"Warning: could not determine n_head from checkpoint; using default {n_head}. "
            "Pass --n-head if this is wrong."
        )

    cfg = GPTConfig(
        vocab_size=int(vocab_size),
        n_embd=int(n_embd),
        n_layer=n_layer,
        n_head=int(n_head),
        n_kv_head=int(n_kv_head),
        use_activation_checkpointing=False,
    )
    print(
        f"Checkpoint config: n_layer={cfg.n_layer}  n_embd={cfg.n_embd}  "
        f"n_head={cfg.n_head}  n_kv_head={cfg.n_kv_head}  vocab_size={cfg.vocab_size}"
    )
    return cfg



def _sample_top_k(logits: torch.Tensor, top_k: int | None, temperature: float) -> torch.Tensor:
    """Apply temperature + top-k, return a single sampled token id (shape: 1,1)."""
    if temperature < 1e-6:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold, _ = torch.topk(logits, k)
        logits[logits < threshold[:, [-1]]] = float("-inf")
    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.9,
    top_k: int | None = 50,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Auto-regressively sample max_new_tokens tokens, appending to idx."""
    for _ in range(max_new_tokens):
        # Crop to the model's maximum context length
        ctx = idx if idx.size(1) <= model.config.sequence_len else idx[:, -model.config.sequence_len :]
        logits = model(ctx)        # (1, T, vocab_size)
        logits = logits[:, -1, :]  # last position only
        next_token = _sample_top_k(logits, top_k, temperature)
        idx = torch.cat((idx, next_token), dim=1)
        if eos_id is not None and next_token.item() == eos_id:
            break
    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained autoresearch checkpoint."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Once upon a time",
        help="Text prompt to continue (default: 'Once upon a time')",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_pre_eval.pt",
        help="Path to the .pt checkpoint file (default: checkpoint_pre_eval.pt)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Number of new tokens to generate (default: 150)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature — 0 = always pick most likely word, higher = more random (default: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff, 0 to disable (default: 50)",
    )
    # Config overrides — only needed if auto-detection fails or is wrong
    parser.add_argument("--n-head",  type=int, default=None, help="Override detected n_head (rarely needed)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    try:
        tokenizer = Tokenizer.from_directory()
    except (FileNotFoundError, OSError):
        print(
            "\nTokenizer not found.\n"
            "Run 'uv run prepare.py' first to download the dataset and build the tokenizer."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Checkpoint — load weights first so we can detect config from shapes
    # ------------------------------------------------------------------
    try:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except FileNotFoundError:
        print(
            f"\nCheckpoint not found: {args.checkpoint}\n"
            "Run 'uv run train.py' first to produce a checkpoint."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Model — config inferred from checkpoint weight shapes
    # ------------------------------------------------------------------
    config = _config_from_state_dict(state_dict, n_head_override=args.n_head)
    model = GPT(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Encode prompt and run generation
    # ------------------------------------------------------------------
    token_ids = tokenizer.encode(args.prompt, prepend=tokenizer.get_bos_token_id())
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    print(f"Prompt : {args.prompt!r}")
    print("-" * 60)

    eos_id = tokenizer.get_eos_token_id()
    amp_enabled = device == "cuda"
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp_enabled):
        out = generate(
            model,
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            eos_id=eos_id,
        )

    generated_ids = out[0, len(token_ids) :].tolist()
    print(tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
