# chat.py
# Author: Thomas J McLeish
# License: MIT
#
# Launch a local browser UI for the trained model.
# The model continues whatever text you type, token by token.
#
# Usage:
#   uv run chat.py
#   uv run chat.py --checkpoint checkpoint_pre_eval.pt --port 7860
#
# Prerequisites:
#   uv run prepare.py   (one-time setup)
#   uv run train.py     (produces checkpoint_pre_eval.pt)

import argparse
import sys

import torch
import gradio as gr

from prepare import Tokenizer
from generate import _config_from_state_dict, _sample_top_k
from train import GPT


# ---------------------------------------------------------------------------
# Load model (once, at startup)
# ---------------------------------------------------------------------------

def _load(checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        tokenizer = Tokenizer.from_directory()
    except (FileNotFoundError, OSError):
        print("Tokenizer not found. Run 'uv run prepare.py' first.")
        sys.exit(1)

    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run 'uv run train.py' first to produce a checkpoint.")
        sys.exit(1)

    config = _config_from_state_dict(state_dict)
    model = GPT(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model ready.\n")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------

def _make_respond(model: GPT, tokenizer: Tokenizer, device: str):
    """Return a Gradio-compatible streaming response function."""

    @torch.no_grad()
    def respond(message: str, history: list, max_tokens: int, temperature: float, top_k: int):
        """Encode the prompt and stream generated tokens back to the UI."""
        token_ids = tokenizer.encode(message, prepend=tokenizer.get_bos_token_id())
        idx = torch.tensor([token_ids], dtype=torch.long, device=device)
        prompt_len = len(token_ids)
        top_k_val = top_k if top_k > 0 else None

        amp_enabled = device == "cuda"
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp_enabled):
            for _ in range(max_tokens):
                ctx = (
                    idx
                    if idx.size(1) <= model.config.sequence_len
                    else idx[:, -model.config.sequence_len :]
                )
                logits = model(ctx)[:, -1, :]
                next_token = _sample_top_k(logits, top_k_val, temperature)
                idx = torch.cat((idx, next_token), dim=1)
                # Yield the full continuation so far — Gradio replaces the
                # partial response each time rather than appending.
                yield tokenizer.decode(idx[0, prompt_len:].tolist())

    return respond


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
**This is a text completion model**, not a chat assistant. It will continue whatever you type in the style of the text it was trained on (by default, short children's stories).

Type the beginning of a sentence or story and press Enter. The model generates a continuation in real time, word by word.

The quality of the output reflects how many training runs have completed and how well the settings have been tuned — watching it improve is the point of the project.
"""


def build_ui(model: GPT, tokenizer: Tokenizer, device: str) -> gr.Blocks:
    respond = _make_respond(model, tokenizer, device)

    with gr.Blocks(title="autoresearch — local model") as demo:
        gr.Markdown("## autoresearch — local text generation")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.ChatInterface(
                    fn=respond,
                    type="messages",
                    additional_inputs=[
                        gr.Slider(20, 500, value=150, step=10,  label="Max tokens"),
                        gr.Slider(0.1, 2.0, value=0.9, step=0.05, label="Temperature  (higher = more varied)"),
                        gr.Slider(0, 200, value=50,   step=5,   label="Top-k  (0 = off)"),
                    ],
                    additional_inputs_accordion=gr.Accordion("Generation settings", open=False),
                )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Local chatbot UI for the trained model.")
    parser.add_argument("--checkpoint", default="checkpoint_pre_eval.pt", help="Path to .pt checkpoint")
    parser.add_argument("--port", type=int, default=7860, help="Local port (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab automatically")
    args = parser.parse_args()

    model, tokenizer, device = _load(args.checkpoint)
    demo = build_ui(model, tokenizer, device)
    demo.launch(server_port=args.port, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
