"""recipe-b.py = the starter's train.py (at 6ad8ddd, version 91ffb4c37015) plus only the TinyStories night's three attention changes:
full key/value heads (d967b0b7055d), plain causal attention with all-long windows (14c8214ecc32), and no logit
softcap (8fa477d3ed43). Each edit must match exactly once.

    uv run library/recipes/make_recipe_b.py [OUT]     writes library/recipes/recipe-b.py, or OUT
"""
import subprocess
import sys
from pathlib import Path

STARTER = "6ad8ddd"  # train.py with the starter kit's recipe (library/README.md)
ROOT = Path(__file__).resolve().parents[2]

EDITS = [
    ("N_KV_HEAD = 1  # MQA: all query heads share 1 KV head. None = full MHA (n_kv_head=n_head)",
     "N_KV_HEAD = 3  # GQA was 1 (MQA); 3 = full MHA at 3 heads. None = full MHA (n_kv_head=n_head)"),
    ("        attn_mask = self._get_sdpa_mask(T, window_size, q.device)\n"
     "        y = F.scaled_dot_product_attention(\n"
     "            q,\n"
     "            k,\n"
     "            v,\n"
     "            attn_mask=attn_mask,\n"
     "            is_causal=False,\n",
     "        window = window_size[0] if isinstance(window_size, tuple) else window_size\n"
     "        if window is not None and 0 <= window < T:\n"
     "            attn_mask = self._get_sdpa_mask(T, window_size, q.device)\n"
     "            is_causal = False\n"
     "        else:\n"
     "            attn_mask = None  # plain causal: lets SDPA use its fastest kernel\n"
     "            is_causal = True\n"
     "        y = F.scaled_dot_product_attention(\n"
     "            q,\n"
     "            k,\n"
     "            v,\n"
     "            attn_mask=attn_mask,\n"
     "            is_causal=is_causal,\n"),
    ('WINDOW_PATTERN = "SSSL"   # sliding window pattern: L=full, S=half context',
     'WINDOW_PATTERN = "L"   # all layers full context: plain causal SDPA, no mask tensor'),
    ("        softcap = 15\n"
     "        logits = self.lm_head(x).float()\n"
     "        logits = softcap * torch.tanh(logits / softcap)\n",
     "        logits = self.lm_head(x).float()\n"),
]

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "library" / "recipes" / "recipe-b.py"
text = subprocess.run(["git", "show", f"{STARTER}:train.py"], cwd=ROOT, capture_output=True, text=True,
                      encoding="utf-8", check=True).stdout.replace("\r\n", "\n")
for old, new in EDITS:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"expected one match, found {n}: {old[:60]!r}")
    text = text.replace(old, new)
out.write_text(text, encoding="utf-8", newline="\n")
print(f"{out} written")
