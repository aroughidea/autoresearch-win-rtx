#!/usr/bin/env bash
# The library's six baseline runs: both datasets x three tokenizers, the starter kit's recipe,
# no agent. Each run's file goes to library/baselines/runs/ and its weights to
# library/checkpoints/ (not committed). About an hour on a 12 GB laptop GPU.
# Run from the repo root: bash library/make_baselines.sh
set -u
mkdir -p library/baselines/runs library/checkpoints library/logs
for dataset in tinystories folktales; do
  for tokenizer in own phi3 gpt2; do
    echo "== $dataset / $tokenizer  $(date +%H:%M)"
    uv run prepare.py --dataset "$dataset" --tokenizer "$tokenizer" > "library/logs/prepare-$dataset-$tokenizer.log" 2>&1 \
      || { echo "prepare failed, see library/logs/prepare-$dataset-$tokenizer.log"; exit 1; }
    marker=$(mktemp)
    uv run train.py > "library/logs/train-$dataset-$tokenizer.log" 2>&1
    grep -E '^(val_bpb|peak_vram_mb|total_seconds):' "library/logs/train-$dataset-$tokenizer.log"
    run=$(find runs -name '*.json' -newer "$marker" | head -1)
    rm -f "$marker"
    if [ -z "$run" ]; then echo "no run file: the run failed, see library/logs/train-$dataset-$tokenizer.log"; continue; fi
    mv "$run" "library/baselines/runs/$dataset-$tokenizer.json"
    cp checkpoint_pre_eval.pt "library/checkpoints/$dataset-$tokenizer.pt"
  done
done
uv run prepare.py --dataset tinystories --tokenizer own > /dev/null 2>&1   # back to the default pair
git checkout -- checkpoint_pre_eval.pt
echo "== done $(date +%H:%M)"
