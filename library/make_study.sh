#!/usr/bin/env bash
# The ten-minute study: the six baselines (both datasets x three tokenizers, the starter kit's
# recipe) and the Folktales agent session's best recipe, each trained for 10 minutes instead of 5.
# Run files go to library/study-10min/runs/, weights to library/checkpoints/study-*.pt (not
# committed). About 1 hour 45 minutes on a 12 GB laptop GPU. Nothing else may train meanwhile:
# the active dataset is machine-wide.
# Run from the repo root: bash library/make_study.sh
set -u
STARTER=6ad8ddd     # train.py with the starter kit's recipe (library/README.md)
AGENT_BEST=f9352a8  # the Folktales session's best recipe: MATRIX_LR 0.20
export AUTORESEARCH_TIME_BUDGET=600
mkdir -p library/study-10min/runs library/checkpoints library/logs

one_run() {  # dataset tokenizer recipe-commit name
  local dataset=$1 tokenizer=$2 recipe=$3 name=$4
  echo "== $name  ($dataset / $tokenizer, recipe $recipe)  $(date +%H:%M)"
  git show "$recipe:train.py" > train.py
  uv run prepare.py --dataset "$dataset" --tokenizer "$tokenizer" > "library/logs/study-prepare-$name.log" 2>&1 \
    || { echo "prepare failed, see library/logs/study-prepare-$name.log"; return 1; }
  local marker; marker=$(mktemp)
  uv run train.py > "library/logs/study-train-$name.log" 2>&1
  grep -E '^(val_bpb|total_seconds|training_seconds):' "library/logs/study-train-$name.log"
  local run; run=$(find runs -name '*.json' -newer "$marker" | head -1); rm -f "$marker"
  if [ -z "$run" ]; then echo "no run file: see library/logs/study-train-$name.log"; return 0; fi
  mv "$run" "library/study-10min/runs/$name.json"
  cp checkpoint_pre_eval.pt "library/checkpoints/study-$name.pt"
}

for dataset in tinystories folktales; do
  for tokenizer in own phi3 gpt2; do
    one_run "$dataset" "$tokenizer" "$STARTER" "$dataset-$tokenizer" || exit 1
  done
done
one_run folktales own "$AGENT_BEST" folktales-own-agent-best || exit 1

git checkout -- train.py checkpoint_pre_eval.pt
unset AUTORESEARCH_TIME_BUDGET
uv run prepare.py --dataset tinystories --tokenizer own > /dev/null 2>&1   # back to the default pair
echo "== done $(date +%H:%M)"
