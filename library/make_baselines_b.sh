#!/usr/bin/env bash
# The recipe B baselines: both datasets x three tokenizers, recipe B (library/recipes/recipe-b.py,
# the starter recipe plus the TinyStories night's three attention changes), 10 minutes each, no agent.
# Preparing the data is kept apart from training, so training never downloads or builds anything:
#
#   bash library/make_baselines_b.sh prep    download or build each pair's data and vocabulary; trains nothing
#   bash library/make_baselines_b.sh check   confirm every pair is already prepared; trains nothing
#   bash library/make_baselines_b.sh train   check, then a speed check and the six runs (about 1 hour 50 minutes)
#   DRY_RUN=1 bash library/make_baselines_b.sh train    print the planned runs; touches nothing
#
# Run files go to library/baselines-b-10min/runs/<dataset>-<tokenizer>.json, weights to
# library/checkpoints/b10-<dataset>-<tokenizer>.pt (not committed), logs to library/logs/b10-*.log.
# Nothing else may train meanwhile: the active dataset is machine-wide. Run from the repo root.
set -u
STARTER=6ad8ddd                        # train.py with the starter kit's recipe (library/README.md)
RECIPE_B=library/recipes/recipe-b.py   # version c9ba08e4f442
OUT=library/baselines-b-10min/runs
FLOOR_A=1190                           # steps at 600 s, TinyStories / own: 95% of 1,258 (12 October)
FLOOR_B=2980                           # 95% of 3,137, the TinyStories session's equivalent version
PAIRS="tinystories:own tinystories:phi3 tinystories:gpt2 folktales:own folktales:phi3 folktales:gpt2"

# A prepare.py log in which nothing was downloaded or built: the pair was already prepared.
log_is_cached() {
  grep -q "^Tokenizer: already trained at" "$1" \
    && ! grep -q -E "^Data: (downloading|folktales ->|migrated|copied)|^Tokenizer: training" "$1"
}

version_of() { uv run python -c "import record, sys; print(record.version_id(sys.stdin.read()))"; }

prepare_pairs() {  # mode: prep (anything missing is fetched) or check (nothing may be missing)
  local mode=$1 pair dataset tokenizer log missing=0
  mkdir -p library/logs
  for pair in $PAIRS; do
    dataset=${pair%%:*}; tokenizer=${pair##*:}
    log=library/logs/b10-prepare-$dataset-$tokenizer.log
    if ! uv run prepare.py --dataset "$dataset" --tokenizer "$tokenizer" > "$log" 2>&1; then
      echo "prepare failed: $dataset / $tokenizer, see $log"; return 1
    fi
    if log_is_cached "$log"; then echo "ready: $dataset / $tokenizer"
    elif [ "$mode" = prep ]; then echo "prepared now: $dataset / $tokenizer"
    else echo "not prepared before this check: $dataset / $tokenizer (see $log)"; missing=1
    fi
  done
  uv run prepare.py --dataset tinystories --tokenizer own > /dev/null 2>&1   # back to the default pair
  echo "active: tinystories / own"
  if [ "$missing" = 1 ]; then echo "run prep first, so that training never downloads or builds anything"; return 1; fi
}

one_run() {  # dataset tokenizer recipe-file destination floor
  local dataset=$1 tokenizer=$2 recipe=$3 dest=$4 floor=$5 name marker run
  name=$(basename "$dest" .json)
  echo "== $name ($dataset / $tokenizer, recipe $(version_of < "$recipe"))  $(date +%H:%M)"
  [ "${DRY_RUN:-}" = 1 ] && { echo "   -> $dest, speed floor $floor"; return 0; }
  cp "$recipe" train.py
  uv run prepare.py --dataset "$dataset" --tokenizer "$tokenizer" > /dev/null 2>&1 || { echo "prepare failed"; return 1; }
  marker=$(mktemp)
  uv run train.py > "library/logs/b10-train-$name.log" 2>&1
  grep -E '^(val_bpb|num_steps|training_seconds):' "library/logs/b10-train-$name.log"
  run=$(find runs -name '*.json' -newer "$marker" | head -1); rm -f "$marker"
  if [ -z "$run" ]; then echo "no run file: see library/logs/b10-train-$name.log"; return 1; fi
  mkdir -p "$(dirname "$dest")" library/checkpoints
  mv "$run" "$dest"
  case "$dest" in "$OUT"/*) cp checkpoint_pre_eval.pt "library/checkpoints/b10-$name.pt" ;; esac   # the collection's weights
  [ "$floor" = - ] || uv run library/speed_gate.py "$floor" "$dest"
}

train() {
  local starter_py pair dataset tokenizer floor
  if [ "${DRY_RUN:-}" != 1 ]; then prepare_pairs check || return 1; fi
  export AUTORESEARCH_TIME_BUDGET=600
  starter_py=$(mktemp)
  git show "$STARTER:train.py" > "$starter_py"
  # First the starter recipe once, as a speed check against 12 October; not part of the collection.
  one_run tinystories own "$starter_py" library/logs/b10-speed-check-a.json "$FLOOR_A" || { rm -f "$starter_py"; return 1; }
  rm -f "$starter_py"
  for pair in $PAIRS; do
    dataset=${pair%%:*}; tokenizer=${pair##*:}
    floor=-; [ "$pair" = tinystories:own ] && floor=$FLOOR_B   # the one pair with a known speed
    one_run "$dataset" "$tokenizer" "$RECIPE_B" "$OUT/$dataset-$tokenizer.json" "$floor" || return 1
  done
}

main() {
  case "${1:-}" in
    prep) prepare_pairs prep ;;
    check) prepare_pairs check ;;
    train)
      train; status=$?
      if [ "${DRY_RUN:-}" != 1 ]; then
        git checkout -- train.py
        uv run prepare.py --dataset tinystories --tokenizer own > /dev/null 2>&1   # back to the default pair
      fi
      unset AUTORESEARCH_TIME_BUDGET
      echo "== done $(date +%H:%M)"
      return $status ;;
    *) echo "usage: bash library/make_baselines_b.sh prep|check|train"; return 2 ;;
  esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; exit $?; fi
