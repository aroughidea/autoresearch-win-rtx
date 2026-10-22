#!/usr/bin/env bash
# The recipe B baselines: both datasets x three tokenizers, recipe B (library/recipes/recipe-b.py,
# the starter recipe plus the TinyStories night's three attention changes), no agent.
# Preparing the data is kept apart from training, so training never downloads or builds anything:
#
#   bash library/make_baselines_b.sh prep    download or build each pair's data and vocabulary; trains nothing
#   bash library/make_baselines_b.sh check   confirm every pair is already prepared; trains nothing
#   bash library/make_baselines_b.sh train   check, then a speed check and the six 10-minute runs (about 1 h 50 min)
#   MINUTES=5 PAIRS="tinystories:own folktales:own" bash library/make_baselines_b.sh train
#                                            the 5-minute pair, for 5 against 10 minutes (about 30 min)
#   DRY_RUN=1 bash library/make_baselines_b.sh train    print the planned runs; touches nothing
#
# Run files go to library/baselines-b-<minutes>min/runs/<dataset>-<tokenizer>.json, weights to
# library/checkpoints/b<minutes>-<dataset>-<tokenizer>.pt (not committed), logs to library/logs/b<minutes>-*.log.
# Nothing else may train meanwhile: the active dataset is machine-wide. Run from the repo root.
set -u
STARTER=6ad8ddd                        # train.py with the starter kit's recipe (library/README.md)
RECIPE_B=library/recipes/recipe-b.py   # version c9ba08e4f442
MINUTES=${MINUTES:-10}
case "$MINUTES" in 5|10) ;; *) echo "MINUTES must be 5 or 10, not $MINUTES"; exit 2 ;; esac
TAG=b$MINUTES
OUT=library/baselines-b-${MINUTES}min/runs
FLOOR_A=1190                           # steps at 600 s, TinyStories / own: 95% of 1,258 (12 October)
FLOOR_B=2980                           # 95% of 3,137, the TinyStories session's equivalent version
ALL_PAIRS="tinystories:own tinystories:phi3 tinystories:gpt2 folktales:own folktales:phi3 folktales:gpt2"
PAIRS=${PAIRS:-$ALL_PAIRS}

# A prepare.py log in which nothing was downloaded or built: the pair was already prepared.
log_is_cached() {
  grep -q "^Tokenizer: already trained at" "$1" \
    && ! grep -q -E "^Data: (downloading|folktales ->|migrated|copied)|^Tokenizer: training" "$1"
}

# record.version_id: the first 12 hex digits of the SHA-256 of the code with line endings made uniform.
version_of() { tr -d '\r' < "$1" | sha256sum | cut -c1-12; }

# The 5-minute TinyStories / own floor: 47.5% of the 10-minute run's steps (half, less 5%), once that has run.
floor_b5() {
  local ten=${B10_TINYSTORIES:-library/baselines-b-10min/runs/tinystories-own.json} steps
  steps=$(grep -o '"num_steps": *[0-9]*' "$ten" 2>/dev/null | head -1 | grep -o '[0-9]*$')
  if [ -n "$steps" ]; then echo $((steps * 475 / 1000)); else echo -; fi
}

prepare_pairs() {  # mode: prep (anything missing is fetched) or check (nothing may be missing)
  local mode=$1 pair dataset tokenizer log missing=0
  mkdir -p library/logs
  for pair in $ALL_PAIRS; do
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
  echo "== $name ($dataset / $tokenizer, recipe $(version_of "$recipe"))  $(date +%H:%M)"
  [ "${DRY_RUN:-}" = 1 ] && { echo "   -> $dest, speed floor $floor"; return 0; }
  cp "$recipe" train.py
  uv run prepare.py --dataset "$dataset" --tokenizer "$tokenizer" > /dev/null 2>&1 || { echo "prepare failed"; return 1; }
  marker=$(mktemp)
  uv run train.py > "library/logs/$TAG-train-$name.log" 2>&1
  grep -E '^(val_bpb|num_steps|training_seconds):' "library/logs/$TAG-train-$name.log"
  run=$(find runs -name '*.json' -newer "$marker" | head -1); rm -f "$marker"
  if [ -z "$run" ]; then echo "no run file: see library/logs/$TAG-train-$name.log"; return 1; fi
  mkdir -p "$(dirname "$dest")" library/checkpoints
  mv "$run" "$dest"
  case "$dest" in "$OUT"/*) cp checkpoint_pre_eval.pt "library/checkpoints/$TAG-$name.pt" ;; esac   # the collection's weights
  [ "$floor" = - ] || uv run library/speed_gate.py "$floor" "$dest"
}

train() {
  local starter_py pair dataset tokenizer floor floor_ts
  echo "Recipe B baselines: $MINUTES min each, into $OUT"
  if [ "${DRY_RUN:-}" != 1 ]; then prepare_pairs check || return 1; fi
  export AUTORESEARCH_TIME_BUDGET=$((MINUTES * 60))
  if [ "$MINUTES" = 10 ]; then
    floor_ts=$FLOOR_B
    starter_py=$(mktemp)
    git show "$STARTER:train.py" > "$starter_py"
    # First the starter recipe once, as a speed check against 12 October; not part of the collection.
    one_run tinystories own "$starter_py" library/logs/b10-speed-check-a.json "$FLOOR_A" || { rm -f "$starter_py"; return 1; }
    rm -f "$starter_py"
  else
    floor_ts=$(floor_b5)   # gated against tonight's 10-minute run instead of a separate check
  fi
  for pair in $PAIRS; do
    dataset=${pair%%:*}; tokenizer=${pair##*:}
    floor=-; [ "$pair" = tinystories:own ] && floor=$floor_ts   # the one pair with a known speed
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
