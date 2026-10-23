#!/usr/bin/env bash
# Phase 1 (T10): seed population of the frozen configuration.
#   scripts/phase1_seeds.sh                      seeds 0-19 -> runs/pop/<name>/s<seed>/, one after another
#   MODE=sbatch scripts/phase1_seeds.sh          one SLURM GPU job per seed (separate GPUs; no time-slicing)
#   scripts/phase1_seeds.sh --population xdist   5 seeds each of the stage-3 distributions, all fine-tuned from
#                                                one shared end-of-stage-2 checkpoint (trained first if missing)
#   DRY_RUN=1 ...                                print the commands only
# Reads configs/paper_frozen.json (format: configs/paper_frozen.example.json; CONFIG=... to override).
# Seeds whose run directory holds a "done" marker are skipped. Every run saves 24 checkpoints (~every 50 k steps).
# GPU jobs avoid gpu01-gpu03 (reserved for other work); override with EXCLUDE_NODES=<slurm node list>.
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # small helper calls; avoids the per-user thread limit
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/paper_frozen.json}
MODE=${MODE:-sequential}
SEEDS=${SEEDS:-$(seq 0 19)}
POPULATION=main
[ "${1:-}" = "--population" ] && POPULATION=${2:?main or xdist}
[ -f "$CONFIG" ] || { echo "$CONFIG missing: freeze the configuration first (G4)" >&2; exit 1; }
module load python/python-3.13.2 2>/dev/null || true
PY=.venv/bin/python
cfg() { $PY -c "import json, shlex, sys; c = json.load(open('$CONFIG')); v = eval(sys.argv[1], {}, {'c': c}); \
print(' '.join(map(shlex.quote, map(str, v))) if isinstance(v, list) else v)" "$1"; }
NAME=$(cfg 'c["name"]')
ENV=$(cfg 'c["env"]')

run() {  # run_dir command...
  local dir=$1; shift
  [ -f "$dir/done" ] && { echo "skip $dir (done)"; return 0; }
  local cmd="cd $PWD && module load python/python-3.13.2 2>/dev/null; $* && touch $dir/done"
  if [ -n "${DRY_RUN:-}" ]; then echo "$cmd"
  elif [ "$MODE" = sbatch ]; then
    mkdir -p runs/slurm
    sbatch --parsable --job-name="pop-$(basename "$(dirname "$dir")")-$(basename "$dir")" --partition=gpu --exclude="${EXCLUDE_NODES-gpu[01-03]}" \
      --gres=gpu:1 --cpus-per-task=16 --mem=48G --time=1-00:00:00 --output="runs/slurm/pop-%j.out" ${DEPEND:-} \
      --wrap="$cmd"
  else bash -c "$cmd"; fi
}

if [ "$POPULATION" = main ]; then
  for seed in $SEEDS; do
    run "runs/pop/$NAME/s$seed" $PY train.py "$ENV" $(cfg 'c["train_args"]') --steps "$(cfg 'c["steps"]')" \
      --n-checkpoints 24 --seed "$seed" --name "pop/$NAME/s$seed"
  done
else
  stage2="runs/pop/${NAME}_xdist/stage2"
  # later --curriculum flags override the frozen one (argparse keeps the last occurrence)
  stage2_cmd=($PY train.py "$ENV" $(cfg 'c["train_args"]') $(cfg 'c["xdist"]["stage2_args"]')
              --steps "$(cfg 'c["xdist"]["stage2_steps"]')" --seed 1000 --name "pop/${NAME}_xdist/stage2")
  if [ "$MODE" = sbatch ] && [ -z "${DRY_RUN:-}" ]; then
    job=$(run "$stage2" "${stage2_cmd[@]}")
    [[ $job =~ ^[0-9]+$ ]] && DEPEND="--dependency=afterok:$job"  # stage-3 jobs wait for the shared checkpoint
  else
    run "$stage2" "${stage2_cmd[@]}"
  fi
  init="$stage2/final_reachao2.zip"
  for dist in $(cfg 'c["xdist"]["distributions"]'); do
    for seed in $(seq 0 $(( $(cfg 'c["xdist"]["seeds"]') - 1 ))); do
      run "runs/pop/${NAME}_xdist/$dist/s$seed" $PY train.py "$ENV" $(cfg 'c["train_args"]') --init "$init" \
        --curriculum "$dist:$(cfg 'c["xdist"]["stage3_horizon"]')" --steps "$(cfg 'c["xdist"]["stage3_steps"]')" \
        --n-checkpoints 24 --seed "$seed" --name "pop/${NAME}_xdist/$dist/s$seed"
    done
  done
fi
