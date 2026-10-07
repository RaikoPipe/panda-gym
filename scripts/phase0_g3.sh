#!/usr/bin/env bash
# Phase 0 / G3: UTD x batch sweep for the frozen semantics, cheapest (low-UTD) cells first.
#   COLLISION_DEF=strict scripts/phase0_g3.sh              # all cells one after another on this machine
#   COLLISION_DEF=strict MODE=sbatch scripts/phase0_g3.sh  # one SLURM GPU job per cell (separate GPUs)
# Each cell trains runs/sweep/<cell>, then evaluates the final model and the validation-best of the last 5
# checkpoints on the VALIDATION scenes (panda_gym/scenes.py) and the final model on reachao3 (in-distribution).
# Summary: python scripts/summarize_g3.py -> runs/sweep/g3_summary.csv
set -eu
cd "$(dirname "$0")/.."
: "${COLLISION_DEF:?set the frozen collision definition (G1/G2 outcome), e.g. COLLISION_DEF=strict}"
MODE=${MODE:-sequential}
STEPS=${STEPS:-1200000}
CURRICULUM=${CURRICULUM:-"reachao1:50:0.9:300000 reachao2:75:0.9:300000 reachao3:100::300000"}
EPISODES=${EPISODES:-200}
module load python/python-3.13.2 2>/dev/null || true
VALIDATION=$(.venv/bin/python -c "from panda_gym.scenes import VALIDATION; print(' '.join(VALIDATION))")
[ -n "$VALIDATION" ] || { echo "panda_gym/scenes.py VALIDATION is empty (scene split not filled in)" >&2; exit 1; }

cell() {  # algo utd batch seed
  local algo=$1 utd=$2 batch=$3 seed=$4 name="g3_$1_utd$2_b$3_s$4" preset=""
  [ "$algo" = tqc ] && preset="--preset thesis_tqc"
  echo "cd $PWD && module load python/python-3.13.2 && set -e
[ -f runs/sweep/$name/done ] && exit 0
.venv/bin/python train.py PandaReachAO-v3 $preset --algo $algo --utd $utd --batch-size $batch --n-envs 16 \
  --seed $seed --steps $STEPS --curriculum $CURRICULUM --collision-def $COLLISION_DEF --n-checkpoints 24 \
  --name sweep/$name
last5=\$(ls runs/sweep/$name/checkpoints/model_reachao3_*_steps.zip | sed 's/.*model_reachao3_\\([0-9]*\\)_steps.zip/\\1 &/' | sort -n | tail -5 | cut -d' ' -f2)
.venv/bin/python evaluate.py runs/sweep/$name/final_reachao3.zip --scenes reachao3 --collision-def $COLLISION_DEF \
  --episodes $EPISODES --workers 16 --out runs/eval/${name}_indist.jsonl
.venv/bin/python evaluate.py runs/sweep/$name/final_reachao3.zip --scenes $VALIDATION --collision-def $COLLISION_DEF \
  --episodes $EPISODES --workers 16 --out runs/eval/${name}_final_val.jsonl
.venv/bin/python evaluate.py \$last5 --select-best --scenes $VALIDATION --collision-def $COLLISION_DEF \
  --episodes $EPISODES --workers 16 --out runs/eval/${name}_last5_val.jsonl
touch runs/sweep/$name/done"
}

mkdir -p runs/sweep runs/eval
for utd in 0.125 0.25 0.5 1; do
  for algo in tqc crossq; do
    for batch in 256 1024; do
      for seed in 0 1; do
        if [ "$MODE" = sbatch ]; then
          sbatch --job-name="g3-$algo-$utd-$batch-$seed" --partition=gpu --gres=gpu:1 --cpus-per-task=20 \
            --time=1-00:00:00 --output="runs/sweep/slurm-%j-$algo-$utd-$batch-$seed.out" \
            --wrap="$(cell $algo $utd $batch $seed)"
        else
          bash -c "$(cell $algo $utd $batch $seed)"
        fi
      done
    done
  done
done
