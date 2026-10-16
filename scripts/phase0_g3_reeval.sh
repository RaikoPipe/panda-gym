#!/usr/bin/env bash
# Re-evaluate finished G3 runs on the current VALIDATION scenes (panda_gym/scenes.py): the final model and the last
# 5 checkpoints of every runs/sweep/g3_*/ with a "done" marker. Needed after the validation set changed (raised_shelves
# and tabletop fixed mid-sweep, kitchen and countertop dropped, 2026-10-13). Old records move to runs/eval/g3_superseded/.
#   sbatch --partition=big --cpus-per-task=32 --mem=64G --time=12:00:00 --output=runs/slurm/g3-reeval-%j.out \
#     --export=ALL,COLLISION_DEF=strict scripts/phase0_g3_reeval.sh
#SBATCH --job-name=g3-reeval
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # small helper calls; avoids the per-user thread limit
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
ROOT=$(pwd -P)  # re-entered before every evaluation (stale working directories on BeeGFS)
: "${COLLISION_DEF:?e.g. COLLISION_DEF=strict}"
EPISODES=${EPISODES:-200}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
module load python/python-3.13.2 2>/dev/null || true
VALIDATION=$(.venv/bin/python -c "from panda_gym.scenes import VALIDATION; print(' '.join(VALIDATION))")
mkdir -p runs/eval/g3_superseded
for run in runs/sweep/g3_*/; do
  name=$(basename "$run")
  [ -f "$run/done" ] || continue
  for kind in final_val last5_val; do
    out="runs/eval/${name}_${kind}.jsonl"
    if [ -f "$out" ] && grep -q '"scene": "kitchen"' "$out"; then  # evaluated on the old validation set
      mv "$out" "${out%.jsonl}.run_meta.json" runs/eval/g3_superseded/ 2>/dev/null || true
    fi
    [ -f "$out" ] && continue
    cd "$ROOT"
    if [ $kind = final_val ]; then
      models="$run/final_reachao3.zip"; select=""
    else
      models=$(ls "$run"/checkpoints/model_reachao3_*_steps.zip | sed 's/.*model_reachao3_\([0-9]*\)_steps.zip/\1 &/' \
               | sort -n | tail -5 | cut -d' ' -f2); select="--select-best"
    fi
    .venv/bin/python evaluate.py $models $select --scenes $VALIDATION --collision-def "$COLLISION_DEF" \
      --episodes "$EPISODES" --workers "$WORKERS" --out "${out%.jsonl}.part.jsonl"
    mv "${out%.jsonl}.part.run_meta.json" "${out%.jsonl}.run_meta.json" && mv "${out%.jsonl}.part.jsonl" "$out"
  done
done
