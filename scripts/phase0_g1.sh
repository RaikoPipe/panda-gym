#!/usr/bin/env bash
# Phase 0 / G1: evaluation-only ablation of the collision definition and the horizon (no retraining).
#   AGENTS="runs/cmp/*/final_reachao3.zip runs/ReachAO/final_reachao3.zip" scripts/phase0_g1.sh
#   sbatch --partition=big --cpus-per-task=32 --time=24:00:00 scripts/phase0_g1.sh   (CPU only; set AGENTS in the env)
# One file per agent x definition x horizon in runs/eval/g1_*.jsonl; existing files are skipped (resumable).
# Horizons are control steps of 40 ms: 100 = 4 s (the thesis episode length at both of its control rates:
# 400 x 10 ms or 100 x 40 ms), 150 = 6 s (HEAD default), 400 = 16 s (design v4).
#SBATCH --job-name=g1-eval
#SBATCH --output=runs/slurm-%j-g1.out
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # small helper calls; avoids the per-user thread limit
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
ROOT=$(pwd -P)  # re-entered before every run: on BeeGFS a long job's working directory can go stale on a compute
                # node ("failed to make path absolute" in every new Python process)
module load python/python-3.13.2 2>/dev/null || true
PY=.venv/bin/python
AGENTS=${AGENTS:-"runs/cmp/*/final_reachao3.zip runs/ReachAO/final_reachao3.zip"}
DEFS=${DEFS:-"strict thesis strict_notable thesis_table thesis_substep"}
HORIZONS=${HORIZONS:-"100 150 400"}
EPISODES=${EPISODES:-200}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
mkdir -p runs/eval
for agent in $AGENTS; do
  [ -f "$agent" ] || { echo "missing agent $agent" >&2; exit 1; }
  tag=$(echo "${agent#runs/}" | sed 's#/#_#g; s#\.zip$##')
  for def in $DEFS; do
    for h in $HORIZONS; do
      out="runs/eval/g1_${tag}_${def}_h${h}.jsonl"
      [ -f "$out" ] && continue
      cd "$ROOT"
      $PY evaluate.py "$agent" --scenes library1 library2 narrow_tunnel tunnel workshop workshop2 industrial bookshelves \
        --collision-def "$def" --horizon "$h" --episodes "$EPISODES" --workers "$WORKERS" --out "${out%.jsonl}.part.jsonl" \
        && mv "${out%.jsonl}.part.run_meta.json" "${out%.jsonl}.run_meta.json" && mv "${out%.jsonl}.part.jsonl" "$out"
    done
  done
done
cd "$ROOT" && PYTHONPATH=. $PY scripts/summarize_g1.py
