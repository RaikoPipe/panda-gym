#!/usr/bin/env bash
# Phase 0 / G2: training-semantics ablation. The thesis TQC preset (train.py --preset thesis_tqc, 10 ms control
# steps) trained under
#   head:   strict collision definition, terminate on collision, HER penalty from the stored per-transition flag
#   thesis: thesis collision definition, truncate on collision, live-flag HER penalty (thesis bug, fidelity only)
# 3 seeds each, one SLURM GPU job per run; then each agent is evaluated under both collision definitions on the
# G1 scenes plus reachao3, at 400 control steps of 10 ms (4 s, as G1's horizon 100 and the thesis evaluation).
#   scripts/phase0_g2.sh            submit everything (training jobs + dependent evaluation jobs)
# Summary: PYTHONPATH=. .venv/bin/python scripts/summarize_g1.py runs/eval/g2_*.jsonl --out runs/eval/g2_summary.csv
set -eu
cd "$(dirname "$0")/.."
STEPS=${STEPS:-900000}  # the preset caps each of the three curriculum stages at 300 k steps
SEEDS=${SEEDS:-"0 1 2"}
EPISODES=${EPISODES:-200}
SCENES="reachao3 library1 library2 narrow_tunnel tunnel workshop workshop2 industrial bookshelves"
declare -A SEMANTICS=(
  [head]="--collision-def strict --collision-semantics terminate --her-penalty-source info"
  [thesis]="--collision-def thesis --collision-semantics truncate --her-penalty-source live"
)
mkdir -p runs/slurm runs/eval
for sem in head thesis; do
  for seed in $SEEDS; do
    name="g2_${sem}_s${seed}"
    [ -f "runs/$name/done" ] && continue
    train=$(sbatch --parsable --job-name="$name" --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=48G \
      --time=1-00:00:00 --output="runs/slurm/train-$name-%j.out" --wrap="cd $PWD && module load python/python-3.13.2 \
      && .venv/bin/python train.py PandaReachAO-v3 --preset thesis_tqc ${SEMANTICS[$sem]} --steps $STEPS \
      --n-checkpoints 18 --seed $seed --name $name && touch runs/$name/done")
    evals=""
    for def in strict thesis; do
      evals="$evals cd $PWD && .venv/bin/python evaluate.py runs/$name/final_reachao3.zip --scenes $SCENES --collision-def $def \
        --n-substeps 5 --horizon 400 --episodes $EPISODES --workers 32 --out runs/eval/${name}_$def.jsonl &&"
    done
    ev=$(sbatch --parsable --dependency=afterok:$train --job-name="eval-$name" --partition=big --cpus-per-task=32 \
      --mem=64G --time=12:00:00 --output="runs/slurm/eval-$name-%j.out" \
      --wrap="cd $PWD && module load python/python-3.13.2 && $evals true")
    echo "$name: train $train, eval $ev"
  done
done
