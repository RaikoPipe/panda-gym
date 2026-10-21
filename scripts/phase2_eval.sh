#!/usr/bin/env bash
# Phase 2: single-policy evaluation of the seed population, and NEO on the same episodes.
#   scripts/phase2_eval.sh            one CPU job per seed (waits for the seed's training job if it is still queued
#                                     or running), plus one NEO job
#   SEEDS="0 1" scripts/phase2_eval.sh   NEO=0 ...   skip NEO      DRY_RUN=1 ...   print the job scripts
#   POP=<name>_xdist/reachao3 SEEDS="0 1 2 3 4" CONDITIONS="strict:150" NEO=0 ...   another population, one condition
# Per seed (configs/paper_frozen.json):
#   1. checkpoint selection: the best of the last 5 checkpoints on VALIDATION (200 episodes per scene, episode seeds
#      from 10_000, strict, horizon 150) -> runs/pop/<name>/s<seed>/selected.txt
#   2. the selected checkpoint on reachao3 + VALIDATION + TEST, 500 episodes per scene, episode seeds from 20_000
#      (disjoint from the selection episodes, so validation results are not selection-biased), for the strict and
#      the thesis collision definition at horizons 150 and 400 -> runs/eval/p2_<name>_s<seed>_<def>_h<h>.jsonl
# NEO: the same scenes, episodes, definitions and horizons -> runs/eval/p2_neo_<def>_h<h>.jsonl
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # small helper calls; avoids the per-user thread limit
cd "$(dirname "$0")/.."
ROOT=$(pwd -P)
module load python/python-3.13.2 2>/dev/null || true
PY=.venv/bin/python
CONFIG=${CONFIG:-configs/paper_frozen.json}
NAME=$($PY -c "import json; print(json.load(open('$CONFIG'))['name'])")
POP=${POP:-$NAME}  # population directory under runs/pop, e.g. POP=${NAME}_xdist/reachao3
OUT=${POP//\//-}  # output name stem
CONDITIONS=${CONDITIONS:-"strict:150 strict:400 thesis:150 thesis:400"}
SEEDS=${SEEDS:-$(seq 0 19)}
read -r VALIDATION TEST <<< "$($PY -c "from panda_gym import scenes as S; print(','.join(S.VALIDATION), ','.join(S.TEST))")"
VALIDATION=${VALIDATION//,/ }
TEST=${TEST//,/ }
SCENES=${SCENES:-"reachao3 $VALIDATION $TEST"}  # override + TAG=<name> to (re-)evaluate a subset into separate files
EVAL="--episodes 500 --episode-seed-base 20000 --workers 32"
mkdir -p runs/slurm runs/eval

submit() {  # name dependency script
  if [ -n "${DRY_RUN:-}" ]; then printf '### %s %s\n%s\n' "$1" "$2" "$3"; return; fi
  sbatch --parsable --job-name="$1" --partition=big --cpus-per-task=32 --mem=64G --time=1-00:00:00 \
    --output="runs/slurm/$1-%j.out" $2 --wrap="$3"
}

for seed in $SEEDS; do
  run="runs/pop/$POP/s$seed"
  job=$(squeue -u "$USER" -h -n "pop-$(basename "$POP")-s$seed" -o %i | head -1)  # job names of phase1_seeds.sh
  dep=${job:+--dependency=afterok:$job}
  if [ -z "$job" ] && [ ! -f "$run/done" ]; then echo "s$seed: not trained and no training job, skipped" >&2; continue; fi
  script="set -e; module load python/python-3.13.2
cd $ROOT
if [ ! -f $run/selected.txt ]; then
  if [ ! -f runs/eval/p2_${OUT}_s${seed}_select.jsonl ]; then
    last5=\$(ls $run/checkpoints/model_reachao3_*_steps.zip | sed 's/.*model_reachao3_\([0-9]*\)_steps.zip/\1 &/' | sort -n | tail -5 | cut -d' ' -f2)
    $PY evaluate.py \$last5 --select-best --scenes $VALIDATION --collision-def strict --horizon 150 --episodes 200 \
      --episode-seed-base 10000 --workers 32 --out runs/eval/p2_${OUT}_s${seed}_select.jsonl
  fi
  cd $ROOT  # re-entered before every Python process (stale working directories on BeeGFS)
  $PY -c \"import json, collections
succ = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open('runs/eval/p2_${OUT}_s${seed}_select.jsonl'):
    r = json.loads(line); succ[r['model']][r['scene']].append(r['outcome'] == 'success')
score = {m: sum(sum(v) / len(v) for v in s.values()) / len(s) for m, s in succ.items()}
open('$run/selected.txt', 'w').write(max(score, key=score.get) + '\n')\"
fi
cd $ROOT
model=\$(cat $run/selected.txt)
for c in $CONDITIONS; do def=\${c%:*}; h=\${c#*:}
  out=runs/eval/p2_${OUT}_s${seed}${TAG:+_$TAG}_\${def}_h\${h}.jsonl
  [ -f \$out ] && continue
  cd $ROOT
  $PY evaluate.py \$model --scenes $SCENES --collision-def \$def --horizon \$h $EVAL --out \${out%.jsonl}.part.jsonl
  mv \${out%.jsonl}.part.run_meta.json \${out%.jsonl}.run_meta.json && mv \${out%.jsonl}.part.jsonl \$out
done"
  echo "s$seed: eval job $(submit "p2${TAG:+-$TAG}-$(basename "$POP")-s$seed" "$dep" "$script")${job:+ (after training job $job)}"
done

if [ "${NEO:-1}" = 1 ]; then
  script="set -e; module load python/python-3.13.2
for c in $CONDITIONS; do def=\${c%:*}; h=\${c#*:}
  out=runs/eval/p2_neo${TAG:+_$TAG}_\${def}_h\${h}.jsonl
  [ -f \$out ] && continue
  cd $ROOT
  $PY evaluate.py --controller neo --scenes $SCENES --collision-def \$def --horizon \$h $EVAL --out \${out%.jsonl}.part.jsonl
  mv \${out%.jsonl}.part.run_meta.json \${out%.jsonl}.run_meta.json && mv \${out%.jsonl}.part.jsonl \$out
done"
  echo "NEO: eval job $(submit "p2${TAG:+-$TAG}-neo" "" "$script")"
fi
