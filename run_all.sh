#!/usr/bin/env bash
# Overnight queue: train every task one after another (runs don't share the GPU well), then record the videos.
# ~12 h on an RTX 3070 laptop GPU. Logs: runs/*.log, TensorBoard: tensorboard --logdir runs
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
train() { name=$1; shift; [ -f "runs/$name/done" ] && return; $PY train.py "$@" --name "$name" > "runs/$name.log" 2>&1 && touch "runs/$name/done"; }
mkdir -p runs

# obstacle avoidance: curriculum on random obstacles, then fine-tuning on the benchmark scenes
AO_SCENES="library1 library2 narrow_tunnel workshop"
train ReachAO PandaReachAO-v3 --curriculum reachao1:50:0.9 reachao2:75:0.9 reachao3:100 --steps 1200000 --n-checkpoints 30
for s in $AO_SCENES; do
  train "ReachAO_$s" PandaReachAO-v3 --init runs/ReachAO/final_reachao3.zip --curriculum "$s:150" --steps 150000 --n-checkpoints 6
done

# classic manipulation tasks, roughly by difficulty
train Reach PandaReach-v3 --steps 30000 --learning-starts 2000 --eval-freq 5000 --n-checkpoints 6
train Push PandaPush-v3 --steps 400000
train PickAndPlace PandaPickAndPlace-v3 --steps 1000000
train Slide PandaSlide-v3 --steps 1000000
train Flip PandaFlip-v3 --steps 800000
train Stack PandaStack-v3 --steps 1200000

# videos
for t in Reach Push PickAndPlace Slide Flip Stack; do
  $PY record.py "runs/$t" "Panda$t-v3"
done
for st in reachao1:50 reachao2:75 reachao3:100; do  # curriculum stages, each in its own scene
  $PY record.py runs/ReachAO PandaReachAO-v3 --scenario "${st%:*}" --max-episode-steps "${st#*:}" \
    --checkpoint-filter "${st%:*}" --model "final_${st%:*}.zip"
done
for s in $AO_SCENES; do
  $PY record.py runs/ReachAO PandaReachAO-v3 --scenario "$s" --max-episode-steps 150 --model final_reachao3.zip \
    --checkpoint-filter none --name "ReachAO_${s}_zeroshot"  # generalist on the unseen scene
  $PY record.py "runs/ReachAO_$s" PandaReachAO-v3 --scenario "$s" --max-episode-steps 150 --model "final_$s.zip" \
    --name "ReachAO_${s}_finetuned"
done
