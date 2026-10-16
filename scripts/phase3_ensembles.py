"""Phase 3: ensemble evaluation of the seed population (RQ2), as a SLURM job array on the CPU partition.

    python scripts/phase3_ensembles.py                 write runs/p3/commands.txt and submit the array
    python scripts/phase3_ensembles.py --dry-run       only write and count the commands

Members: the validation-selected checkpoint of every population seed (runs/pop/<name>/s*/selected.txt, Phase 2).
  K-sweep    K in --k (random subsets: --subsets per K, numpy.random.default_rng(0) via analysis.subsets; one subset
             when K = all members) x --strategies, --episodes per scene (default 200)
  headline   all members and the top --top-k members by Phase 2 validation success x --strategies, 500 episodes
Scenes: VALIDATION + TEST; strict collisions, horizon 150 (configs/paper_frozen.json), episode seeds from 20 000 as in
Phase 2, so single policies (K = 1) and the per-episode oracle come from the Phase 2 records on the same episodes.
Output: runs/eval/p3_<group>_<strategy>_k<K>_<i>.jsonl; finished files are skipped on resubmission.
"""
import argparse
import glob
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.subsets import subsets
from panda_gym import scenes as S

ROOT = Path(__file__).resolve().parent.parent


def members(name):
    paths = sorted(glob.glob(str(ROOT / f"runs/pop/{name}/s*/selected.txt")), key=lambda p: int(Path(p).parent.name[1:]))
    return [Path(p).read_text().strip() for p in paths]


def validation_success(name, collision_def, horizon):
    """Mean validation success per selected checkpoint, from the Phase 2 records."""
    succ = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(str(ROOT / f"runs/eval/p2_{name}_s*_{collision_def}_h{horizon}.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["scene"] in S.VALIDATION:
                succ[r["model"]][r["scene"]].append(r["outcome"] == "success")
    return {m: float(np.mean([np.mean(v) for v in s.values()])) for m, s in succ.items()}


def main(args):
    frozen = json.loads((ROOT / "configs/paper_frozen.json").read_text())
    ev = frozen["evaluation"]
    pool = members(frozen["name"])
    if len(pool) < 2:
        raise SystemExit("no population members with selected.txt (run Phase 2 first)")
    scenes = " ".join(S.VALIDATION + S.TEST)
    common = (f"--scenes {scenes} --collision-def {ev['collision_def']} --horizon {ev['horizon']} "
              f"--episode-seed-base 20000 --workers {args.workers}")
    jobs = []  # (output file, models, strategy, episodes)
    for k, groups in subsets(pool, [k for k in args.k if k <= len(pool)], n=args.subsets).items():
        for i, group in enumerate(groups):
            for strategy in args.strategies:
                jobs.append((f"runs/eval/p3_sweep_{strategy}_k{k}_{i}.jsonl", group, strategy, args.episodes))
    scores = validation_success(frozen["name"], ev["collision_def"], ev["horizon"])
    top = sorted(pool, key=lambda m: scores.get(m, -1), reverse=True)[:args.top_k]
    for group_name, group in (("all", pool), (f"top{args.top_k}", top)):
        for strategy in args.strategies:
            jobs.append((f"runs/eval/p3_headline-{group_name}_{strategy}_k{len(group)}_0.jsonl", group, strategy, 500))
    out_dir = ROOT / "runs/p3"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for out, group, strategy, episodes in jobs:
        part = out.replace(".jsonl", ".part.jsonl")
        lines.append(f"[ -f {out} ] || {{ cd {ROOT} && .venv/bin/python evaluate.py --models {' '.join(group)} "
                     f"--ensemble-strategy {strategy} {common} --episodes {episodes} --out {part} && "
                     f"mv {part.replace('.jsonl', '.run_meta.json')} {out.replace('.jsonl', '.run_meta.json')} && "
                     f"mv {part} {out}; }}")
    (out_dir / "commands.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "top_members.json").write_text(json.dumps({m: scores.get(m) for m in top}, indent=1))
    episodes = sum(e * len(S.VALIDATION + S.TEST) for *_, e in jobs)
    print(f"{len(pool)} members, {len(jobs)} ensemble evaluations, {episodes:,} episodes -> {out_dir}/commands.txt")
    print(f"top-{args.top_k} by validation: {[Path(m).parent.parent.name for m in top]}")
    if args.dry_run:
        return
    n = min(args.array, len(lines))
    script = (f"set -e; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; "
              f"module load python/python-3.13.2; cd {ROOT}; "
              f"awk -v t=$SLURM_ARRAY_TASK_ID -v n={n} '(NR - 1) % n == t' runs/p3/commands.txt | bash -e")
    sub = subprocess.run(["sbatch", "--parsable", f"--array=0-{n - 1}", "--job-name=p3-ens", "--partition=big",
                          f"--cpus-per-task={args.workers}", "--mem=64G", "--time=1-00:00:00",
                          "--output=runs/slurm/p3-%A_%a.out", f"--wrap={script}"],
                         capture_output=True, text=True, check=True)
    print(f"submitted array job {sub.stdout.strip()} with {n} tasks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", nargs="+", type=int, default=[2, 3, 5, 10, 20])
    ap.add_argument("--subsets", type=int, default=20)
    ap.add_argument("--strategies", nargs="+",
                    default=["min_variance", "critic_argmax", "mean", "precision_weighted", "random"])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--array", type=int, default=12, help="array tasks (each runs every n-th command)")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
