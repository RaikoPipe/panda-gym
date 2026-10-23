"""Phase 3: ensemble evaluation of the seed population (RQ2), as a SLURM job array on the CPU partition.

    python scripts/phase3_ensembles.py                 write runs/p3/commands.txt and submit the array
    python scripts/phase3_ensembles.py --dry-run       only write and count the commands

Members: the validation-selected checkpoint of every population seed (runs/pop/<name>/s*/selected.txt, Phase 2).
  K-sweep    K in --k (random subsets: --subsets per K, numpy.random.default_rng(0) via analysis.subsets; one subset
             when K = all members) x --strategies, --episodes per scene (default 200)
  headline   all members and the top --top-k members by Phase 2 validation success x --strategies, 500 episodes
  conditions the all-members headline under the secondary collision definition and at horizon 400 (p3c_* files)
  snapshot   RQ3: the last --snapshot-k checkpoints of single runs (--snapshot-seeds) x --snapshot-strategies
  xdist      RQ3: cross-distribution population (runs/pop/<name>_xdist/<dist>/s*/selected.txt), pure vs. mixed
             ensembles of equal size (see xdist_groups) x --snapshot-strategies (p3x_* files)
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


def xdist_members(name):
    """{distribution: validation-selected checkpoints of its runs} for the cross-distribution population."""
    out = {}
    for dist_dir in sorted((ROOT / f"runs/pop/{name}_xdist").iterdir()):
        sel = sorted(dist_dir.glob("s*/selected.txt"), key=lambda p: int(p.parent.name[1:]))
        if sel:
            out[dist_dir.name] = [p.read_text().strip() for p in sel]
    return out


def xdist_groups(by_dist, n=20, seed=0):
    """Matched comparison (RQ3): ensembles from one training distribution vs. mixtures of equal size.
    pure-<dist> K=5 (all runs of a distribution); pure-reachao3 K=3 (all 3-subsets: the matched control for mixed K=3);
    mixed K=3 (one run per distribution), mixed K=5 (2 + 2 + 1 across distributions), mixed K=15 (all)."""
    from itertools import combinations
    rng = np.random.default_rng(seed)
    dists = sorted(by_dist)
    groups = [(f"pure-{d}", [m]) for d, m in ((d, by_dist[d]) for d in dists)]
    groups += [("pure-reachao3", [list(c) for c in combinations(by_dist["reachao3"], 3)])]
    mixed3 = [[str(rng.choice(by_dist[d])) for d in dists] for _ in range(n)]
    mixed5 = []
    for i in range(n):
        single = dists[i % len(dists)]  # the distribution that contributes one run (rotates)
        mixed5.append([str(m) for d in dists for m in rng.choice(by_dist[d], 1 if d == single else 2, replace=False)])
    groups += [("mixed", mixed3), ("mixed", mixed5), ("mixed", [[m for d in dists for m in by_dist[d]]])]
    return [(name, group if isinstance(group[0], list) else group) for name, group in groups]


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
    strict, h = ev["collision_def"], ev["horizon"]
    jobs = []  # (output file, models, strategy, episodes, collision definition, horizon)
    if "sweep" in args.groups:
        for k, groups in subsets(pool, [k for k in args.k if k <= len(pool)], n=args.subsets).items():
            for i, group in enumerate(groups):
                for strategy in args.strategies:
                    jobs.append((f"runs/eval/p3_sweep_{strategy}_k{k}_{i}.jsonl", group, strategy, args.episodes, strict, h))
    scores = validation_success(frozen["name"], strict, h)
    top = sorted(pool, key=lambda m: scores.get(m, -1), reverse=True)[:args.top_k]
    if "headline" in args.groups:
        for group_name, group in (("all", pool), (f"top{args.top_k}", top)):
            for strategy in args.strategies:
                jobs.append((f"runs/eval/p3_headline-{group_name}_{strategy}_k{len(group)}_0.jsonl", group, strategy, 500,
                             strict, h))
    if "conditions" in args.groups:  # secondary columns: thesis collision definition, horizon 400
        for cdef, horizon in ((strict, 400), (ev["secondary_collision_def"], h), (ev["secondary_collision_def"], 400)):
            for strategy in args.strategies:
                jobs.append((f"runs/eval/p3c_{cdef}_h{horizon}_headline-all_{strategy}_k{len(pool)}_0.jsonl", pool,
                             strategy, 500, cdef, horizon))
    if "snapshot" in args.groups:  # RQ3: the last K checkpoints of one run (SEERL), same episodes as the K-sweep
        for seed in args.snapshot_seeds:
            run = ROOT / f"runs/pop/{frozen['name']}/s{seed}"
            ckpts = sorted(run.glob("checkpoints/model_reachao3_*_steps.zip"), key=lambda p: int(p.stem.split("_")[-2]))
            for k in args.snapshot_k:
                group = [str(p.relative_to(ROOT)) for p in ckpts[-k:]]
                for strategy in args.snapshot_strategies:
                    jobs.append((f"runs/eval/p3_snapshot-s{seed}_{strategy}_k{k}_0.jsonl", group, strategy,
                                 args.episodes, strict, h))
    if "xdist" in args.groups:
        by_dist = xdist_members(frozen["name"])
        if set(by_dist) != {"reachao3", "reachao_dr_plr", "reachao_dr_adr_plr"} or any(len(v) != 5 for v in by_dist.values()):
            raise SystemExit(f"cross-distribution population incomplete: { {d: len(v) for d, v in by_dist.items()} }")
        counter = defaultdict(int)
        for name, subsets_ in xdist_groups(by_dist, n=args.subsets):
            for group in subsets_:
                i = counter[(name, len(group))]
                counter[(name, len(group))] += 1
                for strategy in args.snapshot_strategies:
                    jobs.append((f"runs/eval/p3x_{name}_{strategy}_k{len(group)}_{i}.jsonl", group, strategy,
                                 args.episodes, strict, h))
    out_dir = ROOT / "runs/p3"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for out, group, strategy, episodes, cdef, horizon in jobs:
        part = out.replace(".jsonl", ".part.jsonl")
        common = (f"--scenes {scenes} --collision-def {cdef} --horizon {horizon} --episode-seed-base 20000 "
                  f"--workers {args.workers}")
        lines.append(f"[ -f {out} ] || {{ cd {ROOT} && .venv/bin/python evaluate.py --models {' '.join(group)} "
                     f"--ensemble-strategy {strategy} {common} --episodes {episodes} --out {part} && "
                     f"mv {part.replace('.jsonl', '.run_meta.json')} {out.replace('.jsonl', '.run_meta.json')} && "
                     f"mv {part} {out}; }}")
    commands = out_dir / f"commands_{'-'.join(args.groups)}.txt"
    commands.write_text("\n".join(lines) + "\n")
    (out_dir / "top_members.json").write_text(json.dumps({m: scores.get(m) for m in top}, indent=1))
    episodes = sum(job[3] * len(S.VALIDATION + S.TEST) for job in jobs)
    print(f"{len(pool)} members, {len(jobs)} ensemble evaluations, {episodes:,} episodes -> {commands}")
    print(f"top-{args.top_k} by validation: {[Path(m).parent.parent.name for m in top]}")
    if args.dry_run:
        return
    n = min(args.array, len(lines))
    script = (f"set -e; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; "
              f"module load python/python-3.13.2; cd {ROOT}; "
              f"awk -v t=$SLURM_ARRAY_TASK_ID -v n={n} '(NR - 1) % n == t' {commands.relative_to(ROOT)} | bash -e")
    sub = subprocess.run(["sbatch", "--parsable", f"--array=0-{n - 1}", "--job-name=p3-ens", "--partition=big",
                          f"--cpus-per-task={args.workers}", "--mem=64G", "--time=1-00:00:00",
                          "--output=runs/slurm/p3-%A_%a.out", f"--wrap={script}"],
                         capture_output=True, text=True, check=True)
    print(f"submitted array job {sub.stdout.strip()} with {n} tasks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["sweep", "headline"],
                    choices=["sweep", "headline", "conditions", "snapshot", "xdist"])
    ap.add_argument("--k", nargs="+", type=int, default=[2, 3, 5, 10, 20])
    ap.add_argument("--snapshot-seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--snapshot-k", nargs="+", type=int, default=[5, 10])
    ap.add_argument("--snapshot-strategies", nargs="+", default=["mean", "critic_argmax", "min_variance"])
    ap.add_argument("--subsets", type=int, default=20)
    ap.add_argument("--strategies", nargs="+",
                    default=["min_variance", "critic_argmax", "mean", "precision_weighted", "random"])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--array", type=int, default=12, help="array tasks (each runs every n-th command)")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
