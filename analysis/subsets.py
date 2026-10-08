"""Random ensemble subsets for RQ2/RQ3 (50 per K, numpy.random.default_rng(0)) and the evaluate.py commands for them.

    python -m analysis.subsets --members 'runs/pop/crossq_utd1/s*/final_reachao3.zip' --k 1 2 3 5 10 20 \
        --strategies min_variance mean critic_argmax --out analysis/out/subsets.json \
        --eval-args "--scenes tunnel library2 --collision-def strict --horizon 150 --episodes 500"

Writes the subsets (JSON) and prints one evaluate.py command per strategy x subset (K = 1 subsets are the single
policies, evaluated without --ensemble-strategy). With K = len(members) there is a single subset.
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


def subsets(members, ks, n=50, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        if k > len(members):
            continue
        if math.comb(len(members), k) <= n:  # few combinations: take all of them
            from itertools import combinations
            out[k] = [list(c) for c in combinations(members, k)]
        else:
            seen = set()
            while len(seen) < n:
                seen.add(tuple(sorted(rng.choice(len(members), size=k, replace=False).tolist())))
            out[k] = [[members[i] for i in idx] for idx in sorted(seen)]
    return out


def main(args):
    members = sorted({m for pat in args.members for m in glob.glob(pat)})
    if not members:
        raise SystemExit("no members")
    subs = subsets(members, args.k, args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({str(k): v for k, v in subs.items()}, indent=1))
    for k, groups in subs.items():
        for i, group in enumerate(groups):
            models = " ".join(group)
            if k == 1:
                print(f".venv/bin/python evaluate.py {models} {args.eval_args} --out runs/eval/{args.tag}_single_{i}.jsonl")
                continue
            for strategy in args.strategies:
                print(f".venv/bin/python evaluate.py --models {models} --ensemble-strategy {strategy} {args.eval_args} "
                      f"--out runs/eval/{args.tag}_{strategy}_k{k}_{i}.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True, help="model files or globs")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 2, 3, 5, 10, 20])
    ap.add_argument("--n", type=int, default=50, help="random subsets per K")
    ap.add_argument("--strategies", nargs="+", default=["min_variance"])
    ap.add_argument("--eval-args", default="")
    ap.add_argument("--tag", default="rq2")
    ap.add_argument("--out", default="analysis/out/subsets.json")
    main(ap.parse_args())
