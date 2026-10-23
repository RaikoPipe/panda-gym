"""RQ3, cross-distribution population: does mixing training distributions add diversity beyond seeds?

    python -m analysis.rq3_xdist

All members are fine-tuned from one stage-2 checkpoint with the same budget (configs/paper_frozen.json, xdist), so
pure and mixed ensembles of equal K differ only in the training distributions of their members.
Inputs: runs/eval/p2_<name>_xdist-<dist>_s*_strict_h150.jsonl (single policies, validation-selected checkpoints) and
runs/eval/p3x_<group>_<strategy>_k<K>_<i>.jsonl (ensembles; groups pure-<dist> and mixed, see
scripts/phase3_ensembles.py). Outputs (analysis/out/):
  rq3x_single.csv     validation/test success per training distribution (single policies)
  rq3x_ensembles.csv  validation/test success per group x K x strategy: mean over subsets, 2.5-97.5 % range
  rq3x_matched.csv    mixed minus pure-reachao3 at K = 3 (all 10 vs 20 subsets) and mixed K = 5 minus each pure K = 5,
                      on test episodes, with a bootstrap CI over subsets
  rq3x_scenes.csv     per-scene test success of the K = 5 / K = 15 ensembles (mean strategy) and the single policies
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.load import outdir
from panda_gym import scenes as S

NAME = json.loads(Path("configs/paper_frozen.json").read_text())["name"]


def success_by_scene(path):
    succ = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        succ[r["scene"]].append(r["outcome"] == "success")
    return {s: float(np.mean(v)) for s, v in succ.items()}


def role_means(by_scene):
    return {role: float(np.mean([by_scene[s] for s in scenes if s in by_scene]))
            for role, scenes in (("validation", S.VALIDATION), ("test", S.TEST))}


def bootstrap_diff(a, b, reps=5000, seed=0):
    """Difference of means of two sets of subset scores, with a percentile bootstrap CI (resampling subsets)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    boots = [rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean() for _ in range(reps)]
    return a.mean() - b.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    out = outdir()
    singles = []
    for f in glob.glob(f"runs/eval/p2_{NAME}_xdist-*_s*_strict_h150.jsonl"):
        dist, seed = re.search(r"xdist-(\w+?)_s(\d+)_strict", f).groups()
        singles.append(dict(distribution=dist, seed=int(seed), **role_means(success_by_scene(f)),
                            **{f"scene:{k}": v for k, v in success_by_scene(f).items()}))
    singles = pd.DataFrame(singles)
    single_table = singles.groupby("distribution")[["validation", "test"]].agg(["mean", "min", "max"])
    single_table.to_csv(out / "rq3x_single.csv")

    rows = []
    for f in glob.glob("runs/eval/p3x_*.jsonl"):
        if f.endswith(".part.jsonl"):
            continue
        group, strategy, k, i = re.search(r"p3x_(.+?)_(mean|critic_argmax|min_variance)_k(\d+)_(\d+)\.jsonl", f).groups()
        by_scene = success_by_scene(f)
        rows.append(dict(group=group, strategy=strategy, K=int(k), subset=int(i), **role_means(by_scene),
                         **{f"scene:{s}": v for s, v in by_scene.items()}))
    ens = pd.DataFrame(rows)
    table = ens.groupby(["group", "K", "strategy"]).agg(
        n_subsets=("test", "size"), validation=("validation", "mean"), test=("test", "mean"),
        test_lo=("test", lambda x: x.quantile(0.025)), test_hi=("test", lambda x: x.quantile(0.975))).reset_index()
    table.to_csv(out / "rq3x_ensembles.csv", index=False)

    matched = []
    for strategy in sorted(ens["strategy"].unique()):
        sel = lambda g, k: ens[(ens.group == g) & (ens.K == k) & (ens.strategy == strategy)]["test"].to_numpy()
        comparisons = [("mixed K=3 vs pure-reachao3 K=3", sel("mixed", 3), sel("pure-reachao3", 3))]
        comparisons += [(f"mixed K=5 vs pure-{d} K=5", sel("mixed", 5), sel(f"pure-{d}", 5))
                        for d in ("reachao3", "reachao_dr_plr", "reachao_dr_adr_plr")]
        for label, a, b in comparisons:
            if a.size and b.size:
                diff, lo, hi = bootstrap_diff(a, b)
                matched.append(dict(strategy=strategy, comparison=label, mixed=a.mean(), pure=b.mean(), diff=diff,
                                    lo=lo, hi=hi, n_mixed=a.size, n_pure=b.size))
    matched = pd.DataFrame(matched)
    matched.to_csv(out / "rq3x_matched.csv", index=False)

    scene_cols = [f"scene:{s}" for s in S.TEST]
    scenes = {f"single {d}": singles[singles.distribution == d][scene_cols].mean() for d in sorted(singles.distribution.unique())}
    for g, k in (("pure-reachao3", 5), ("pure-reachao_dr_plr", 5), ("pure-reachao_dr_adr_plr", 5), ("mixed", 5), ("mixed", 15)):
        part = ens[(ens.group == g) & (ens.K == k) & (ens.strategy == "mean")]
        scenes[f"{g} K={k} (mean)"] = part[scene_cols].mean()
    scene_table = pd.DataFrame(scenes).T
    scene_table.columns = [c.removeprefix("scene:") for c in scene_table.columns]
    scene_table.to_csv(out / "rq3x_scenes.csv")

    pd.set_option("display.width", 200)
    print("--- single policies per training distribution (validation-selected checkpoints, 500 episodes/scene)")
    print(single_table.round(3).to_string())
    print("--- ensembles (200 episodes/scene; mean over subsets)")
    print(table.round(3).to_string(index=False))
    print("--- matched comparisons on test (difference of subset means, bootstrap 95 % CI over subsets)")
    print(matched.round(3).to_string(index=False))
    print("--- test success per scene")
    print(scene_table.round(2).to_string())


if __name__ == "__main__":
    main()
