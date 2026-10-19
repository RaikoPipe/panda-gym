"""RQ2: does an inference-time ensemble beat the validation-selected single policy?

    python -m analysis.rq2 'runs/eval/rq2_*.jsonl' --collision-def strict --horizon 150

Inputs: single-policy records of every member and ensemble records (analysis/subsets.py), one condition.
Outputs (analysis/out/):
  rq2_by_k.csv/.pdf  test (and validation) success per strategy and K: mean over subsets and the subsets' 2.5-97.5 %
                     range; the oracle (an episode counts as solved if any member solved it) as a labelled upper bound
  rq2_paired.csv     per strategy and K: paired difference to the validation-selected single policy on test
                     episodes (mean over subsets per episode), stratified-bootstrap 95 % CI
  rq2_mcnemar.csv    exact McNemar test per test scene for every (strategy, K) with a single subset (e.g. K = all)
The single policy is selected on validation scenes only (highest validation mean); test scenes never select.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import stats  # noqa: E402
from analysis.load import condition, load, outdir  # noqa: E402

KEY = ["scene", "episode_seed"]


def select_single(single):
    """Validation-selected single policy: the member path with the highest mean validation success."""
    val = single[single["role"] == "validation"]
    if val.empty:
        raise SystemExit("no validation records of single policies: cannot select the baseline")
    per_scene = val.groupby(["members", "scene"])["success"].mean()
    best = per_scene.groupby(level=0).mean().idxmax()
    return best[0]


def episode_success(df):
    """Success per (scene, episode_seed) as a Series."""
    return df.set_index(KEY)["success"].astype(float)


def oracle(single, members, role):
    per_member = [episode_success(single[(single["members"] == (m,)) & (single["role"] == role)]) for m in members]
    if any(s.empty for s in per_member):
        return None
    return pd.concat(per_member, axis=1).max(axis=1)


def by_k(df, single):
    ens = df[df["strategy"] != "single"]
    rows = []
    groups = list(ens.groupby(["strategy", "K"]))
    singles_as_k1 = [("single", 1, single)]
    for (strategy, k), g in [((s, k), g) for (s, k), g in groups] + [((s, k), g) for s, k, g in singles_as_k1]:
        for role in ("validation", "test"):
            sub = g[g["role"] == role]
            if sub.empty:
                continue
            per_subset = sub.groupby("members")["success"].mean()
            rows.append(dict(strategy=strategy, K=k, role=role, mean=per_subset.mean(), n_subsets=len(per_subset),
                             lo=per_subset.quantile(0.025), hi=per_subset.quantile(0.975)))
    for k, g in ens.groupby("K"):  # oracle over the same subsets as the ensembles
        for role in ("validation", "test"):
            vals = [o.mean() for m in g["members"].unique() if (o := oracle(single, m, role)) is not None]
            if vals:
                rows.append(dict(strategy="oracle (upper bound)", K=k, role=role, mean=np.mean(vals),
                                 n_subsets=len(vals), lo=np.quantile(vals, 0.025), hi=np.quantile(vals, 0.975)))
    return pd.DataFrame(rows).sort_values(["role", "strategy", "K"])


def paired(df, single, baseline):
    base = episode_success(single[(single["members"] == (baseline,)) & (single["role"] == "test")])
    rows, mc = [], []
    for (strategy, k), g in df[df["strategy"] != "single"].groupby(["strategy", "K"]):
        test = g[g["role"] == "test"]
        if test.empty:
            continue
        ens = test.groupby(KEY)["success"].mean()  # fraction of subsets that solved each episode
        both = pd.concat([ens.rename("ens"), base.rename("base")], axis=1, join="inner")
        if both.empty:
            continue
        diff = both["ens"] - both["base"]
        mean, lo, hi = stats.paired_bootstrap(diff.to_numpy(), both.index.get_level_values("scene"))
        rows.append(dict(strategy=strategy, K=k, ensemble=both["ens"].mean(), baseline=both["base"].mean(),
                         diff=mean, lo=lo, hi=hi, episodes=len(both), n_subsets=test["members"].nunique()))
        if test["members"].nunique() == 1:
            for scene, s in both.groupby(level="scene"):
                p, only_ens, only_base = stats.mcnemar(s["base"] > 0.5, s["ens"] > 0.5)  # (p, b-only, a-only)
                mc.append(dict(strategy=strategy, K=k, scene=scene, p=p, only_baseline=only_base,
                               only_ensemble=only_ens, episodes=len(s)))
    return pd.DataFrame(rows), pd.DataFrame(mc)


def plot(table, baseline_test, path):
    test = table[table["role"] == "test"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for strategy, g in test.groupby("strategy"):
        if strategy == "single":
            continue
        style = "--" if strategy.startswith("oracle") else "-"
        ax.plot(g["K"], g["mean"], style, marker="o", label=strategy)
        ax.fill_between(g["K"], g["lo"], g["hi"], alpha=0.15)
    ax.axhline(baseline_test, color="k", lw=1, label="validation-selected single")
    ax.set_xscale("log")
    ax.set_xlabel("ensemble size K")
    ax.set_ylabel("test success rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(args):
    df = condition(load(args.records), args.collision_def, args.horizon)
    single = df[df["strategy"] == "single"]
    baseline = select_single(single)
    out = outdir()
    table = by_k(df, single)
    pairs, mc = paired(df, single, baseline)
    table.to_csv(out / f"rq2{args.tag}_by_k.csv", index=False)
    pairs.to_csv(out / f"rq2{args.tag}_paired.csv", index=False)
    mc.to_csv(out / f"rq2{args.tag}_mcnemar.csv", index=False)
    base_test = single[(single["members"] == (baseline,)) & (single["role"] == "test")]["success"].mean()
    plot(table, base_test, out / f"rq2{args.tag}_by_k.pdf")
    print(f"validation-selected single policy: {baseline} (test success {base_test:.1%})")
    print(table.to_string(index=False))
    print(pairs.to_string(index=False) if len(pairs) else "(no ensemble records)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    ap.add_argument("--collision-def")
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _sweep")
    main(ap.parse_args())
