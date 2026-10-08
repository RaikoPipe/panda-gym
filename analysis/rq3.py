"""RQ3: sources of diversity (seed, snapshot, cross-distribution ensembles) and the cost of ensembling.

    python -m analysis.rq3 'runs/eval/rq3_*.jsonl' --collision-def strict --horizon 150

The source of an ensemble follows from its members' run directories:
  snapshot  all members are checkpoints of one run (SEERL; 1/K of the training cost of a seed ensemble)
  xdist     members from different stage-3 distributions of a cross-distribution population (runs/pop/*_xdist/<dist>)
  hybrid    xdist members mixed with others
  seed      members from different runs of one configuration
Outputs (analysis/out/):
  rq3_sources.csv      test/validation success per source x strategy x K (mean over subsets, 2.5-97.5 % range)
  rq3_jerk.csv/.pdf    mean joint jerk against the member-switching rate (switches per step), per ensemble episode,
                       with Spearman rho; single-policy jerk as the reference (H3b)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from analysis.load import condition, load, outdir, run_dir  # noqa: E402


def source(members):
    runs = [run_dir(m) for m in members]
    if len(set(runs)) == 1:
        return "snapshot"
    xdist = [any(p.endswith("_xdist") for p in r.parts) for r in runs]
    if all(xdist):
        dists = {r.parent.name for r in runs}
        return "xdist" if len(dists) > 1 else "seed"
    return "hybrid" if any(xdist) else "seed"


def sources_table(ens):
    rows = []
    for (src, strategy, k, role), g in ens.groupby(["source", "strategy", "K", "role"]):
        if role not in ("validation", "test"):
            continue
        per_subset = g.groupby("members")["success"].mean()
        rows.append(dict(source=src, strategy=strategy, K=k, role=role, mean=per_subset.mean(),
                         lo=per_subset.quantile(0.025), hi=per_subset.quantile(0.975), n_subsets=len(per_subset)))
    return pd.DataFrame(rows)


def jerk_table(ens, single):
    ep = ens[ens["steps"] > 1].copy()
    ep["switch_rate"] = ep["switches"] / (ep["steps"] - 1)
    rows = []
    for (src, strategy), g in ep.groupby(["source", "strategy"]):
        rho, p = spearmanr(g["switch_rate"], g["mean_jerk"]) if g["switch_rate"].nunique() > 1 else (float("nan"),) * 2
        rows.append(dict(source=src, strategy=strategy, episodes=len(g), mean_switch_rate=g["switch_rate"].mean(),
                         mean_jerk=g["mean_jerk"].mean(), spearman_rho=rho, p=p))
    rows.append(dict(source="single policies", strategy="single", episodes=len(single), mean_switch_rate=0.0,
                     mean_jerk=single["mean_jerk"].mean(), spearman_rho=float("nan"), p=float("nan")))
    return pd.DataFrame(rows), ep


def plot_jerk(ep, single_jerk, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for (src, strategy), g in ep.groupby(["source", "strategy"]):
        ax.scatter(g["switch_rate"], g["mean_jerk"], s=4, alpha=0.4, label=f"{src} / {strategy}")
    ax.axhline(single_jerk, color="k", lw=1, label="single policies (mean)")
    ax.set_xlabel("member switches per step")
    ax.set_ylabel("mean joint jerk [rad/s³]")
    ax.legend(fontsize=6, markerscale=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(args):
    df = condition(load(args.records), args.collision_def, args.horizon)
    single = df[df["strategy"] == "single"]
    ens = df[df["strategy"] != "single"].copy()
    if ens.empty:
        raise SystemExit("no ensemble records")
    ens["source"] = ens["members"].map(source)
    out = outdir()
    table = sources_table(ens)
    jerk, ep = jerk_table(ens, single)
    table.to_csv(out / "rq3_sources.csv", index=False)
    jerk.to_csv(out / "rq3_jerk.csv", index=False)
    plot_jerk(ep, single["mean_jerk"].mean() if len(single) else float("nan"), out / "rq3_jerk.pdf")
    print(table.to_string(index=False))
    print(jerk.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    ap.add_argument("--collision-def")
    ap.add_argument("--horizon", type=int)
    main(ap.parse_args())
