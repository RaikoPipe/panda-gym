"""RQ1: zero-shot transfer and seed specialization of single policies.

    python -m analysis.rq1 'runs/eval/pop_*.jsonl' --collision-def strict --horizon 150

Outputs (analysis/out/):
  rq1_success_matrix.csv/.pdf  success per run (seed) x scene
  rq1_spearman.csv             Spearman rho between in-distribution and OOD success across seeds (H1b)
  rq1_aggregates.csv           mean and IQM over seeds per scene role, stratified-bootstrap 95 % CIs (rliable)
  rq1_heterogeneity.csv        per OOD scene: chi-square test that all seeds share one success rate (H1b, no model)
  rq1_mixed.csv                Bayesian mixed logistic model success ~ scene + (1|seed) + (1|seed:scene) on OOD
                               episodes: random-effect SDs (H1b: seed x scene interaction), variational fit with
                               95 % intervals and the posterior mode as a cross-check, each with a converged flag
"""
import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency, spearmanr  # noqa: E402
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM  # noqa: E402
from statsmodels.tools.sm_exceptions import ConvergenceWarning  # noqa: E402

from analysis import stats  # noqa: E402
from analysis.load import condition, load, outdir  # noqa: E402
from panda_gym import scenes as S  # noqa: E402


def success_matrix(single):
    return single.pivot_table(index="run", columns="scene", values="success", aggfunc="mean")


def spearman_table(matrix):
    indist = [s for s in S.IN_DISTRIBUTION if s in matrix]
    if not indist or len(matrix) < 3:
        return pd.DataFrame()
    x = matrix[indist].mean(axis=1)
    rows = []
    for role, members in (("validation", S.VALIDATION), ("test", S.TEST), ("ood", S.VALIDATION + S.TEST)):
        cols = [s for s in members if s in matrix]
        if cols:
            rho, p = spearmanr(x, matrix[cols].mean(axis=1))
            rows.append(dict(ood=role, rho=rho, p=p, n_seeds=len(matrix), scenes=" ".join(cols)))
    return pd.DataFrame(rows)


def aggregate_table(matrix):
    scores = {}
    for role, members in (("in_distribution", S.IN_DISTRIBUTION), ("validation", S.VALIDATION), ("test", S.TEST)):
        cols = [s for s in members if s in matrix]
        if cols:
            scores[role] = matrix[cols].to_numpy()
    return pd.DataFrame([{"role": k, "n_seeds": len(v), "n_scenes": v.shape[1], **agg}
                         for (k, v), agg in zip(scores.items(), stats.aggregates(scores).values())])


def mixed_effects(single, maxiter=5000):
    """Variational fit (the reported intervals) and the posterior mode (MAP) as a cross-check; both record whether
    the optimizer converged."""
    ood = single[single["role"].isin(["validation", "test"])].copy()
    if ood["run"].nunique() < 2 or ood["scene"].nunique() < 2:
        return pd.DataFrame()
    ood["seed_id"] = ood["run"].astype("category").cat.codes
    ood["y"] = ood["success"].astype(float)
    model = BinomialBayesMixedGLM.from_formula(
        "y ~ C(scene)", {"seed": "0 + C(seed_id)", "seed_x_scene": "0 + C(seed_id):C(scene)"}, ood)
    rows = []
    for method in ("vb", "map"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opts = {"maxiter": maxiter, "gtol": 1e-6}
            fit = (model.fit_vb(scale_fe=True, minim_opts=opts) if method == "vb"
                   else model.fit_map(minim_opts=opts, scale_fe=True))
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        for name, mean, sd in zip(model.vcp_names, fit.vcp_mean, fit.vcp_sd):  # log standard deviations
            rows.append(dict(method=method, converged=converged, effect=name, sd=np.exp(mean),
                             sd_lo=np.exp(mean - stats.Z * sd), sd_hi=np.exp(mean + stats.Z * sd),
                             log_sd_mean=mean, log_sd_posterior_sd=sd))
    return pd.DataFrame(rows)


def heterogeneity(single):
    """Per OOD scene: chi-square test that all seeds have the same success rate (distribution-free check of H1b)."""
    rows = []
    for scene, g in single[single["role"].isin(["validation", "test"])].groupby("scene"):
        table = pd.crosstab(g["run"], g["success"])
        if table.shape[1] == 2 and table.shape[0] > 1:
            chi2, p, dof, _ = chi2_contingency(table)
            rate = g.groupby("run")["success"].mean()
            rows.append(dict(scene=scene, chi2=chi2, dof=dof, p=p, min=rate.min(), max=rate.max()))
    return pd.DataFrame(rows)


def plot_matrix(matrix, path):
    fig, ax = plt.subplots(figsize=(1 + 0.6 * matrix.shape[1], 1 + 0.3 * matrix.shape[0]))
    im = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=60, ha="right")
    ax.set_yticks(range(matrix.shape[0]), [r.split("/")[-1] for r in matrix.index])
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(args):
    df = condition(load(args.records), args.collision_def, args.horizon)
    single = df[df["strategy"] == "single"]
    out = outdir()
    matrix = success_matrix(single)
    matrix.to_csv(out / "rq1_success_matrix.csv")
    plot_matrix(matrix, out / "rq1_success_matrix.pdf")
    for name, table in (("spearman", spearman_table(matrix)), ("aggregates", aggregate_table(matrix)),
                        ("heterogeneity", heterogeneity(single)), ("mixed", mixed_effects(single))):
        table.to_csv(out / f"rq1_{name}.csv", index=False)
        print(f"--- {name}\n{table.to_string(index=False) if len(table) else '(not enough data)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", help="per-episode JSONL files or globs")
    ap.add_argument("--collision-def")
    ap.add_argument("--horizon", type=int)
    main(ap.parse_args())
