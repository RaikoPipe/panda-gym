"""Statistics of the research design (section 5): rliable aggregates, paired episode comparisons, Wilson intervals."""
import numpy as np
from rliable import library as rly
from rliable import metrics
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

Z = 1.959964
REPS = 2000


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + Z ** 2 / (2 * n)) / (1 + Z ** 2 / n)
    half = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / (1 + Z ** 2 / n)
    return centre - half, centre + half


def aggregates(score_dict, reps=REPS):
    """{name: [runs x tasks] success matrix} -> {name: dict(mean, iqm and their stratified-bootstrap 95 % CIs)}."""
    fn = lambda x: np.array([metrics.aggregate_mean(x), metrics.aggregate_iqm(x)])  # noqa: E731
    point, ci = rly.get_interval_estimates(score_dict, fn, reps=reps)
    return {k: dict(mean=point[k][0], mean_lo=ci[k][0, 0], mean_hi=ci[k][1, 0],
                    iqm=point[k][1], iqm_lo=ci[k][0, 1], iqm_hi=ci[k][1, 1]) for k in score_dict}


def probability_of_improvement(x, y, reps=REPS):
    """P(X > Y) for [runs x tasks] matrices X, Y, with a stratified-bootstrap 95 % CI."""
    point, ci = rly.get_interval_estimates({"p": (x, y)}, metrics.probability_of_improvement, reps=reps)
    return float(point["p"]), float(ci["p"][0]), float(ci["p"][1])


def paired_bootstrap(diff, strata, reps=REPS, seed=0):
    """Mean of per-episode differences with a 95 % CI, resampling episodes within strata (scenes)."""
    diff, strata = np.asarray(diff, dtype=float), np.asarray(strata)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    boots = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([rng.choice(g, size=len(g)) for g in groups])
        boots[r] = diff[idx].mean()
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def mcnemar(a, b):
    """Exact McNemar test on paired binary outcomes a, b (same episodes). Returns (p value, b-only, a-only)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    table = [[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]]
    return float(_mcnemar(table, exact=True).pvalue), int(table[1][0]), int(table[0][1])
