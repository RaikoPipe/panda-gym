"""T11 analysis on synthetic records with known structure."""
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

import analysis.load as L
from analysis import rq1, rq2, rq3, stats
from analysis.subsets import subsets

SCENES = {"reachao3": 0.9, "tunnel": 0.3, "workshop2": 0.2, "library2": 0.6, "narrow_tunnel": 0.4}


def record(model, scene, i, success, strategy="single", members=None, switches=None):
    r = dict(model=model, strategy=strategy, scene=scene, horizon=150, collision_def="strict", deterministic=True,
             ensemble_options=None if strategy == "single" else ["pre_tanh", "sum_log_std"], episode_seed=i,
             hash=f"{scene}-{i}", outcome="success" if success else "collision", steps=50, path_length=0.5,
             mean_ee_speed=0.3, mean_jerk=10.0 + (switches or 0), max_jerk=50.0, latency_ms_mean=1.0,
             latency_ms_p99=2.0, collision_link=None)
    if switches is not None:
        r.update(members=[[0, 50]], switches=switches)
    return r


@pytest.fixture()
def records(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "OUT", tmp_path / "out")
    rng = np.random.default_rng(0)
    members = [f"runs/pop/x/s{s}/final_reachao3.zip" for s in range(6)]
    skill = {m: {sc: np.clip(p + rng.normal(0, 0.15), 0, 1) for sc, p in SCENES.items()} for m in members}
    single = {m: {sc: rng.random(40) < skill[m][sc] for sc in SCENES} for m in members}
    rows = [record(m, sc, i, single[m][sc][i]) for m in members for sc in SCENES for i in range(40)]
    for k, groups in subsets(members, [2, 6], n=3).items():
        for group in groups:
            for sc in SCENES:
                for i in range(40):
                    ok = np.mean([single[m][sc][i] for m in group]) >= 0.5
                    rows.append(record(group, sc, i, ok, "min_variance", group, switches=int(rng.integers(0, 20))))
    snapshot = [f"runs/pop/x/s0/checkpoints/model_reachao3_{n}_steps.zip" for n in (1, 2, 3)]
    rows += [record(snapshot, sc, i, single[members[0]][sc][i], "min_variance", snapshot, switches=3)
             for sc in SCENES for i in range(40)]
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path), members, skill


def test_load_derives_runs_seeds_and_roles(records):
    path, members, _ = records
    df = L.load(path)
    single = df[df["strategy"] == "single"]
    assert sorted(single["seed"].unique()) == list(range(6))
    assert set(df["role"]) == {"train", "validation", "test"}
    assert df[df["strategy"] != "single"]["K"].max() == 6


def test_unshared_episodes_are_rejected(records, tmp_path):
    path, _, _ = records
    bad = tmp_path / "bad.jsonl"
    rows = [json.loads(line) for line in open(path)]
    rows[0]["hash"] = "different"
    bad.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        L.load(str(bad))


def test_rq_scripts_write_their_outputs(records, tmp_path):
    path, members, skill = records
    args = Namespace(records=[path], collision_def="strict", horizon=150, tag="")
    for module in (rq1, rq2, rq3):
        module.main(args)
    out = tmp_path / "out"
    for name in ("rq1_success_matrix.csv", "rq1_spearman.csv", "rq1_aggregates.csv", "rq1_mixed.csv",
                 "rq2_by_k.csv", "rq2_paired.csv", "rq2_mcnemar.csv", "rq3_sources.csv", "rq3_jerk.csv"):
        assert (out / name).exists(), name
    mixed = pd.read_csv(out / "rq1_mixed.csv")
    assert set(mixed["effect"]) == {"seed", "seed_x_scene"}
    by_k = pd.read_csv(out / "rq2_by_k.csv")
    assert {"min_variance", "oracle (upper bound)", "single"} <= set(by_k["strategy"])
    assert (by_k[by_k["strategy"] == "oracle (upper bound)"]["mean"] <= 1).all()
    # single-subset groups (seed ensemble K = 6, snapshot ensemble K = 3): one test per test scene each
    assert len(pd.read_csv(out / "rq2_mcnemar.csv")) == 4
    assert set(pd.read_csv(out / "rq3_sources.csv")["source"]) == {"seed", "snapshot"}


def test_baseline_is_selected_on_validation_only(records):
    path, members, skill = records
    single = L.load(path).query("strategy == 'single'")
    best = rq2.select_single(single)
    val = single[single["role"] == "validation"].groupby("members")["success"].mean()
    assert (best,) == val.idxmax()


def test_paired_bootstrap_and_mcnemar():
    mean, lo, hi = stats.paired_bootstrap(np.r_[np.ones(30), np.zeros(70)], np.repeat(["a", "b"], 50))
    assert lo < mean == 0.3 < hi
    p, b_only, a_only = stats.mcnemar(np.r_[np.ones(20), np.zeros(20)], np.zeros(40))
    assert (b_only, a_only) == (0, 20) and p < 1e-4


def test_subsets_are_reproducible():
    members = [f"m{i}" for i in range(20)]
    a, b = subsets(members, [3, 20], n=50), subsets(members, [3, 20], n=50)
    assert a == b and len(a[3]) == 50 and len(a[20]) == 1


def test_mcnemar_counts_are_labelled_by_method(records, tmp_path):
    """only_ensemble counts episodes the ensemble solved and the baseline did not."""
    path, _, _ = records
    rq2.main(Namespace(records=[path], collision_def="strict", horizon=150, tag=""))
    m = pd.read_csv(tmp_path / "out" / "rq2_mcnemar.csv")
    df = L.load(path)
    single = df[df["strategy"] == "single"]
    base = rq2.episode_success(single[(single["members"] == (rq2.select_single(single),)) & (single["role"] == "test")])
    for _, row in m.iterrows():
        ens = df[(df["strategy"] == row["strategy"]) & (df["K"] == row["K"]) & (df["scene"] == row["scene"])]
        e = rq2.episode_success(ens)
        b = base.loc[e.index]
        assert row["only_ensemble"] == int(((e > 0.5) & (b < 0.5)).sum())
        assert row["only_baseline"] == int(((e < 0.5) & (b > 0.5)).sum())
