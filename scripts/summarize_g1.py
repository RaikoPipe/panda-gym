"""Summarize the G1 ablation: success per agent x collision definition x horizon and scene, with Wilson 95 % CIs.

    python scripts/summarize_g1.py [runs/eval/g1_*.jsonl ...]

Prints the table, writes runs/eval/g1_summary.csv, and checks that all runs saw the same episodes (hashes).
Besides the mean over all evaluated scenes it reports the validation and test means (panda_gym/scenes.py): G1 decides
the collision definition and horizon, so the read-out must rest on the validation mean, not on test scenes.
"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from panda_gym.scenes import TEST, VALIDATION

Z = 1.959964


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + Z ** 2 / (2 * n)) / (1 + Z ** 2 / n)
    half = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / (1 + Z ** 2 / n)
    return centre - half, centre + half


def load(files):
    records = []
    for f in files:
        with open(f) as fh:
            records += [json.loads(line) for line in fh]
    return records


def check_shared_episodes(records):
    hashes = defaultdict(set)
    for r in records:
        hashes[(r["scene"], r["episode_seed"])].add(r["hash"])
    bad = {k: v for k, v in hashes.items() if len(v) > 1}
    if bad:
        print(f"WARNING: {len(bad)} episodes differ between runs (not shared): {sorted(bad)[:5]} ...")
    return not bad


def main(files):
    files = [f for f in files if not f.endswith(".part.jsonl")]
    records = load(files)
    if not records:
        raise SystemExit("no G1 records")
    check_shared_episodes(records)
    scenes = list(dict.fromkeys(r["scene"] for r in records))
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["collision_def"], r["horizon"])].append(r)
    rows = []
    for (model, cdef, horizon), recs in sorted(groups.items()):
        row = {"agent": model, "collision_def": cdef, "horizon": horizon}
        k_all = n_all = 0
        for scene in scenes:
            outcomes = [r["outcome"] for r in recs if r["scene"] == scene]
            k, n = outcomes.count("success"), len(outcomes)
            lo, hi = wilson(k, n)
            row.update({f"{scene}": k / n if n else float("nan"), f"{scene}_lo": lo, f"{scene}_hi": hi,
                        f"{scene}_collision": outcomes.count("collision") / n if n else float("nan")})
            k_all, n_all = k_all + k, n_all + n
        lo, hi = wilson(k_all, n_all)
        row.update(mean=k_all / n_all, mean_lo=lo, mean_hi=hi, episodes=n_all)
        for role, members in (("val", VALIDATION), ("test", TEST)):
            outcomes = [r["outcome"] for r in recs if r["scene"] in members]
            k, n = outcomes.count("success"), len(outcomes)
            lo, hi = wilson(k, n)
            row.update({f"{role}_mean": k / n if n else float("nan"), f"{role}_lo": lo, f"{role}_hi": hi})
        rows.append(row)

    width = max(len(Path(r["agent"]).parent.name or r["agent"]) for r in rows)
    print(f"{'agent':{width}s} {'definition':15s} {'h':>4s} " + " ".join(f"{s[:11]:>11s}" for s in scenes)
          + f" {'mean [95% CI]':>20s} {'VAL mean [95% CI]':>22s} {'test':>6s}")
    for r in rows:
        print(f"{Path(r['agent']).parent.name or r['agent']:{width}s} {r['collision_def']:15s} {r['horizon']:4d} "
              + " ".join(f"{r[s]:11.0%}" for s in scenes)
              + f" {r['mean']:6.1%} [{r['mean_lo']:.1%}, {r['mean_hi']:.1%}]"
              + f" {r['val_mean']:8.1%} [{r['val_lo']:.1%}, {r['val_hi']:.1%}] {r['test_mean']:6.1%}")
    print("Decide on the VAL mean (validation scenes); the test scenes are listed for information only.")
    out = Path("runs/eval/g1_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("runs/eval/g1_*.jsonl")))
