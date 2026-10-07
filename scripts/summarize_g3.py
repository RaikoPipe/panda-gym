"""Summarize the G3 sweep: cost and quality per algorithm x UTD x batch x seed.

    python scripts/summarize_g3.py [--target 0.9]

Per run (runs/sweep/g3_*): wall-clock time, env steps/s, wall-clock time until the in-distribution evaluation
during training first reaches --target in the last curriculum stage, in-distribution success of the final model
(200 shared episodes on reachao3), validation mean of the final model and of the validation-best of the last 5
checkpoints. Writes runs/sweep/g3_summary.csv.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def success(path, model=None):
    if not path.exists():
        return None
    recs = [json.loads(line) for line in open(path)]
    if model is not None:
        recs = [r for r in recs if r["model"] == model]
    scenes = sorted({r["scene"] for r in recs})
    return float(np.mean([np.mean([r["outcome"] == "success" for r in recs if r["scene"] == s]) for s in scenes]))


def best_of(path):
    if not path.exists():
        return None, None
    models = sorted({json.loads(line)["model"] for line in open(path)})
    scores = {m: success(path, m) for m in models}
    best = max(scores, key=scores.get)
    return best, scores[best]


def main(args):
    rows = []
    for run in sorted(Path("runs/sweep").glob("g3_*")):
        meta_path = run / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        a = meta["args"]
        evals = [json.loads(line) for line in open(run / "evals.jsonl")] if (run / "evals.jsonl").exists() else []
        last_stage = evals[-1]["stage"] if evals else None
        hit = next((e for e in evals if e["stage"] == last_stage and (e["success_rate"] or 0) >= args.target), None)
        wall = evals[-1]["wall_s"] if evals else None
        steps = meta.get("num_timesteps")
        best, best_val = best_of(Path("runs/eval") / f"{run.name}_last5_val.jsonl")
        rows.append({
            "run": run.name, "algo": a["algo"], "utd": a["effective_utd"], "batch": a["batch_size"],
            "seed": a["seed"], "done": (run / "done").exists(), "env_steps": steps, "wall_h": wall and wall / 3600,
            "env_steps_per_s": steps / wall if steps and wall else None,
            "time_to_target_h": hit and hit["wall_s"] / 3600, "steps_to_target": hit and hit["timesteps"],
            "indist_success": success(Path("runs/eval") / f"{run.name}_indist.jsonl"),
            "val_mean_final": success(Path("runs/eval") / f"{run.name}_final_val.jsonl"),
            "val_mean_last5_best": best_val, "last5_best_model": best,
        })
    if not rows:
        raise SystemExit("no G3 runs in runs/sweep")
    fmt = lambda x, f: "-" if x is None else format(x, f)
    print(f"{'run':32s} {'h':>5s} {'st/s':>5s} {'t->tgt h':>8s} {'indist':>6s} {'val fin':>7s} {'val l5':>6s}")
    for r in rows:
        print(f"{r['run']:32s} {fmt(r['wall_h'], '5.1f')} {fmt(r['env_steps_per_s'], '5.0f')} "
              f"{fmt(r['time_to_target_h'], '8.2f')} {fmt(r['indist_success'], '6.0%')} "
              f"{fmt(r['val_mean_final'], '7.0%')} {fmt(r['val_mean_last5_best'], '6.0%')}")
    out = Path("runs/sweep/g3_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, default=0.9, help="in-distribution success target (design v3)")
    main(ap.parse_args())
