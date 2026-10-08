"""Per-episode evaluation records (evaluate.py --out) as one DataFrame, with shared-episode checks.

Columns added to the raw records:
  members   tuple of model paths (one for single policies)
  K         ensemble size (1 for single policies)
  run       run directory of a single policy (checkpoints count as their run), None for ensembles
  seed      training seed of `run` (run_meta.json, else an "s<seed>" directory name), None if unknown
  method    hashable key of a policy: (strategy, members, ensemble options)
  role      scene role from panda_gym/scenes.py: train, validation, test or other
  success   outcome == "success"
"""
import glob
import json
import re
from pathlib import Path

import pandas as pd

from panda_gym import scenes as S

OUT = Path(__file__).parent / "out"


def run_dir(model):
    path = Path(model).parent
    return path.parent if path.name == "checkpoints" else path


def seed_of(run):
    meta = run / "run_meta.json"
    if meta.exists():
        return json.loads(meta.read_text())["args"].get("seed")
    m = re.search(r"s(\d+)$", run.name)
    return int(m.group(1)) if m else None


def role(scene):
    for name, members in (("train", S.TRAIN), ("validation", S.VALIDATION), ("test", S.TEST)):
        if scene in members:
            return name
    return "other"


def load(patterns, check=True):
    files = sorted({f for pat in ([patterns] if isinstance(patterns, str) else patterns) for f in glob.glob(pat)})
    files = [f for f in files if not f.endswith(".part.jsonl")]
    if not files:
        raise FileNotFoundError(f"no records match {patterns}")
    rows = [json.loads(line) for f in files for line in open(f)]
    df = pd.DataFrame(rows)
    df["members"] = df["model"].map(lambda m: tuple(m) if isinstance(m, list) else (m,))
    df["K"] = df["members"].map(len)
    single = df["strategy"] == "single"
    runs = df.loc[single, "members"].map(lambda m: run_dir(m[0]))
    df["run"] = None
    df.loc[single, "run"] = runs.map(str)
    seeds = {r: seed_of(Path(r)) for r in df.loc[single, "run"].unique()}
    df["seed"] = df["run"].map(seeds)
    options = df["ensemble_options"].map(lambda o: tuple(o) if isinstance(o, list) else o) \
        if "ensemble_options" in df else None
    df["method"] = list(zip(df["strategy"], df["members"], options if options is not None else [None] * len(df)))
    df["role"] = df["scene"].map(role)
    df["success"] = df["outcome"] == "success"
    if check:
        check_shared_episodes(df)
    return df


def check_shared_episodes(df):
    """Every method must have seen the same start state for each (scene, episode seed)."""
    hashes = df.groupby(["scene", "episode_seed"])["hash"].nunique()
    bad = hashes[hashes > 1]
    if len(bad):
        raise ValueError(f"{len(bad)} episodes were not shared between methods, e.g. {list(bad.index[:3])}")


def condition(df, collision_def=None, horizon=None):
    """Restrict to one evaluation condition (collision definition, horizon)."""
    if collision_def is not None:
        df = df[df["collision_def"] == collision_def]
    if horizon is not None:
        df = df[df["horizon"] == horizon]
    return df


def outdir():
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT
