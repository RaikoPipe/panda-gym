"""Scene roles and shared-episode indexing for the paper experiments.

SCENARIO_ORDER fixes each scene's index in the episode seeds (evaluate.py: seed = base + 1000 * index + episode), so
it is append-only: never reorder or remove entries, or old evaluations stop being comparable.
"""

SCENARIO_ORDER = (
    "reachao1", "reachao2", "reachao3", "wangexp-3", "reachao_rand", "reachao_dr", "reachao_dr_plr",
    "reachao_dr_adr", "reachao_dr_adr_plr", "library1", "library2", "narrow_tunnel", "tunnel", "workshop",
    "workshop2", "industrial", "bookshelves", "wall", "kasys", "kitchen", "countertop", "raised_shelves",
    "tabletop", "tabletop2", "warehouse", "dynamic_cross_slow", "dynamic_cross_fast",
)

# default horizon (control steps at n_substeps=20, i.e. 40 ms each) per scene; others use 150
DEFAULT_HORIZON = {"reachao1": 50, "reachao2": 75, "reachao3": 100, "wangexp-3": 100, "library1": 200}

# in-distribution scenes (training curriculum) and the benchmark scenes of evaluate.py
IN_DISTRIBUTION = ("reachao3", "wangexp-3")
BENCHMARK = ("library1", "library2", "narrow_tunnel", "tunnel", "workshop", "workshop2", "industrial", "bookshelves")

# Scene split of the research design v4, section 4. wall, kasys and tabletop2 (and warehouse, which loads tabletop2)
# are unfinished environments that were never meant for testing (authors, 2026-10-08) and are not part of the split.
# library1 is a test scene: an agent reached 20-24 % at horizon 400 (G1, ReachAO_s2; T6 rule).
TRAIN = ("reachao1", "reachao2", "reachao3", "reachao_dr", "reachao_dr_plr", "reachao_dr_adr", "reachao_dr_adr_plr")
VALIDATION = ("tunnel", "workshop2", "industrial", "bookshelves", "kitchen", "countertop", "raised_shelves", "tabletop")
TEST = ("narrow_tunnel", "library2", "workshop", "library1", "dynamic_cross_slow", "dynamic_cross_fast")

# Unfinished environments (fail scripts/check_scenes.py; robot starts inside the scene or goals out of reach), never
# to be used for validation or testing. All other scenarios pass the check (runs/scene_check.json, 2026-10-08).
UNFINISHED_SCENES = ("wall", "tabletop2", "warehouse", "kasys")
assert not set(VALIDATION + TEST) & set(UNFINISHED_SCENES), "unfinished scenes in VALIDATION/TEST"


def scene_index(scene: str) -> int:
    return SCENARIO_ORDER.index(scene)


def horizon(scene: str) -> int:
    return DEFAULT_HORIZON.get(scene, 150)


def check_selectable(scenes) -> None:
    """Model selection is only allowed on validation scenes, and only once the split is frozen."""
    if not VALIDATION or not TEST:
        raise ValueError("scene split (panda_gym/scenes.py TRAIN/VALIDATION/TEST) is not filled in yet")
    test = sorted(set(scenes) & set(TEST))
    if test:
        raise ValueError(f"refusing to select on test scenes {test}")
