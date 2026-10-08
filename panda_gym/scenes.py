"""Scene roles and shared-episode indexing for the paper experiments.

SCENARIO_ORDER fixes each scene's index in the episode seeds (evaluate.py: seed = base + 1000 * index + episode), so
it is append-only: never reorder or remove entries, or old evaluations stop being comparable.
"""

SCENARIO_ORDER = (
    "reachao1", "reachao2", "reachao3", "wangexp-3", "reachao_rand", "reachao_dr", "reachao_dr_plr",
    "reachao_dr_adr", "reachao_dr_adr_plr", "library1", "library2", "narrow_tunnel", "tunnel", "workshop",
    "workshop2", "industrial", "bookshelves", "wall", "kasys", "kitchen", "countertop", "raised_shelves",
    "tabletop", "tabletop2", "warehouse",
)

# default horizon (control steps at n_substeps=20, i.e. 40 ms each) per scene; others use 150
DEFAULT_HORIZON = {"reachao1": 50, "reachao2": 75, "reachao3": 100, "wangexp-3": 100, "library1": 200}

# in-distribution scenes (training curriculum) and the benchmark scenes of evaluate.py
IN_DISTRIBUTION = ("reachao3", "wangexp-3")
BENCHMARK = ("library1", "library2", "narrow_tunnel", "tunnel", "workshop", "workshop2", "industrial", "bookshelves")

# Scene split of the research design v4, section 4, restricted to scenes that pass scripts/check_scenes.py.
# Left out for now: kasys, tabletop2 (validation in the design; fail the check), wall (test once repaired),
# library1 (test only if an agent reaches non-zero success at horizon 400, T6), the dynamic scene (T9, not built).
TRAIN = ("reachao1", "reachao2", "reachao3", "reachao_dr", "reachao_dr_plr", "reachao_dr_adr", "reachao_dr_adr_plr")
VALIDATION = ("tunnel", "workshop2", "industrial", "bookshelves", "kitchen", "countertop", "raised_shelves", "tabletop")
TEST = ("narrow_tunnel", "library2", "workshop")

# scripts/check_scenes.py, 20 resets each (runs/scene_check.json, 2026-10-08, after the scene fixes):
#   wall                  robot starts inside the wall in 20/20 resets (panda_link3); repair needs a 36 cm shift
#   tabletop2, warehouse  a second table top at 0.48-0.50 m spans the robot base (panda_link3 inside it, 20/20);
#                         warehouse also loads tabletop2 (its own asset puts shelving on the robot base)
#   kasys                 goals at x = 1.4-1.7 m, out of reach (20/20); the scene is laid out for another base
# Fixed on 2026-10-08: workshop/workshop3/wang-3 (unreachable goals are resampled), showcase (crashed),
# reachao_dr/_plr (fallback obstacle could touch the start pose), reachao_rand_start (start pose check).
FAILED_SCENE_CHECK = ("wall", "tabletop2", "warehouse", "kasys")
assert not set(VALIDATION + TEST) & set(FAILED_SCENE_CHECK), "scenes failing check_scenes.py in VALIDATION/TEST"


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
