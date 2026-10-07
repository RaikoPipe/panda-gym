# Paper changelog (branch `paper/phase0`)

Research design draft v4, section 7. Each entry says what changed and how it was verified. Defaults reproduce
HEAD `0596f5d` behaviour except where an entry says otherwise.

## T1 Configurable collision definition — 2026-10-07

- `ReachAOConfig`: `collision_ignore_links`, `table_collides`, `collision_semantics` (`terminate`/`truncate`),
  `her_penalty_source` (`info`/`live`), and one field the design did not list, `collision_check`
  (`contact`: contact manifold of every substep, HEAD; `distance`: closest-point query once after all substeps, as
  in the thesis code `1624534`). Presets `ReachAOConfig.strict()`, `.thesis()`, `.from_collision_def(name)`;
  named definitions in `COLLISION_DEFS`: `strict`, `strict_notable`, `thesis`, `thesis_table`, plus
  `thesis_substep` (thesis link/table set, HEAD timing), which separates the timing effect from the link set.
- `Task.is_truncated()` hook; `RobotTaskEnv.step` returns `truncated=True` on a collision under `truncate`.
- `her_penalty_source="live"` reproduces the thesis reward exactly: `-(d >= eps) - 100 * live_flag`, no
  collision gating. In training, the HER reward is then computed by the training env (`train.py`), not by the
  in-process reward env, whose live flag would always be False.
- `task.collision_link` names (robot link, body) of the last collision (evaluation records).
- Findings while implementing (see report):
  - The ReachAO URDF has no collision geometry on `panda_leftfinger`, `panda_rightfinger`, `panda_grasptarget`.
    The thesis exclusion of these links is a no-op. `a816fc9` moved three of the hand's six collision shapes onto
    the new `panda_ee` link; the robot's collision geometry is identical to the thesis URDF. The design's T1 test
    "a sphere touching only panda_leftfinger" is therefore impossible; replaced by a test that these links have no
    geometry and a test of `collision_ignore_links` on `panda_link4`.
  - The thesis checked collisions once per control step; a penetration that the solver resolves within the 20
    substeps is invisible to it (`test_step_end_check_misses_penetration_resolved_within_the_step`).
- Verified: `test/collision_def_test.py` (17 tests), full suite.

## T2 Evaluation harness — 2026-10-07

- `evaluate.py` rewritten, old call form kept (`python evaluate.py model.zip --scenes ...`; default episodes now
  200 as specified). Flags: `--models` (files or globs), `--horizon`, `--episodes`, `--collision-def`,
  `--episode-seed-base`, `--deterministic/--stochastic`, `--out`, `--ensemble-strategy/--space/--reduce` (T7),
  `--select-best` (refuses unless the scene split is filled in and no TEST scene is involved), `--workers`.
- Shared episodes: seed = base + 1000 * `scenes.scene_index(scene)` + i; `SCENARIO_ORDER` is append-only.
  Per-episode JSON lines with the fields of the design plus a hash of (goal, obstacle poses, q0).
- A fresh env per episode instead of a reused DummyVecEnv: PyBullet's solver order depends on an env's history,
  and `panda_link1` permanently rubs on the table, so a reused env did not replay an episode seed (same start
  hash, different trajectory). Costs about 0.25 s per episode.
- Two determinism fixes in the package (behaviour change, ordering only):
  - `PyBullet.__init__` sets `deterministicOverlappingPairs=1` (contact pairs ordered by body id instead of memory
    address).
  - `RobotTaskEnv.__init__` resets with `seed=0`: scenes that sample the start pose by IK (narrow_tunnel,
    industrial, workshop2) start IK from the current joint state, which came from an unseeded constructor reset.
    SB3 re-seeds every env at its first reset, so training is unaffected.
- Verified: two runs of the same model give identical outcomes, hashes, steps and path lengths; 20 episodes on
  reachao3 take ~35 s; `test/evaluate_test.py` (order/chunking independence, shared episodes across collision
  definitions on random-start scenes, record fields). One unexplained failure of the order test occurred once
  right after the pair-ordering fix and did not recur in ~25 runs; records carry hashes so any mismatch is caught.

## T3 G1 runner — 2026-10-07

- `scripts/phase0_g1.sh` (resumable, SLURM-ready, CPU only) and `scripts/summarize_g1.py` (Wilson 95 % CIs,
  `runs/eval/g1_summary.csv`, warns if runs did not see the same episodes).
- Grid: the design's four definitions plus `thesis_substep`; horizons 100 / 150 / 400 control steps. 100 steps
  (4 s) was added because the thesis episode was 4 s at both of its control rates (400 x 10 ms, 100 x 40 ms);
  "400" in the thesis means 10 ms steps, not HEAD's 40 ms.
- Verified: end to end on a smoke model (8 episodes); the hash check caught the IK start-pose bug fixed in T2.
- Not run: the agents (`runs/cmp/*`, `runs/ReachAO/final_reachao3.zip`) are not on this machine.

## T4 Thesis preset and training flags — 2026-10-07

- `train.py`: `--preset thesis_tqc`, `--train-freq`, `--gradient-steps`, `--utd` (now exact: fractional UTDs map
  to `train_freq`; the old `round(utd * n_envs)` silently clamped to >= 1 gradient step per vec step),
  `--learning-rate`, `--lr-schedule`, `--use-sde`, `--log-std-init`, `--net-arch`, `--tau`, `--buffer-size`,
  `--collision-def`, `--collision-semantics`, `--her-penalty-source`, `--n-substeps`. Explicit flags override the
  preset, the preset overrides the defaults (unchanged: CrossQ, UTD 1, gamma 0.95, ...).
- Curriculum stages accept a step cap: `scenario:max_ep_steps[:threshold[:max_stage_steps]]`.
- Logs `time/env_steps_per_s` and `time/grad_steps_per_s` every 10 k steps; `runs/<name>/evals.jsonl` records
  each evaluation with wall-clock time (for G3 time-to-target); `runs/<name>/run_meta.json` (git SHA, dirty flag,
  args, ReachAOConfig, package versions, host, start/end) via `panda_gym/run_meta.py`, also written by evaluate.py.
- Note: the thesis preset is UTD 0.125 (8 gradient steps every 8 steps of 8 envs), not UTD 1.
- Not in the preset: the thesis base training ran at `n_substeps = 5` (1624534:run/train.py `base_train`);
  available as `--n-substeps 5`, pending a decision.
- Verified: `train.py PandaReachAO-v3 --preset thesis_tqc --steps 20000` runs (334 env steps/s on an A30);
  run_meta.json shows the preset values and effective UTD 0.125; a thesis-semantics run (thesis definition,
  truncate, live penalty) trains.

## T5 G3 sweep runner — 2026-10-07

- `scripts/phase0_g3.sh` (sequential or `MODE=sbatch`, one GPU job per cell; low-UTD cells first; requires
  `COLLISION_DEF` and a filled-in VALIDATION list) and `scripts/summarize_g3.py` (`runs/sweep/g3_summary.csv`).
- Verified: one miniature cell end to end (training with stage caps and threshold, evals.jsonl, in-distribution
  and validation evaluation, summary); `--select-best` refuses while the split is empty.

## T6 Scene integrity — 2026-10-07

- `scripts/check_scenes.py`: all 34 scenarios x 20 resets -> `runs/scene_check.json`; `--search-wall`.
  Reachability is measured for the point the env uses as end-effector position, the centre of mass of link 11
  (`panda_rightfinger` in the current URDF; it was `panda_grasptarget` in the thesis URDF before `panda_ee` was
  inserted and shifted the indices).
- Fails: wall, tabletop2, warehouse (start inside an obstacle, 20/20), kasys (goals out of reach, 20/20),
  workshop/workshop3 (2/20 goals out of reach), wang-3 (1/20), showcase (crashes). Training scenes reachao_dr,
  reachao_dr_plr, reachao_rand_start start in collision in 1/20 resets. Recorded in
  `panda_gym/scenes.py: FAILED_SCENE_CHECK`, which VALIDATION/TEST may not contain.
- `wall` not repaired: the smallest +x shift that passes (20/20 collision-free, >= 5 cm to panda_link1-3) is
  36 cm, which moves the wall from the base to in front of the robot; the wall does not block the straight
  start-goal path at any offset, including the original. Stop-and-ask point 4.
- `library1` horizon-400 check pending (needs the agents).

## T7 Ensemble inference — 2026-10-07

- `panda_gym/ensemble.py`: `PolicyEnsemble(models, strategy, space, reduce, rng)` with mean, precision_weighted,
  bayes_sample, min_variance, critic_argmax, random. sigma via the actors' distributions (correct under gSDE);
  identical architectures run as one vmapped forward pass, heterogeneous ones member by member.
- Verified: `test/ensemble_test.py` (23 tests): K=1 reproduces `model.predict(deterministic=True)` to 1e-6 for
  TQC, TQC+gSDE and CrossQ under four strategies; identical members fuse to the member; shapes; vmapped == looped
  statistics; gSDE sigma is state dependent; K=20 latency 2.3-2.4 ms/step on one CPU thread.

## Other

- `pyproject.toml`: `scipy` added to the dependencies (imported by `panda_gym.envs.tasks.flip`; a fresh install
  could not import panda_gym); pytest runs with the repository root on the path.
