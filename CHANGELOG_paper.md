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

## T6b Scene fixes — 2026-10-08

- `workshop`, `workshop3`, `wang-3`: goals the end effector cannot reach are resampled (`ReachAO.goal_reachable`:
  position-only IK for the end-effector point within 1 cm, robot pose restored; enabled per scene by
  `require_reachable_goal`). The goal distribution becomes the reachable part of the old goal box.
- `reachao_rand_start`: the start-pose sampler read the previous physics step's contact manifold, so it never saw
  a collision. With a real check it looped forever, because it runs as `robot.reset()` before the episode's
  obstacles are placed (they still sit at their creation point next to the base). It now only rejects table
  contacts (`robot_touches_table`; obstacle clearance is `set_coll_free_robot`'s job) and gives up after 1000
  tries (neutral pose).
- `reachao_dr`, `reachao_dr_plr`: the fallback obstacle (placed when no other placement succeeded) is now checked
  for clearance from the start pose and the goal like every other placement; parked if no position is found.
- `showcase`: built before the task had an RNG and passed a position as the obstacle name; fixed.
- Not fixed (need the intended layout): `wall` (stop-and-ask point 4), `tabletop2`/`warehouse` (a second table top
  at 0.48-0.50 m spans the robot base; `create_scenario_warehouse` loads `tabletop2`, and the unused `warehouse`
  asset puts shelving on the base), `kasys` (laid out for a base about 1 m further forward).
- `evaluate.py`, `scripts/check_scenes.py`: one BLAS/OpenMP thread per worker. With the default, pools of workers
  exhausted the per-user thread limit on a shared node (numpy import hung).
- Verified: `scripts/check_scenes.py` on all scenarios: only wall, tabletop2, warehouse, kasys fail. Goals and
  start poses of reachao1-3, library2 and workshop2 are identical before and after (the training runs started on
  930af2b are unaffected). `test/scenes_test.py` (8 tests), full suite (117 passed).

## Scene split (design v4, section 4) — 2026-10-08

- `panda_gym/scenes.py`: TRAIN = reachao1-3 and the DR variants; VALIDATION = tunnel, workshop2, industrial,
  bookshelves, kitchen, countertop, raised_shelves, tabletop; TEST = narrow_tunnel, library2, workshop.
  Left out: kasys and tabletop2 (validation in the design, fail check_scenes.py), wall (test once repaired),
  library1 (pending its horizon-400 check, T6), the dynamic scenes (T9, not built). `evaluate.py --select-best`
  now works on validation scenes and refuses test scenes.
- `scripts/summarize_g1.py` also reports the validation and test means. G1 runs on the design's 8 checked scenes,
  three of which (narrow_tunnel, library2, workshop) are test scenes; the G1 decision should rest on the
  validation mean (tunnel, workshop2, industrial, bookshelves).
- Verified: `test/scenes_test.py` (split disjoint, known, checked; selection refuses test scenes), summary on
  synthetic records, full suite.

## Unfinished scenes — 2026-10-08

- Authors' decision: `wall`, `kasys` and `tabletop2` are unfinished environments that were never meant for testing;
  they leave the design (no wall repair, stop-and-ask point 4 closed). `warehouse` loads `tabletop2` and is treated
  the same. `scenes.FAILED_SCENE_CHECK` renamed to `UNFINISHED_SCENES`.

## G2 setup — 2026-10-08

- `--preset thesis_tqc` now includes `n_substeps=5` (10 ms control steps, as the thesis base training; authors'
  decision). Episode lengths of the preset curriculum (50/75/100 steps) are thus 0.5/0.75/1 s, as in the thesis.
- `evaluate.py --n-substeps`: agents are evaluated at the control rate they were trained with; recorded per episode.
- `scripts/phase0_g2.sh`: 3 seeds x {head, thesis} semantics as SLURM GPU jobs, each followed by an evaluation job
  under both collision definitions on reachao3 + the 8 G1 scenes at 400 x 10 ms = 4 s.
  `scripts/summarize_g1.py --out` summarizes any set of runs (G2: runs/eval/g2_summary.csv).
- Verified: 6 k-step run with the thesis semantics at n_substeps=5 (run_meta shows the config), evaluation at
  `--n-substeps 5 --horizon 400` (400 steps recorded), full suite.

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

## T9 Dynamic-obstacle test scenes — 2026-10-08

- `dynamic_cross_slow` (0.1 m/s) and `dynamic_cross_fast` (0.3 m/s): two 5 cm mass-0 spheres per episode, each on a
  straight line through a uniformly random point between the start end-effector position and the goal, random
  direction (horizontal plus a vertical component in [-0.3, 0.3] before normalization), spawned on a sphere of 1 m
  around the shoulder (outside the ~0.85 m reach) and above the table, moved by their base velocity. Goals as in
  reachao3, neutral start. Trajectories are seeded by the episode seed. Added to `scenes.TEST` and `SCENARIO_ORDER`.
- `--obstacle-obs` for train.py and evaluate.py (e.g. `vectors+past`, the history of closest-point vectors). Default
  observation unchanged (partial observability: no obstacle velocity).
- Acceptance: check_scenes.py passes both. A policy holding q0 (150 steps = 6 s) collides in **32 %** (slow, 100
  episodes; within 20-60 %) and **81 %** (fast; above 60 %). Sampling the crossing point only from the last 75 % of
  the start-goal segment gave 20 % / 74 %, so it was not adopted; the fast level is reported as is (decision for the
  authors: e.g. one sphere at the fast level, or accept).
- Verified: `test/dynamic_scene_test.py` (speed, spawn outside the envelope and clear of the robot, seeded
  trajectories, slow-level collision band, fixed history-observation shape).

## T10 Seed-population runner — 2026-10-08

- `scripts/phase1_seeds.sh`: reads `configs/paper_frozen.json` (format: `configs/paper_frozen.example.json`;
  refuses to run without it), seeds 0-19 into `runs/pop/<name>/s<seed>/` with 24 checkpoints, skips `done`;
  `--population xdist` trains the shared end-of-stage-2 checkpoint once, then 5 seeds per stage-3 distribution
  with `--init`; `MODE=sbatch` submits one GPU job per run (stage 3 waits for stage 2), `DRY_RUN=1` prints.
- Verified: dry runs against the example config (20 main runs; 1 + 15 xdist runs).

## T11 Analysis — 2026-10-08

- `analysis/`: `load.py` (per-episode records -> DataFrame with members, K, run, seed, role, success; refuses
  records whose episodes were not shared, by hash), `stats.py` (rliable mean/IQM with stratified-bootstrap CIs,
  probability of improvement, paired stratified bootstrap, exact McNemar, Wilson), `rq1.py`, `rq2.py`, `rq3.py`
  (CSV + PDF in `analysis/out/`), `subsets.py` (50 random subsets per K with `default_rng(0)`, prints the
  evaluate.py commands). Run as `python -m analysis.rq1 <records> --collision-def ... --horizon ...`.
- Choices the design left open:
  - The RQ1 mixed model is the Bayesian `BinomialBayesMixedGLM` (`success ~ C(scene)` with seed and seed x scene
    variance components) on OOD episodes; reported as posterior random-effect SDs with 95 % intervals.
  - RQ2 baseline: the single policy with the highest mean validation success. Paired differences use, per test
    episode, the fraction of subsets that solved it. McNemar is reported where a (strategy, K) has one subset.
  - RQ3 source from the members' run directories: snapshot (one run), xdist (`*_xdist/<dist>` with several
    distributions), hybrid, seed.
- `analysis` extra in pyproject.toml: pandas<3, statsmodels, rliable, arch>=5.3.1,<8, matplotlib (rliable needs
  arch<8, which needs pandas<3).
- Verified: `test/analysis_test.py` on synthetic records (derived columns, hash check, all outputs written,
  validation-only baseline selection, statistics, reproducible subsets).

## Runner hardening — 2026-10-08

- Two G1 jobs on BeeGFS failed after ~20 min: every new Python process reported "failed to make path absolute"
  (the job's working directory went stale on the compute node). `phase0_g1.sh` and `phase0_g2.sh` now re-enter
  the repository by absolute path before every evaluation. Completed result files were kept (runners resume).

## T8 NEO reference — 2026-10-08

- `panda_gym/baselines/neo.py`: port of the thesis NEO (`9e12871^:panda_gym/envs/robots/panda.py:compute_action_neo`
  plus the collision damper of the thesis' roboticstoolbox fork, `RaikoPipe/robotics-toolbox-python` e07d313).
  Same gains, QP weights (incl. the thesis error e with rpy * pi/180), joint-limit damper, manipulability term,
  one distance per link-obstacle pair (panda_link1..panda_hand + panda_ee, obstacles without the table),
  IK-derived target orientation, velocity control (`control_type="jsd"`, actions clipped to +-1 rad/s as in the
  thesis), 25 Hz. roboticstoolbox -> pybullet Jacobians, Gurobi -> OSQP (`qpsolvers[osqp]` in the train extra).
- Corrected relative to the thesis: its damper multiplied a world-frame normal with an end-effector-frame Jacobian of
  a tool offset equal to the link's world pose; here it is the world-frame Jacobian of the closest point.
- pybullet's `calculateJacobian` takes `localPosition` in the link's URDF frame, not the centre-of-mass frame its
  documentation suggests (a first version using the COM frame was off by up to 0.04 m/rad). pybullet's kinematics
  are single precision: finite-difference checks need steps around 1e-3 rad.
- `evaluate.py --controller neo` runs NEO on the shared episodes (record `strategy: "neo"`).
- Acceptance (reachao3, thesis collision definition): 4 % (h100 = 4 s, 50 ep.), 36 % (h150, 50), 70 % (h200, 50),
  **76 % (h400 = 16 s, 200 ep.; Wilson 95 % CI about 70-81 %)**, no collisions, timeouts otherwise; median 158 steps
  (6.3 s) to success, 6 ms/step. Thesis Tab. 6-4: 60.8 % -> +15 pp, outside +-10 pp: stop-and-ask point 5. Likely
  causes: the corrected damper Jacobian, and the thesis' NEO evaluation setting (the thesis-era code ran NEO at
  100 Hz for 100 steps = 1 s, which cannot produce 60.8 %, so the reported number used another setting).
- Verified: `test/neo_test.py` (Jacobian vs. central differences for three links, state untouched by a control
  step, finite actions with active obstacle constraints, shared episodes through the harness).

## Decisions after G1/G2 — 2026-10-08

- Frozen collision definition: strict (thesis as a secondary column). Frozen horizon: 150 control steps (6 s).
- `library1` joins TEST (G1: ReachAO_s2 reaches 18-24 %, non-zero at horizon 400; T6 rule).
- NEO: the corrected port is accepted; its +15 pp deviation from the thesis value is reported, not tuned.
- `dynamic_cross_fast` uses one sphere (two collided in 81 % of hold-q0 episodes). Re-measured on 200 episodes:
  slow (two spheres) 36 %, fast (one sphere) 65 %, i.e. at the upper edge of the 20-60 % band (95 % CI ~58-71 %).
- G3 (`scripts/phase0_g3.sh`): the TQC cells use the thesis preset at `--n-substeps 20` (the frozen 25 Hz control
  rate), and the last curriculum stage is uncapped, so every cell trains 1.2 M steps.

## Cluster usage — 2026-10-09

- GPU jobs submitted by the runners exclude gpu01-gpu03 (reserved for other work; `EXCLUDE_NODES` overrides).
