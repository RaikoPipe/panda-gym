from dataclasses import dataclass, field
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer, DictReplayBuffer
from classes.hyperparameters import Hyperparameters


@dataclass
class TrainConfig:
    # wandb settings
    name: str = 'default'
    job_type: str = 'train'
    group: str = "default"

    # learning settings
    algorithm: str = 'TQC'
    replay_buffer_class: object = HerReplayBuffer
    policy_type: str = 'MultiInputPolicy'
    learning_starts: int = 10000
    prior_steps: int = 0
    seed: int = 0

    # performance settings
    n_envs: int = 8

    # environment settings
    env_name: str = 'PandaReachAO-v3'
    randomize_robot_pose: bool = False
    truncate_on_collision: bool = True
    terminate_on_success: bool = True
    fixed_target: list = None

    # rewards settings
    reward_type: str = 'sparse'
    collision_reward: int = -100

    # goal condition settings
    goal_condition: str = 'reach'
    ee_error_thresholds: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])
    speed_thresholds: list[float] = field(default_factory=lambda: [0.5, 0.1, 0.01])
    safety_distance: float = 0.0

    # temporal settings
    max_timesteps: int = 600_000
    max_ep_steps: list[int] = field(default_factory=lambda: [50, 75, 100])
    n_substeps: int = 20

    # curriculum setup
    stages: list[str] = field(default_factory=lambda: ["reachao1", "reachao2", "reachao3"])
    success_thresholds: list[float] = field(default_factory=lambda: [0.9, 0.9, 1.0])

    # evaluation settings
    n_eval_envs: int = 32
    eval_freq: int = 10_000
    benchmark_eval_freq: int = 50_000
    n_eval_episodes: int = 100
    n_benchmark_eval_episodes: int = 100

    # observations and actions
    obs_type: tuple = ("ee", "js")
    control_type: str = "js"
    action_limiter: str = "clip"
    limiter: str = "sim"
    task_observations: dict = field(default_factory=lambda: {'obstacles': "vectors+closest_per_link", 'prior': None})

    # visualization
    render: bool = False
    show_goal_space: bool = False
    show_debug_labels: bool = False
    debug_collision: bool = False

    # snapshot settings
    snapshot_freq: int = 50_000

    # hyperparams
    hyperparams: Hyperparameters = field(default_factory=lambda: Hyperparameters(algorithm='TQC'))
