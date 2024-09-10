from dataclasses import dataclass, field
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer, DictReplayBuffer
from classes.hyperparameters import Hyperparameters

@dataclass
class TrainConfig:
    # wandb settings
    name: str = 'default'
    job_type: str = 'train'

    # learning settings
    algorithm: str = 'TQC'
    replay_buffer_class: str = HerReplayBuffer
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

    # temporal settings
    max_timesteps: int = 1_000_000
    max_ep_steps: list[int] = field(default_factory=lambda: [75, 150, 200])
    n_substeps: int = 20

    # curriculum setup
    stages: list[str] = field(default_factory=lambda: ["reachao1", "reachao2", "reachao3"])
    success_thresholds: list[float] = field(default_factory=lambda: [0.8, 0.8, 1.0])
    eval_freq: int = 5000

    # observations and actions
    obs_type: tuple = ("ee", "js")
    control_type: str = "js"
    action_limiter: str = "clip"
    limiter: str = "sim"
    task_observations: dict = field(default_factory=lambda: {'obstacles': "vectors", 'prior': None})

    # visualization
    render: bool = False
    show_goal_space: bool = False
    show_debug_labels: bool = False

    # hyperparams
    hyperparams: Hyperparameters = field(default_factory=lambda: Hyperparameters(algorithm='TQC'))

