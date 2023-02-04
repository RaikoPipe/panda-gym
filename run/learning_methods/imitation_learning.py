import numpy as np
from time import sleep
from stable_baselines3 import TD3
from algorithms.SAC_hybrid.prior_controller_neo import NEO
from copy import copy


def fill_replay_buffer(model: TD3, num_steps=10000):
    """
    Fill replay buffer of the given model with prior actions.
    :param prior: The prior
    :param num_steps: (int) number of steps to fill the replay buffer
    :return: model with filled replay buffer
    """
    env = model.env.envs[0]

    episode_rewards = [0.0]
    obs, _ = env.reset()

    replay_buffer = model.replay_buffer

    done_events = []
    for i in range(num_steps):
        action = env.robot.compute_action_neo(env.task.goal, env.task.dummy_obstacles)

        env.robot.set_action(action, clip=False)
        env.sim.step()
        next_obs, reward, done, truncated, info, = env.step(np.zeros(7))
        replay_buffer.add(obs, next_obs, action, reward, done, [info])

        model.train(gradient_steps=-1)

        obs = next_obs

        # Stats
        episode_rewards[-1] += reward
        if done or truncated:
            if info["is_success"]:
                print("Success!")
                done_events.append(1)
            elif info["is_truncated"]:
                print("Collision...")
                done_events.append(-1)
            else:
                print("Timeout...")
                done_events.append(0)
            obs, _ = env.reset()
            # unnecessary step (?) reset panda

            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = np.mean(episode_rewards[-100:])
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
    print(f"Success Rate: {done_events.count(1) / len(done_events)}")
    print(f"Collision Rate: {done_events.count(-1) / len(done_events)}")

    return replay_buffer
