import numpy as np
from time import sleep
from stable_baselines3 import TD3
from algorithms.SAC_hybrid.prior_controller_neo import NEO


def fill_replay_buffer(model: TD3, num_steps=10000):
    """
    Fill replay buffer of the given model with prior actions.
    :param prior: The prior
    :param num_steps: (int) number of steps to fill the replay buffer
    :return: model with filled replay buffer
    """
    env = model.env.envs[0]
    prior = NEO(env)
    episode_rewards = [0.0]
    last_obs, _ = env.reset()

    done_events = []
    for i in range(num_steps):
        action = prior.compute_action(env.task.goal)

        obs, reward, done, truncated, info, = env.step(action)
        model.replay_buffer.add(last_obs, obs, action, reward, done, [info])

        last_obs = obs


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
            prior.panda_rtb.q = prior.panda_rtb.qr
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = np.mean(episode_rewards[-100:])
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
    print(f"Success Rate: {done_events.count(1)/len(done_events)}")
    print(f"Collision Rate: {done_events.count(-1) / len(done_events)}")

    return model