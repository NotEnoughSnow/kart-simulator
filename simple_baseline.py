import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import kartSimulator.sim.observation_types as obs_types
from stable_baselines3.ppo import MlpPolicy


import kartSimulator.sim.base_env as empty_gym_sim
import kartSimulator.sim.simple_env as empty_gym_full_sim

def train(sim_type, steps_to_train, **kwargs):
    # Create a vectorized environment (only one process for simplicity)
    env = sim_type.KartSim(render_mode=None, train=True, **kwargs)

    env.reset()

    # Normalize the environment (optional but recommended)
    # env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # Create the PPO agent
    model = PPO(MlpPolicy, env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=steps_to_train)

    # Save the trained model (optional)
    model.save("simple_bl")

    print("done training")

def evaluate(sim_type, num_episodes, deterministic, **kwargs):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    env = sim_type.KartSim(render_mode="human", train=False, **kwargs)

    model = PPO.load("archive/simple_bl.zip")


    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        terminated = False
        truncated = False
        obs, _ = env.reset()
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward




# Create your custom continuous environment (replace with your own)

obs_list = [obs_types.DISTANCE,
            obs_types.TARGET_ANGLE,]


kwargs = {
    "obs_seq": obs_list,
}

sim_type = empty_gym_full_sim


#train(sim_type, 10000, **kwargs)

#evaluate(sim_type, 5, False, **kwargs)

env = sim_type.KartSim(render_mode=None, train=True, **kwargs)

env.reset()

# Normalize the environment (optional but recommended)
# env = VecNormalize(env, norm_obs=True, norm_reward=False)

# Create the PPO agent
model = PPO(MlpPolicy, env, verbose=1)

# Train the agent
model.learn(total_timesteps=300000)

env = sim_type.KartSim(render_mode="human", train=False, **kwargs)

for i in range(10):
    terminated = False
    truncated = False
    obs, _ = env.reset()
    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)


# same model, det off