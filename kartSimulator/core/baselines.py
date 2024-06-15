import glob
import os
import time

import numpy as np
from stable_baselines3 import PPO
# import supersuit as ss
from stable_baselines3.ppo import MlpPolicy

import kartSimulator.sim.base_env as base_env
import kartSimulator.sim.simple_env as simple_env

import kartSimulator.sim.observation_types as obs_types

# Create a function to generate a unique directory name
def create_unique_log_dir(base_log_dir, name_prefix):
    log_dir = os.path.join(base_log_dir, name_prefix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def train(env,logs_dir, record_tb, type, steps: int = 100):

    env.reset()

    policy_kwargs = dict(full_std=True)

    unique_log_dir = create_unique_log_dir(logs_dir, type)

    model_args = {
        'batch_size': 256,
        'learning_rate': 0.001,
        'ent_coef': 0.001,
        #'use_sde': True,
        #'policy_kwargs': policy_kwargs,
    }

    if record_tb:
        model_args['tensorboard_log'] = unique_log_dir

    print(model_args)

    # Model
    model = PPO(
        MlpPolicy,
        env,
        verbose=2,
    )
    #     model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, batch_size=256,)
    #     model = DQN("MlpPolicy", env, verbose=1, batch_size=256,)

    # Train
    model.learn(total_timesteps=steps, reset_num_timesteps=1000, progress_bar=True)

    policy_count = 0
    try:
        latest_policy = max(
            glob.glob(f"saved_models/{type}*.zip"), key=os.path.getctime
        )
        print("found :", latest_policy)
        policy_count = int(latest_policy[-5]) + 1


    except ValueError:
        policy_count = 1

    model.save(f"saved_models/{type}_{policy_count}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env, type, deterministic):
    print(env.metadata['name'])

    try:
        latest_policy = max(
            glob.glob(f"saved_models/{type}*.zip"), key=os.path.getctime
        )

    except ValueError:
        print("Policy not found.")
        exit(0)

    print("loading :", latest_policy)

    model = PPO.load(latest_policy)

    obs = env.reset()[0]
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _state = model.predict(obs, deterministic=deterministic)

        # print("actions", action)

        obs, reward, terminated, truncated, _ = env.step(action)

        # VecEnv resets automatically
        # if done:
        #   obs = vec_env.reset()

