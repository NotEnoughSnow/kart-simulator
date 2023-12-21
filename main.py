import sys

import torch

from kartSimulator.core.arguments import get_args
from kartSimulator.core.network import FeedForwardNN
from kartSimulator.core.ppo import PPO
import kartSimulator.evolutionary.core as EO

import kartSimulator.sim.sim as kart_sim

def play(env):
    running = False

    while not running:
        env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            s, r, terminated, truncated, _ = env.step(None)
            total_reward += r
            steps += 1

            if truncated:
                env.reset()
            if terminated:
                print(f"total rewards this lap:{total_reward}")
                # TODO times

    env.close()


def train(env, hyperparameters, actor_model, critic_model):
    model = PPO(env=env, policy_class=FeedForwardNN, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '':  # Don't train from scratch if user accidentally forgets actor/critic model
        print(
            f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # load actor / critic

    model.learn(total_timesteps=10000)


def optimize():
    # TODO
    EO.run()


def test():
    # TODO
    # init policy

    # load model

    # eval policy
    pass


def main(args):
    hyperparameters = {
        'timesteps_per_batch': 2048,
        'max_timesteps_per_episode': 200,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }

    if args.mode == "play":
        env = kart_sim.KartSim(render_mode="human", manual=True)
        play(env=env)
    if args.mode == "train":
        env = kart_sim.KartSim(render_mode=None, manual=False)
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    if args.mode == "optimize":
        # TODO implement evolutionary optimization
        pass
    if args.mode == "test":
        env = kart_sim.KartSim(render_mode="Human", manual=False)
        test()


if __name__ == "__main__":
    args = get_args()

    # you can also directly set the args
    # args.mode = "train"

    main(args)
