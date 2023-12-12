import sys

import torch

import kartSim.kartSim as kart_sim
from core.arguments import get_args
from core.network import FeedForwardNN
from core.ppo import PPO

def play(env):

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            s, r, terminated, truncated, _ = env.step(None)
            total_reward += r

            #print(r)

            steps += 1

            if truncated:
                env.reset()

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

    #load actor / critic

    model.learn(total_timesteps=10000)

def optimize():
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.problems import get_problem
    from pymoo.optimize import minimize

    problem = get_problem("zdt1")

    algorithm = NSGA2(pop_size=100)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 100),
                   seed=1,
                   verbose=True)

    # calculate a hash to show that all executions end with the same result
    print("hash", res.F.sum())

def optimize_2():

    # Import the necessary libraries
    import numpy as np
    import random
    import math

    # Define the problem parameters
    n_var = 10  # number of decision variables
    n_obj = 3  # number of objectives
    n_con = 0  # number of constraints
    var_min = 0  # lower bound of decision variables
    var_max = 1  # upper bound of decision variables

    # Define the objective functions
    # You may need to change these according to your problem definition
    def f1(x):
        # A sample objective function that sums up the values of the decision variables
        return np.sum(x)

    def f2(x):
        # A sample objective function that calculates the standard deviation of the decision variables
        return np.std(x)

    def f3(x):
        # A sample objective function that counts the number of zeros in the decision variables
        return np.count_nonzero(x == 0)

    # Define the constraint functions
    # You may need to change these according to your problem definition
    def g1(x):
        # A sample constraint function that checks if the product of the decision variables is less than 0.5
        return np.prod(x) - 0.5

    # Define the evaluation function
    # This function evaluates a single solution (individual) and returns its objective values and constraint violations
    def evaluate(x):
        # Calculate the objective values
        obj = []
        obj.append(f1(x))
        obj.append(f2(x))
        obj.append(f3(x))

        # Calculate the constraint violations
        con = []
        con.append(g1(x))

        return obj, con

    # Define the population size
    pop_size = 100

    # Define the number of generations
    n_gen = 100

    # Define the crossover probability
    cx_prob = 0.9

    # Define the mutation probability
    mut_prob = 0.1

    # Define the distribution indexes for crossover and mutation
    eta_c = 20
    eta_m = 20

    # Define the selection function
    # This function selects two individuals from the population using binary tournament selection
    # The selection is based on the nondomination rank and the crowding distance
    def select(pop):
        # Choose two random individuals from the population
        a = random.randint(0, pop_size - 1)
        b = random.randint(0, pop_size - 1)
        # Compare their ranks and crowding distances
        if pop[a]["rank"] < pop[b]["rank"]:
            # Choose the one with lower rank
            return pop[a]
        elif pop[a]["rank"] > pop[b]["rank"]:
            # Choose the one with lower rank
            return pop[b]
        else:
            # If their ranks are equal, choose the one with higher crowding distance
            if pop[a]["dist"] > pop[b]["dist"]:
                return pop[a]
            elif pop[a]["dist"] < pop[b]["dist"]:
                return pop[b]
            else:
                # If their crowding distances are equal, choose one at random
                return pop[a] if random.random() < 0.5 else pop[b]

    # Define the crossover function
    # This function performs simulated binary crossover (SBX) on two individuals
    def crossover(a, b):
        # Initialize the offspring
        c = {"x": np.zeros(n_var), "obj": np.zeros(n_obj), "con": np.zeros(n_con), "rank": 0, "dist": 0.0}
        d = {"x": np.zeros(n_var), "obj": np.zeros(n_obj), "con": np.zeros(n_con), "rank": 0, "dist": 0.0}
        # Perform crossover on each decision variable
        for i in range(n_var):
            # Generate a random number
            u = random.random()
            # Calculate the crossover factor
            if u < 0.5:
                bq = (2 * u) ** (1 / (eta_c + 1))
            else:
                bq = (1 / (2 - 2 * u)) ** (1 / (eta_c + 1))
            # Generate the offspring
            c["x"][i] = 0.5 * ((1 + bq) * a["x"][i] + (1 - bq) * b["x"][i])
            d["x"][i] = 0.5 * ((1 - bq) * a["x"][i] + (1 + bq) * b["x"][i])
            # Make sure the offspring are within the bounds
            c["x"][i] = min(max(c["x"][i], var_min), var_max)
            d["x"][i] = min(max(d["x"][i], var_min), var_max)
        # Return the offspring
        return c, d

    # Define the mutation function
    # This function performs polynomial mutation on an individual
    def mutate(a):
        # Perform mutation on each decision variable
        for i in range(n_var):
            # Generate a random number
            u = random.random()
            # Calculate the mutation factor
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta_m + 1)) - 1
            else:
                delta = 1 - (2 - 2 * u) ** (1 / (eta_m + 1))
            # Mutate the individual
            a["x"][i] = a["x"][i] + delta * (var_max - var_min)
            # Make sure the individual is within the bounds
            a["x"][i] = min(max(a["x"][i], var_min), var_max)
        # Return the mutated individual
        return a

    # Define the fast nondominated sort function
    # This function sorts the population according to the nondomination rank
    def fast_nondominated_sort(pop):
        # Initialize the front counter
        front_counter = 0
        # Initialize the first front
        current_front = []
        # Initialize the domination count and dominated set for each individual
        for i in range(pop_size):
            pop[i]["dom_count"] = 0
            pop[i]["dom_set"] = []
            # Compare each individual with the rest of the population
            for j in range(pop_size):
                # Check for domination
                if i != j:
                    # Assume that i dominates j
                    i_dom_j = True
                    # Assume that j dominates i
                    j_dom_i = True
                    # Check for violation of any constraint
                    if np.any(pop[i]["con"] > 0) or np.any(pop[j]["con"] > 0):
                        # If both individuals are infeasible, compare their overall constraint violation
                        if np.any(pop[i]["con"] > 0) and np.any(pop[j]["con"] > 0):
                            i_dom_j = np.sum(pop[i]["con"]) < np.sum(pop[j]["con"])
                            j_dom_i = np.sum(pop[j]["con"]) < np.sum(pop[i]["con"])
                        # If one individual is feasible and the other is not, the feasible one dominates
                        elif np.any(pop[i]["con"] > 0):
                            i_dom_j = False
                            j_dom_i = True
                        elif np.any(pop[j]["con"] > 0):
                            i_dom_j = True
                            j_dom_i = False

def test():

    #init policy

    #load model

    #eval policy
    pass


def main(args):

    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    hyperparameters = {
        #'timesteps_per_batch': 2048,
        'timesteps_per_batch': 512,
        #'max_timesteps_per_episode': 200,
        'max_timesteps_per_episode': 200,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }
    manual = False

    env = kart_sim.KartSim(render_mode=None, manual=manual)

    if manual:
        play(env=env)
    if args.mode == "train":
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    if args.mode == "optimize":
        #optimize()
        optimize_2()

if __name__ == "__main__":
    args = get_args()  # Parse arguments from command line
    #args.actor_model = "ppo_actor.pth"
    #args.critic_model = "ppo_critic.pth"
    args.mode = "optimize"
    main(args)