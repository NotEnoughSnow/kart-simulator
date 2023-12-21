from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from kartSimulator.evolutionary.simulation import SimRace

def run():

    problem = SimRace()

    #problem = get_problem("zdt1")



    algorithm = NSGA2(pop_size=100)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 10),
                   seed=1,
                   verbose=False)

    # calculate a hash to show that all executions end with the same result
    print("hash", res.F)
