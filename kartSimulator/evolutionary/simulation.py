import torch
import torch.nn as nn
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

import kartSimulator.evolutionary.utils as utils

n_int = 4
n_frac = 4


class Network(nn.Module):
    def __init__(self):
        # TODO shape similarly to PPO's NN
        super(Network, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class SimRace(Problem):
    def __init__(self):
        super().__init__(n_var=32,
                         n_obj=2,
                         n_constr=0,
                         xl=0,
                         xu=1,
                         vtype=bool,
                         elementwise_evaluation=True,
                         crossover=SinglePointCrossover(),
                         mutation=BitflipMutation())

        self.network = Network()

    def _evaluate(self, x, out, *args, **kwargs):
        # FIXME does not generate binary specimen
        weights = []
        for i in range(8):
            bin_str = "".join(map(str, x[i*4:(i+1)*4]))
            dec_num = utils.bin2dec(bin_str, n_int, n_frac)
            weights.append(dec_num)

        tensor = torch.tensor(weights)
        self.network.load_state_dict(tensor)
        # FIXME
        #torch.save(self.network.state_dict(), 'network.pth')

        x1 = 0.5
        x2 = 0.75

        input_tensor = torch.tensor([x1, x2])

        output = self.network(input_tensor)

        # TODO define the problem properly
        f1 = output
        f2 = 1 - output

        out["F"] = [f1, f2]