import snntorch as snn
from snntorch import spikegen
import torch, torch.nn as nn

hidden_size = 300  # number of hidden neurons


class SNN(nn.Module):
    def __init__(self, input_size, output_size, num_steps):
        super(SNN, self).__init__()

        self.num_steps = num_steps
        beta1 = 0.9
        beta2 = torch.rand(output_size, dtype=torch.float)  # independent decay rate for each leaky neuron in layer
        # 2: [0, 1)

        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float)
        self.lif1 = snn.Leaky(beta=beta1)
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float)
        self.lif2 = snn.Leaky(beta=beta2, learn_beta=True)

    def forward(self, x):

        x = x.float()

        mem1 = self.lif1.init_leaky().float()
        mem2 = self.lif2.init_leaky().float()
        spk2_rec = []  # record output spikes
        mem2_rec = []  # record output hidden states

        for step in range(self.num_steps):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)  # record spikes
            mem2_rec.append(mem2)  # record membrane

        return torch.stack(spk2_rec, dim=1), torch.stack(mem2_rec)
