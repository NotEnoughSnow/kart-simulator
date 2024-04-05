import snntorch as snn
from snntorch import spikegen
import torch.nn as nn
import torch, torch.nn as nn


hidden_size = 300  # number of hidden neurons


class SNN(nn.Module):
    def __init__(self, input_size, output_size, num_steps):
        super(SNN, self).__init__()

        self.num_steps = num_steps
        beta1 = 0.9 # global decay rate for all leaky neurons in layer 1
        beta2 = torch.rand((output_size), dtype = torch.float) # independent decay rate for each leaky neuron in layer 2: [0, 1)

        # Initialize layers using snnTorch's Leaky integrate-and-fire neurons
        self.fc1 = snn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta1)  # beta is the decay rate of the membrane potential
        self.fc2 = snn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta2)

    def forward(self, x):
        # Initialize the membrane potentials to zero for each forward pass
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = [] # record output spikes
        mem2_rec = [] # record output hidden states

        # Loop over time steps
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2) # record spikes
            mem2_rec.append(mem2) # record membrane

        return torch.stack(spk2_rec), torch.stack(mem2_rec)