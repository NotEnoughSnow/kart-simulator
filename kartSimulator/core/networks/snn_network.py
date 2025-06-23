import snntorch as snn
from snntorch import spikegen
import torch, torch.nn as nn
from snntorch import surrogate

import numpy as np

hidden_size = 128  # Number of hidden neurons

class SNN(nn.Module):
    def __init__(self, input_size, output_size, num_steps, add_weight):
        super(SNN, self).__init__()

        self.num_steps = num_steps
        beta1 = 0.9
        beta2 = 0.9
        #beta3 = torch.rand((output_size), dtype=torch.float)  # Independent decay rate for each output neuron
        beta3 = 0.9

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float)
        self.fc1.weight.data += add_weight
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_size, hidden_size, dtype=torch.float)
        self.fc2.weight.data += add_weight
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=surrogate.fast_sigmoid())

        self.fc3 = nn.Linear(hidden_size, hidden_size, dtype=torch.float)
        self.fc3.weight.data += add_weight
        self.lif3 = snn.Leaky(beta=beta3, spike_grad=surrogate.fast_sigmoid())

        # Linear readout layer
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        # Convert observation to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        # Determine if input is batched or not
        is_batched = x.dim() == 3  # [batch_size, num_steps, input_size] is 3D

        if not is_batched:
            # If not batched, add a batch dimension
            x = x.unsqueeze(0)  # Shape becomes [1, num_steps, input_size]


        batch_size = x.size(0)  # This is 1 if not batched, otherwise the actual batch size

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record the spikes from the last layer
        spk3_rec = []
        mem3_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        # Shape: [batch_size, num_steps, output_size]
        spk3_stacked = torch.stack(spk3_rec, dim=1)
        mem3_stacked = torch.stack(mem3_rec, dim=1)

        # Take the average membrane potential across time (summarize spikes)
        # Shape: [batch_size, output_size]
        avg_spk3 = torch.mean(spk3_stacked, dim=1)
        avg_mem3 = torch.mean(mem3_stacked, dim=1)

        # Apply the linear readout layer to the average membrane potential
        # Shape: [batch_size, output_size]
        readout_output_spk = self.readout(avg_spk3)
        readout_output_mem = self.readout(avg_mem3)

        if not is_batched:
            # Remove the batch dimension if it was added
            readout_output_spk = readout_output_spk.squeeze(0)  # Shape becomes [output_size]
            readout_output_mem = readout_output_mem.squeeze(0)  # Shape becomes [output_size]

            spk3_stacked = spk3_stacked.squeeze(0)  # Shape becomes [num_steps, output_size]



        return readout_output_spk, spk3_stacked