import snntorch as snn
from snntorch import spikegen
import torch, torch.nn as nn
from snntorch import surrogate

import numpy as np

hidden_size = 64  # Number of hidden neurons

import torch.nn.init as init


class SNN_small(nn.Module):
    def __init__(self, input_size, output_size, num_steps, add_weight):
        super(SNN_small, self).__init__()

        self.num_steps = num_steps
        beta1 = 0.9
        beta2 = 0.9
        #beta2 = torch.rand((output_size), dtype=torch.float)  # Independent decay rate for each output neuron

        self.pre_fc = nn.Linear(input_size, hidden_size, dtype=torch.float)
        init.kaiming_uniform_(self.pre_fc.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.pre_fc.bias is not None:
            init.zeros_(self.pre_fc.bias)

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float)
        #self.fc1.weight.data += add_weight
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=surrogate.fast_sigmoid(), threshold=0.7)

        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(hidden_size, hidden_size, dtype=torch.float)
        #self.fc2.weight.data += add_weight
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=surrogate.fast_sigmoid(), threshold=0.7)

        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)


        # Linear readout layer
        self.readout = nn.Linear(hidden_size, output_size)
        init.kaiming_uniform_(self.readout.weight, a=0, mode='fan_in', nonlinearity='linear')
        if self.readout.bias is not None:
            init.zeros_(self.readout.bias)

    def forward(self, x):

        # Convert observation to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        # Determine if input is batched or not
        is_batched = x.dim() == 2  # [batch_size, input_size] is 3D

        if not is_batched:
            # If not batched, add a batch dimension
            x = x.unsqueeze(0)  # Shape becomes [1, input_size]


        batch_size = x.size(0)  # This is 1 if not batched, otherwise the actual batch size

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the spikes from the last layer
        spk2_rec = []
        mem2_rec = []

        x_proj = self.pre_fc(x)  # [batch_size, hidden_dim]
        x_proj = x_proj.unsqueeze(1).repeat(1, self.num_steps, 1)  # [batch_size, num_steps, hidden_dim]

        for step in range(self.num_steps):
            cur1 = self.fc1(x_proj[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        # Shape: [batch_size, num_steps, output_size]
        spk2_stacked = torch.stack(spk2_rec, dim=1)
        mem2_stacked = torch.stack(mem2_rec, dim=1)

        # Take the average membrane potential across time (summarize spikes)
        # Shape: [batch_size, output_size]
        avg_spk2 = torch.mean(spk2_stacked, dim=1)
        avg_mem2 = torch.mean(mem2_stacked, dim=1)

        # Apply the linear readout layer to the average membrane potential
        # Shape: [batch_size, output_size]
        readout_output_spk = self.readout(avg_spk2)
        readout_output_mem = self.readout(avg_mem2)

        if not is_batched:
            # Remove the batch dimension if it was added
            readout_output_spk = readout_output_spk.squeeze(0)  # Shape becomes [output_size]
            readout_output_mem = readout_output_mem.squeeze(0)  # Shape becomes [output_size]

            spk2_stacked = spk2_stacked.squeeze(0)  # Shape becomes [num_steps, output_size]



        return readout_output_spk, spk2_stacked


    # TODO compare weight changes for every step before and after, layer 1
    # TODO average output spike time, ratio of output spikes that spike at least once
    # TODO linear readout layer, all spike trains => feed the spike trains through a FF network
    # TODO sum over a sequence : will be the logits
    # linear readout layer