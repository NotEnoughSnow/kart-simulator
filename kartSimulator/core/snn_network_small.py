import snntorch as snn
from snntorch import spikegen
import torch, torch.nn as nn
from snntorch import surrogate

import numpy as np

hidden_size = 64  # Number of hidden neurons

class SNN_small(nn.Module):
    def __init__(self, input_size, output_size, num_steps):
        super(SNN_small, self).__init__()

        self.num_steps = num_steps
        beta1 = 0.9
        beta2 = torch.rand((output_size), dtype=torch.float)  # Independent decay rate for each output neuron

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float)
        self.fc1.weight.data += 0.005
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float)
        self.fc2.weight.data += 0.005
        self.lif2 = snn.Leaky(beta=beta2, learn_beta=True, spike_grad=surrogate.fast_sigmoid())

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

        # Record the spikes from the last layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            ###

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)


        output_spk = torch.stack(spk2_rec, dim=1)  # Shape: [batch_size, num_steps, output_size]
        output_mem = torch.stack(mem2_rec, dim=1)  # Shape: [batch_size, num_steps, output_size]

        if not is_batched:
            # Remove the batch dimension if it was added
            output_spk = output_spk.squeeze(0)  # Shape becomes [num_steps, output_size]
            output_mem = output_mem.squeeze(0)  # Shape becomes [num_steps, output_size]

        #print("should not be none :", output_spk.grad_fn)  # This should not be None

        return output_spk, output_mem


    # TODO compare weight changes for every step before and after, layer 1
    # TODO average output spike time, ratio of output spikes that spike at least once
    # TODO linear readout layer, all spike trains => feed the spike trains through a FF network
    # TODO sum over a sequence : will be the logits
    # linear readout layer