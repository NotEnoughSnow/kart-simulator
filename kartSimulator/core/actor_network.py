"""
    This file contains a neural network module for us to
    define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ActorNetwork(nn.Module):
    """
         in_dim-128-128-out_dim Feed Forward Neural Network.
    """

    def __init__(self, in_dim, out_dim, log_std_min=-20, log_std_max=2):
        """
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int
                log_std_min - minimum value for log standard deviation
                log_std_max - maximum value for log standard deviation

            Return:
                None
        """
        super(ActorNetwork, self).__init__()

        # Define the base layers
        self.layer1 = nn.Linear(in_dim, 128)  # Increased layer size
        self.layer2 = nn.Linear(128, 128)     # Increased layer size

        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(128, out_dim)
        self.log_std_layer = nn.Linear(128, out_dim)

        # Store the range for log_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        """
            Runs a forward pass on the neural network.

            Parameters:
                obs - observation to pass as input

            Return:
                mean - the mean action values
                log_std - the log standard deviations
        """

        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        # Base network forward pass
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))

        # Output mean action values
        mean = self.mean_layer(activation2)

        # Output log standard deviations and clamp for stability
        log_std = self.log_std_layer(activation2)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std
