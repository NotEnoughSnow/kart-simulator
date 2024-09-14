import torch
import snntorch as snn
import snntorch.functional as SF
import snntorch.spikegen as spikegen
import numpy as np


def generate_spike_trains(observation, num_steps, threshold, shift):
    """
    Generate spike trains from a single observation using a fixed global threshold.

    Parameters:
    - observation: A tensor representing the observation ([observation_dim]).
    - num_steps: The number of timesteps for the spike train.
    - threshold: A single global threshold value to be used for normalization.

    Returns:
    - spike_trains: Tensor of spike trains.
    """

    # Normalize and clip observation
    shifted_obs = np.add(observation, shift)

    # torch version
    # shifted_obs = observation + shift

    normalized_obs = shifted_obs / (threshold + 1e-6)  # Avoid division by zero

    normalized_obs /= 2

    normalized_obs = normalized_obs.clamp(0, 1)  # Clip values to be within [0, 1]

    # Generate spike trains
    spike_trains = spikegen.rate(normalized_obs, num_steps=num_steps)

    # torch version
    # return spike_trains

    return spike_trains.numpy()


def generate_spike_trains_batched(observations, num_steps, threshold, shift):
    """
    Generate spike trains from batched observations using a fixed global threshold.

    Parameters:
    - observations: A tensor representing the batched observations ([batch_size, observation_dim]).
    - num_steps: The number of timesteps for the spike train.
    - threshold: A single global threshold value to be used for normalization.
    - shift: A value to shift the observation range to handle negative values.

    Returns:
    - spike_trains: Tensor of spike trains with shape (batch_size, num_steps, observation_dim).
    """

    shift = shift.numpy()

    # Normalize and shift observations
    normalized_obs = np.add(observations, shift) / (2 * (threshold + 1e-6))  # Avoid division by zero
    normalized_obs = normalized_obs.clamp(0, 1)  # Clip values to [0, 1]

    # Generate spike trains for each observation in the batch
    spike_trains = spikegen.rate(normalized_obs, num_steps=num_steps)

    # Rearrange the output to have shape (batch_size, num_steps, observation_dim)
    spike_trains = spike_trains.permute(1, 0, 2)

    # torch version
    # return spike_trains

    return spike_trains.numpy()


def get_spike_counts(spike_trains):
    """
    Get the total number of spikes for each neuron over all timesteps.

    Parameters:
    - spike_trains: Tensor of spike trains with shape [num_steps, observation_dim].

    Returns:
    - Array of spike counts for each neuron.
    """

    num_steps, num_neurons = spike_trains.shape

    spike_counts = torch.sum(spike_trains, dim=0)

    spike_counts = spike_counts / num_steps

    return spike_counts


def get_spike_counts_batched(spike_trains):
    """
    Get the total number of spikes for each neuron over all timesteps for batched spike trains.

    Parameters:
    - spike_trains: Tensor of spike trains with shape [batch_size, num_steps, observation_dim].

    Returns:
    - Array of spike counts for each neuron in each observation (shape: [batch_size, observation_dim]).
    """
    batch_size, num_steps, num_neurons = spike_trains.shape

    # Sum over the time dimension (dim=1) to get spike counts for each neuron in each observation
    spike_counts = torch.sum(spike_trains, dim=1)

    spike_counts = spike_counts / num_steps

    return spike_counts


def decode_first_spike_batched(spike_trains):
    """
    Decodes the first spike time from batched spike trains using the 'time to first spike' method.

    Parameters:
        spike_trains - The batched spike trains with shape (batch_size, num_steps, num_neurons).

    Returns:
        decoded_vector - A tensor representing the first spike times for each neuron in each batch with gradients retained.
    """
    batch_size, num_steps, num_neurons = spike_trains.shape

    # Create a tensor with time steps and retain gradients
    time_tensor = torch.arange(1, num_steps + 1, dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(
        2).expand(batch_size, num_steps, num_neurons)

    # Multiply spike_trains by the time tensor, masking out non-spike entries
    spike_times = spike_trains * time_tensor

    # Set all zero entries (no spike) to a very high value (greater than num_steps)
    spike_times = spike_times + (1 - spike_trains) * (num_steps + 1)

    # Find the minimum value in each column (i.e., first spike) for each batch
    first_spike_times, _ = spike_times.min(dim=1)

    # Transform the spike times into a format better suited for a categorical
    first_spike_times = (-2 / (num_steps + 1)) * first_spike_times + 2

    # Ensure that this tensor retains gradients
    return first_spike_times


def decode_first_spike(spike_trains):
    """
    Decodes the first spike time from spike trains using the 'time to first spike' method.

    Parameters:
        spike_trains - The spike trains with shape (num_steps, num_neurons).

    Returns:
        decoded_vector - A tensor representing the first spike times for each neuron with gradients retained.
    """
    num_steps, num_neurons = spike_trains.shape

    # Create a tensor with time steps and retain gradients
    time_tensor = torch.arange(1, num_steps + 1, dtype=torch.float32, requires_grad=True).unsqueeze(1).expand(num_steps,
                                                                                                              num_neurons)

    # Multiply spike_trains by the time tensor, masking out non-spike entries
    spike_times = spike_trains * time_tensor

    # Set all zero entries (no spike) to a very high value (greater than num_steps)
    spike_times = spike_times + (1 - spike_trains) * (num_steps + 1)

    # Find the minimum value in each column (i.e., first spike)
    first_spike_times, _ = spike_times.min(dim=0)

    # Transform the spike times into a format better suited for a categorical
    first_spike_times = (-2 / (num_steps + 1)) * first_spike_times + 2

    # Ensure that this tensor retains gradients
    return first_spike_times

def get_first_spike_batched(spike_trains):
    """
    Decodes the first spike time from batched spike trains using the 'time to first spike' method.

    Parameters:
        spike_trains - The batched spike trains with shape (batch_size, num_steps, num_neurons).

    Returns:
        decoded_vector - A tensor representing the first spike times for each neuron in each batch with gradients retained.
    """
    batch_size, num_steps, num_neurons = spike_trains.shape

    time_tensor = torch.arange(1, num_steps + 1, dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(
        2).expand(batch_size, num_steps, num_neurons)
    spike_times = spike_trains * time_tensor
    spike_times = spike_times + (1 - spike_trains) * (num_steps + 1)
    first_spike_times, _ = spike_times.min(dim=1)

    return first_spike_times

def get_first_spike(spike_trains):
    """
    Decodes the first spike time from spike trains using the 'time to first spike' method.

    Parameters:
        spike_trains - The spike trains with shape (num_steps, num_neurons).

    Returns:
        decoded_vector - A tensor representing the first spike times for each neuron with gradients retained.
    """
    num_steps, num_neurons = spike_trains.shape

    time_tensor = torch.arange(1, num_steps + 1, dtype=torch.float32, requires_grad=True).unsqueeze(1).expand(num_steps,
                                                                                                              num_neurons)
    spike_times = spike_trains * time_tensor
    spike_times = spike_times + (1 - spike_trains) * (num_steps + 1)
    first_spike_times, _ = spike_times.min(dim=0)

    return first_spike_times

def compute_spike_metrics(spk_output):
    """
    Compute the average spike time and ratio of neurons that spike at least once.

    Handles both batched ([batch_size, num_steps, output_size]) and unbatched ([num_steps, output_size]) outputs.

    Parameters:
        spk_output: Spiking activity output from the actor network.
                    Shape can be either [batch_size, num_steps, output_size] or [num_steps, output_size].

    Returns:
        avg_spike_time: The average time at which spikes occur
        spike_ratio: The ratio of neurons that spike at least once
    """
    if spk_output.dim() == 3:
        # Batched case: [batch_size, num_steps, output_size]
        spike_times = get_first_spike_batched(spk_output)
        avg_spike_time = torch.mean(spike_times)  # Average spike time

        # Calculate the ratio of neurons that spiked at least once per batch
        spike_ratio = (spk_output.sum(dim=1) > 0).float().mean()

    elif spk_output.dim() == 2:
        # Unbatched case: [num_steps, output_size]
        spike_times = get_first_spike(spk_output)
        avg_spike_time = torch.mean(spike_times)  # Average spike time

        # Calculate the ratio of neurons that spiked at least once
        spike_ratio = (spk_output.sum(dim=0) > 0).float().mean()

    else:
        raise ValueError("spk_output must have 2 or 3 dimensions, got shape: {}".format(spk_output.shape))

    return avg_spike_time.detach(), spike_ratio.detach()


