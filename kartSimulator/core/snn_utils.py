import snntorch as snn
from snntorch import spikegen
import torch


def decode_first_spike_batched(spike_trains):
    """
    Decodes the first spike time from spike trains for batched data using 'time to first spike' method.

    Parameters:
        spike_trains - The batched spike trains with shape (batch_size, num_steps, num_neurons).

    Returns:
        decoded_vector - The decoded first spike times with shape (batch_size, num_neurons).
    """
    batch_size = spike_trains.size(0)
    num_neurons = spike_trains.size(2)
    decoded_vectors = []

    for batch_idx in range(batch_size):
        decoded_vector = [spike_trains.size(1) + 1] * num_neurons

        for neuron_idx in range(num_neurons):
            first_spike = (spike_trains[batch_idx, :, neuron_idx] == 1).nonzero(as_tuple=True)[0]
            if first_spike.nelement() != 0:
                decoded_vector[neuron_idx] = first_spike[0].item() + 1

        decoded_vectors.append(decoded_vector)

    return torch.FloatTensor(decoded_vectors)


def decode_from_spikes_count(spikes):
    spike_counts = torch.sum(spikes, dim=1)
    action = torch.zeros(spikes.size(0))
    max_spike_count = torch.max(spike_counts)
    candidates = torch.where(spike_counts == max_spike_count)[0]
    if len(candidates) > 1:
        action[torch.multinomial(candidates.float(), 1)] = 1
    else:
        action[candidates] = 1
    return action


def encode_to_spikes(data, num_steps):
    """
    Encodes analog signals into spike trains using rate encoding.

    Parameters:
        data - The continuous-valued data to be encoded.
        num_steps - The number of time steps for the spike train.

    Returns:
        spike_train - The encoded spike train.
    """

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6

    # Normalize the data to be between 0 and 1
    normalized_data = (data - data.min()) / (data.max() - data.min() + epsilon)

    normalized_data = torch.clamp(normalized_data, 0, 1)

    # Convert normalized data to spike trains
    # TODO rate vs latency vs delta
    spike_train = spikegen.rate(normalized_data, num_steps=num_steps)

    return spike_train


def encode_to_spikes_batched(data, num_steps):
    """
    Encodes analog signals into spike trains using rate encoding.

    Parameters:
        data - The continuous-valued data to be encoded.
        num_steps - The number of time steps for the spike train.

    Returns:
        spike_train - The encoded spike train.
    """

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6

    data_min = data.min(dim=1, keepdim=True)[0]
    data_max = data.max(dim=1, keepdim=True)[0]

    normalized_data = (data - data_min) / (data_max - data_min + epsilon)
    normalized_data = torch.clamp(normalized_data, 0, 1)

    spike_train = spikegen.rate(normalized_data, num_steps=num_steps)

    return spike_train.transpose(0, 1)