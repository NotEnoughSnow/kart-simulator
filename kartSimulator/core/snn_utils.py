import snntorch as snn
from snntorch import spikegen


def encode_to_spikes(data, num_steps):
    """
    Encodes analog signals into spike trains using rate encoding.

    Parameters:
        data - The continuous-valued data to be encoded.
        num_steps - The number of time steps for the spike train.

    Returns:
        spike_train - The encoded spike train.
    """
    # Normalize the data to be between 0 and 1
    normalized_data = (data - data.min()) / (data.max() - data.min())

    # Convert normalized data to spike trains
    # TODO rate vs latency vs delta
    spike_train = spikegen.rate(normalized_data, num_steps=num_steps)

    return spike_train
