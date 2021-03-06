import torch

from typing import Optional, Union, Iterable, Iterator


def bernoulli(datum: torch.Tensor, time: Optional[int] = None, dt: float = 1.0, **kwargs) -> torch.Tensor:
    # language=rst
    """

    :param datum: Generates Bernoulli-distributed spike trains based on input intensity. Inputs must be non-negative.
                  Spikes correspond to successful Bernoulli trials, with success probability equal to (normalized in
                  [0, 1]) input value.
    :param time: Tensor of shape ``[n_1, ..., n_k]``.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Bernoulli-distributed spikes.

    Keyword arguments:

    :param float max_prob: Maximum probability of spike per Bernoulli trial.
    """
    # Setting kwargs.
    max_prob = kwargs.get('max_prob', 1.0)
    assert 0 <= max_prob <= 1, 'Maximum firing probability must be in range [0, 1]'

    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)
    time = int(time / dt)

    # Normalize inputs and rescale (spike probability proportional to normalized intensity).
    if datum.max() > 1.0:
        datum /= datum.max()

    # Make spike data from Bernoulli sampling.
    if time is None:
        spikes = torch.bernoulli(max_prob * datum)
        spikes = spikes.view(*shape)
    else:
        spikes = torch.bernoulli(max_prob * datum.repeat([time, 1]))
        spikes = spikes.view(time, *shape)

    return spikes.byte()


def bernoulli_loader(data: Union[torch.Tensor, Iterable[torch.Tensor]], time: Optional[int] = None, dt: float = 1.0,
                     **kwargs) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.bernoulli`` to iteratively encode a sequence of data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of Bernoulli spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of Bernoulli-distributed spikes.

    Keyword arguments:

    :param float max_prob: Maximum probability of spike per Bernoulli trial.
    """
    # Setting kwargs.
    max_prob = kwargs.get('max_prob', 1.0)

    for i in range(len(data)):
        yield bernoulli(datum=data[i], time=time, dt=dt, max_prob=max_prob)  # Encode datum as Bernoulli spike trains.


def poisson(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
    # language=rst
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be non-negative. Inter-spike
    intervals (ISIs) for non-negative data incremented by one to avoid zero intervals while maintaining ISI
    distributions.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Bernoulli spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    # Get shape and size of data.
    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)
    time = int(time / dt)

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = torch.zeros(size)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time]))
    intervals[:, datum != 0] += 1

    # Calculate spike times by cumulatively summing over time dimension.
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time] = 0

    # Create tensor of spikes.
    spikes = torch.zeros(time, size).byte()
    spikes[times, torch.arange(size)] = 1
    spikes[0] = 0

    return spikes.view(time, *shape)


def poisson_loader(data: Union[torch.Tensor, Iterable[torch.Tensor]], time: int, dt: float = 1.0,
                   **kwargs) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.poisson`` to iteratively encode a sequence of data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    for i in range(len(data)):
        yield poisson(datum=data[i], time=time, dt=dt)  # Encode datum as Poisson spike trains.


def rank_order(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
    # language=rst
    """
    Encodes data via a rank order coding-like representation. One spike per neuron, temporally ordered by decreasing
    intensity. Inputs must be non-negative.

    :param datum: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)
    time = int(time / dt)

    # Create spike times in order of decreasing intensity.
    datum /= datum.max()
    times = torch.zeros(size)
    times[datum != 0] = 1 / datum[datum != 0]
    times *= time / times.max()  # Extended through simulation time.
    times = torch.ceil(times).long()

    print(times.min(), times.max())

    # Create spike times tensor.
    spikes = torch.zeros(time, size).byte()
    for i in range(size):
        if times[i] != 0:
            spikes[times[i] - 1, i] = 1

    return spikes.reshape(time, *shape)


def rank_order_loader(data: Union[torch.Tensor, Iterable[torch.Tensor]], time: int, dt: float = 1.0,
                      **kwargs) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.rank_order`` to iteratively encode a sequence of data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    for i in range(len(data)):
        yield rank_order(datum=data[i], time=time, dt=dt)  # Encode datum as rank order-encoded spike trains.
