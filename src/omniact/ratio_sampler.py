"""DistributedSampler with optional per-epoch length cap.

Used by trainers to control the sampling ratio between datasets in
multi-embodiment training: pass ``sampled_length=N`` in the ``--dataset``
spec flags to make a dataset contribute exactly ``N`` samples per epoch,
regardless of its real length.

Behavior:
  * ``sampled_length=None`` is byte-identical to ``DistributedSampler``.
  * ``sampled_length == len(dataset)`` is also byte-identical: same seed,
    same shuffle, same DistributedSampler-style padding for non-divisible
    cases.  No way to distinguish.
  * ``sampled_length < len(dataset)`` uses one shuffle (same seed as
    default), takes the first ``sampled_length`` indices, then applies
    DistributedSampler's pad-or-truncate strategy.
  * ``sampled_length > len(dataset)`` cycles through fresh reshuffles to
    fill ``total_size``.  Cycle 0 reuses the default seed so the first
    ``len(dataset)`` indices match the default; cycles 1+ use sub-seeds
    derived from ``(seed, epoch, cycle)`` so resumption stays
    deterministic.
"""

import math

import torch
from torch.utils.data.distributed import DistributedSampler


# Knuth-multiplier constant; mutually prime with anything used as epoch /
# cycle, so seed offsets across (epoch, cycle) don't collide.
_CYCLE_SEED_STRIDE = 2654435761


class RatioSampler(DistributedSampler):
    """DistributedSampler with optional per-epoch ``sampled_length`` cap."""

    def __init__(
        self,
        dataset,
        sampled_length: int | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.full_length = len(dataset)
        self.sampled_length = sampled_length
        if sampled_length is not None:
            if sampled_length <= 0:
                raise ValueError(f"sampled_length must be positive, got {sampled_length}")
            if self.full_length == 0:
                raise ValueError(
                    "sampled_length cannot be set on an empty dataset (full_length=0)"
                )
            if drop_last:
                self.num_samples = sampled_length // self.num_replicas
            else:
                self.num_samples = math.ceil(sampled_length / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.sampled_length is None:
            return super().__iter__()

        # Cycle 0: identical seed to DistributedSampler so the default
        # behavior is recoverable when sampled_length == full_length.
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.full_length, generator=g).tolist()
        else:
            indices = list(range(self.full_length))

        if self.sampled_length <= self.full_length:
            # Truncate to the cap first.  When sampled_length == full_length
            # this slice is a no-op and the rest matches DistributedSampler
            # exactly.
            indices = indices[: self.sampled_length]
            # DistributedSampler-style pad-or-truncate to total_size.
            if not self.drop_last:
                padding_size = self.total_size - len(indices)
                if padding_size > 0:
                    if padding_size <= len(indices):
                        indices += indices[:padding_size]
                    else:
                        indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                indices = indices[: self.total_size]
        else:
            # sampled_length > full_length: append fresh reshuffles for the
            # rest.  Sub-seeds keep resumption deterministic.
            cycle = 1
            while len(indices) < self.total_size:
                if self.shuffle:
                    g = torch.Generator()
                    g.manual_seed(self.seed + self.epoch + cycle * _CYCLE_SEED_STRIDE)
                    indices.extend(torch.randperm(self.full_length, generator=g).tolist())
                else:
                    indices.extend(range(self.full_length))
                cycle += 1
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples
