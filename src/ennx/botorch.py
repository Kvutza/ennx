"""BoTorch integration using ENNX's native, correlated function draws.

This is a CPU tensor boundary without autograd;
evaluate acquisitions on candidate sets instead of gradient-based optimization.
"""

from __future__ import annotations

from typing import override

import numpy as np
import torch
from botorch.models.model import Model as BaseModel
from botorch.posteriors.posterior import Posterior as BasePosterior
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import GetSampler

from .ennx.enn_class import ENN
from .ennx.enn_params import ENNParams, PosteriorFlags


class Model(BaseModel):
    """Expose an existing ENN to BoTorch without replacing its distribution.

    Supports unbatched training data, arbitrary query batches, multiple outputs,
    and boolean observation noise. Bounded outputs, posterior transforms,
    conditioning/fantasies, and input gradients are deliberately unsupported.
    """

    def __init__(self, enn: ENN, params: ENNParams) -> None:
        super().__init__()
        if np.isfinite(enn.y_bounds).any():
            raise ValueError("bounded outputs do not expose exact posterior moments")
        self.enn = enn
        self.params = params

    @property
    def num_outputs(self):
        return self.enn.num_outputs

    @property
    def batch_shape(self):
        return torch.Size()

    @override
    def posterior(
        self, X, output_indices=None, observation_noise=False, posterior_transform=None
    ):
        if X.device.type != "cpu" or X.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "ENNX's BoTorch boundary requires CPU float32/float64 tensors"
            )
        if X.requires_grad:
            raise NotImplementedError(
                "ENNX does not provide input gradients to BoTorch"
            )
        if X.ndim < 2 or X.shape[-1] != self.enn._numdim or X.numel() == 0:
            raise ValueError("queries must have nonempty shape (..., q, dimensions)")
        if not torch.isfinite(X).all():
            raise ValueError("queries must be finite")
        if not isinstance(observation_noise, bool):
            raise NotImplementedError("only boolean observation_noise is supported")
        if posterior_transform is not None:
            raise NotImplementedError(
                "use an MC objective instead of a posterior transform"
            )
        indices = (
            list(range(self.num_outputs))
            if output_indices is None
            else list(output_indices)
        )
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(
                not isinstance(i, int) or i < 0 or i >= self.num_outputs
                for i in indices
            )
        ):
            raise ValueError("output indices must be distinct valid output positions")
        return Posterior(self.enn, self.params, X, indices, observation_noise)


class Posterior(BasePosterior):
    """Native marginals and joint draws with one function seed per MC sample."""

    def __init__(self, enn, params, queries, indices, noise):
        self._enn = enn
        self._params = params
        self._indices = indices
        self._flags = PosteriorFlags(observation_noise=noise)
        self._rows = len(enn)
        self._dtype = queries.dtype
        self._shape = queries.shape[:-1] + torch.Size([len(indices)])
        self._queries = (
            queries.detach().double().numpy().reshape(-1, queries.shape[-1]).copy()
        )
        marginal = enn.posterior(self._queries, params=params, flags=self._flags)
        self._mean = self._tensor(marginal.mu[:, indices])
        self._variance = self._tensor(marginal.se[:, indices] ** 2)

    def _check(self):
        if len(self._enn) != self._rows:
            raise RuntimeError("ENNX changed; request a fresh posterior")

    def _tensor(self, values):
        return torch.as_tensor(np.ascontiguousarray(values), dtype=self.dtype).reshape(
            self._shape
        )

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return self._dtype

    @property
    def mean(self):
        self._check()
        return self._mean

    @property
    def variance(self):
        self._check()
        return self._variance

    @override
    def _extended_shape(self, sample_shape=torch.Size()):
        return sample_shape + self._shape

    @override
    def rsample(self, sample_shape=None):
        shape = torch.Size([1]) if sample_shape is None else sample_shape
        seeds = torch.randint(0, torch.iinfo(torch.int64).max, shape, device="cpu")
        return self.rsample_from_base_samples(shape, seeds)

    @override
    def rsample_from_base_samples(self, sample_shape, base_samples):
        self._check()
        if base_samples.dtype != torch.int64 or base_samples.device.type != "cpu":
            raise ValueError("ENNX base samples must be CPU int64 function seeds")
        if base_samples.shape != sample_shape:
            raise ValueError("ENNX requires exactly one function seed per MC sample")
        if sample_shape.numel() == 0:
            return torch.empty(self._extended_shape(sample_shape), dtype=self.dtype)
        draws, _ = self._enn.posterior_draw(
            self._queries,
            self._params,
            function_seeds=base_samples.reshape(-1).numpy(),
            flags=self._flags,
        )
        values = np.moveaxis(draws[:, self._indices, :], -1, 0)
        return torch.as_tensor(np.ascontiguousarray(values), dtype=self.dtype).reshape(
            self._extended_shape(sample_shape)
        )


class Sampler(MCSampler):
    """Reuse native function seeds across candidates and acquisition evaluations."""

    def __init__(self, sample_shape: torch.Size, seed: int | None = None):
        super().__init__(sample_shape, seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.base_samples = torch.randint(
            0,
            torch.iinfo(torch.int64).max,
            sample_shape,
            generator=generator,
            device="cpu",
        )

    @override
    def forward(self, posterior):
        return posterior.rsample_from_base_samples(self.sample_shape, self.base_samples)

    @override
    def _update_base_samples(self, posterior, base_sampler):
        self._instance_check(base_sampler)
        if self.sample_shape != base_sampler.sample_shape:
            raise ValueError("sampler shapes must match")
        self.base_samples = base_sampler.base_samples.clone()


@GetSampler.register(Posterior)
def _sampler(posterior, sample_shape, seed=None):
    return Sampler(sample_shape, seed)


__all__ = ["Model", "Posterior", "Sampler"]
