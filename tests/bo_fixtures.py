"""CPU coverage of the BO orchestration contract without the native extension."""

from itertools import product
import hashlib
import json
import sys
from dataclasses import FrozenInstanceError, asdict
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("safetensors")

from ops.flame import bo, model
from ops.flame.config import ITERATION, MODEL_ID, REVISION, Config
from ops.flame.layout import Layout


@pytest.fixture(autouse=True)
def cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def bf16(value):
    return jnp.asarray(value, dtype=jnp.bfloat16)


class FakeProposals:
    def __init__(
        self,
        flat,
        changes,
        *,
        seed=17,
        score=0.5,
        radius=0.01,
        candidate_index=2,
        persistence=0.0,
    ):
        self.array = flat.reshape((1, flat.size))
        self.description = (seed, score, radius, changes)
        self.selected_geometry = [(candidate_index, persistence)]
        self.exports = 0

    def describe(self):
        return [self.description]

    def geometry(self):
        return self.selected_geometry

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()

    def __dlpack__(self, *args, **kwargs):
        self.exports += 1
        return self.array.__dlpack__(*args, **kwargs)


class FakeSearchState:
    def __init__(self, flat, proposals, *, best=-10.0, history=2, sampler="correlated"):
        self.best_array = flat
        self.best = best
        self.length = 0.01
        self.restarts = 0
        self.history_len = 1
        self.history = history
        self.sampler = sampler
        self.proposals = iter(proposals)
        self.calls = []
        self.readbacks = 0

    def ask(self, *args, **kwargs):
        self.calls.append(("ask", args, kwargs))
        self.pending = next(self.proposals)
        return self.pending

    def tell(self, proposals, rewards, noise):
        assert proposals is self.pending
        self.calls.append(("tell", proposals, rewards, noise))
        self.reward = rewards[0]

    def sync(self):
        self.calls.append(("sync",))
        accepted = self.reward > self.best
        if accepted:
            self.best = self.reward
            self.best_array = self.pending.array.reshape(-1)
            if self.sampler == "correlated":
                self.length = self.pending.description[2]
        self.history_len = min(self.history, self.history_len + 1)
        return [accepted]

    def read_best(self):
        self.readbacks += 1
        return np.array(jax.lax.bitcast_convert_type(self.best_array, jnp.uint16))


class FakePairedSearch(FakeSearchState):
    def enable_relative(self, *, failure_tolerance):
        assert not self.calls
        self.calls.append(("enable_relative", failure_tolerance))
        self.failure_tolerance = failure_tolerance
        self.failures = 0

    def incumbent(self):
        self.calls.append(("incumbent",))
        return jnp.array(self.best_array, copy=True).reshape((1, -1))

    def tell(self, *args):
        pytest.fail("minibatch observations must not use legacy acceptance")

    def tell_relative(
        self,
        proposals,
        value,
        variance,
        incumbent_value,
        incumbent_variance,
        improvement,
        improvement_variance,
        accept,
        *,
        reject_is_failure=True,
    ):
        assert proposals is self.pending
        self.calls.append(
            (
                "tell_relative",
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                improvement,
                improvement_variance,
                accept,
                reject_is_failure,
            )
        )
        self.decision = accept
        self.reject_is_failure = reject_is_failure
        self.best = value if accept else incumbent_value

    def sync(self):
        self.calls.append(("sync",))
        if self.decision:
            self.best_array = self.pending.array.reshape(-1)
            self.length = self.pending.description[2]
            self.history_len = 1
            self.failures = 0
            return [self.decision]
        self.history_len = min(self.history, self.history_len + 1)
        if self.reject_is_failure:
            self.failures += 1
        if self.reject_is_failure and self.failures == self.failure_tolerance:
            self.length = max(0.0001, self.length / 2)
            self.failures = 0
        return [self.decision]


class FakeProblemEvaluator:
    def __init__(self, losses, population=8):
        self.tokens = [None] * population
        self.values = iter(losses)
        self.batches = []

    def losses(self, flat, indices):
        assert flat.ndim == 2 and flat.shape[0] == 1
        self.batches.append(indices.tolist())
        return np.asarray(next(self.values), dtype=np.float64)


def minibatch_baseline(settings, population=8):
    indices = bo.sample_indices(
        bo.minibatch_rng(settings.minibatch_seed), population, settings.minibatch_size
    )
    return {
        "indices": indices.tolist(),
        "losses": [0.1, 0.1],
        "variance": 0.0,
        "mean_loss": 0.1,
    }


@pytest.fixture
def layout():
    return Layout.from_params({"a": bf16([1, 1]), "b": bf16(2)})


@pytest.fixture
def flat(layout):
    return layout.flatten({"a": bf16([1, 1]), "b": bf16(2)})


@pytest.fixture
def tiny_model():
    config = Config(
        layers=1,
        width=4,
        heads=1,
        vocab=8,
        dense_width=4,
        expert_width=2,
        shared_width=2,
        experts=2,
        top_k=1,
        context=4,
    )
    rng = np.random.default_rng(17)
    params = {
        name: bf16(rng.normal(0, 0.2, shape)) for name, shape in config.shapes().items()
    }
    return config, params
