# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.get_sampler import get_sampler

from ennx import ENN
from ennx.botorch import Model, Sampler
from ennx.ennx.enn_params import ENNParams, PosteriorFlags


@pytest.fixture
def model():
    x = np.random.default_rng(12).normal(size=(16, 2))
    y = np.column_stack((np.sin(x[:, 0]), np.cos(x[:, 1])))
    return Model(ENN(x, y, np.full_like(y, 0.01)), ENNParams(4, 1.0, 1.0))


def test_joint(model):
    x = torch.tensor([[0.1, 0.2], [0.1, 0.2], [0.3, 0.4]], dtype=torch.double)
    sampler = Sampler(torch.Size([2, 3]), seed=13)
    actual = sampler(model.posterior(x))
    expected, _ = model.enn.posterior_draw(
        x.numpy(), model.params, function_seeds=sampler.base_samples.flatten().numpy()
    )
    np.testing.assert_array_equal(
        actual, np.moveaxis(expected, -1, 0).reshape(2, 3, 3, 2)
    )
    assert torch.equal(actual[..., 0, :], actual[..., 1, :])
    assert torch.equal(
        sampler(model.posterior(x[[2, 0, 1]])), actual[..., [2, 0, 1], :]
    )
    assert torch.equal(
        sampler(model.posterior(x))[..., :2, :], sampler(model.posterior(x[:2]))
    )


@pytest.mark.parametrize("noise", [False, True])
def test_moments(model, noise):
    x = torch.arange(24, dtype=torch.double).reshape(2, 2, 3, 2) / 24
    posterior = model.posterior(x, output_indices=[1, 0], observation_noise=noise)
    native = model.enn.posterior(
        x.reshape(-1, 2).numpy(),
        params=model.params,
        flags=PosteriorFlags(observation_noise=noise),
    )
    np.testing.assert_array_equal(
        posterior.mean, native.mu[:, [1, 0]].reshape(2, 2, 3, 2)
    )
    np.testing.assert_array_equal(
        posterior.variance, (native.se[:, [1, 0]] ** 2).reshape(2, 2, 3, 2)
    )
    sampler = get_sampler(posterior, torch.Size([5]), seed=10)
    actual = sampler(posterior)
    assert actual.shape == (5, 2, 2, 3, 2)
    native, _ = model.enn.posterior_draw(
        x.reshape(-1, 2).numpy(),
        model.params,
        function_seeds=sampler.base_samples.numpy(),
        flags=PosteriorFlags(observation_noise=noise),
    )
    np.testing.assert_array_equal(
        actual, np.moveaxis(native[:, [1, 0]], -1, 0).reshape(actual.shape)
    )
    assert posterior.rsample(torch.Size()).shape == posterior.mean.shape
    assert posterior.rsample(torch.Size([0])).shape == (0, 2, 2, 3, 2)


def test_batching(model):
    train_x, train_y, _ = model.enn.train_rows(range(16))
    scalar = Model(ENN(train_x, train_y[:, :1]), model.params)
    x = torch.linspace(0, 1, 12, dtype=torch.double).reshape(3, 2, 2)
    sampler = Sampler(torch.Size([32]), seed=14)
    samples = sampler(scalar.posterior(x))
    assert torch.equal(
        samples,
        torch.cat([sampler(scalar.posterior(batch[None])) for batch in x], dim=1),
    )


def test_acquisitions(model):
    train_x, train_y, _ = model.enn.train_rows(range(16))
    scalar = Model(ENN(train_x, train_y[:, :1]), model.params)
    x = torch.linspace(0, 1, 12, dtype=torch.double).reshape(3, 2, 2)
    sampler = Sampler(torch.Size([32]), seed=14)
    acquisitions = [
        qExpectedImprovement(scalar, best_f=0.0, sampler=sampler),
        qLogExpectedImprovement(scalar, best_f=0.0, sampler=sampler),
        qProbabilityOfImprovement(scalar, best_f=0.0, sampler=sampler),
        qUpperConfidenceBound(scalar, beta=0.2, sampler=sampler),
        qSimpleRegret(scalar, sampler=sampler),
        qLogNoisyExpectedImprovement(
            scalar,
            X_baseline=x[0],
            sampler=sampler,
            prune_baseline=False,
            cache_root=False,
        ),
    ]
    for acquisition in acquisitions:
        actual = acquisition(x)
        assert actual.shape == (3,)
        assert torch.isfinite(actual).all()
        assert torch.equal(actual, acquisition(x))
        # Native samples are bitwise identical above. PyTorch reductions can
        # change their summation order with tensor shape; do not claim otherwise.
        separate = torch.cat([acquisition(batch[None]) for batch in x])
        torch.testing.assert_close(
            actual, separate, rtol=0, atol=32 * torch.finfo(actual.dtype).eps
        )
    draws = sampler(scalar.posterior(x)).squeeze(-1)
    expected = draws.clamp_min(0).amax(-1).mean(0)
    assert torch.equal(acquisitions[0](x), expected)


def test_selection(model):
    train_x, train_y, _ = model.enn.train_rows(range(16))
    scalar = Model(ENN(train_x, train_y[:, :1]), model.params)
    x = torch.linspace(0, 1, 12, dtype=torch.double).reshape(6, 2)
    acquisition = qLogExpectedImprovement(
        scalar, best_f=0.0, sampler=Sampler(torch.Size([32]), seed=14)
    )
    selected, _ = optimize_acqf_discrete(acquisition, q=2, choices=x)
    assert selected.shape == (2, 2)
    assert all(
        any(torch.equal(point, candidate) for candidate in x.reshape(-1, 2))
        for point in selected
    )


def test_rejections(model):
    x = torch.zeros(2, 2, dtype=torch.double)
    with pytest.raises(NotImplementedError, match="gradients"):
        model.posterior(x.clone().requires_grad_())
    with pytest.raises(NotImplementedError, match="boolean"):
        model.posterior(x, observation_noise=torch.ones(2, 2))
    with pytest.raises(NotImplementedError, match="transform"):
        model.posterior(x, posterior_transform=object())
    for bad in [torch.zeros(2), torch.zeros(2, 3), x.long(), x + float("nan")]:
        with pytest.raises(ValueError):
            model.posterior(bad)
    for indices in [[], [0, 0], [-1], [2]]:
        with pytest.raises(ValueError, match="indices"):
            model.posterior(x, output_indices=indices)
    posterior = model.posterior(x)
    with pytest.raises(ValueError, match="int64"):
        posterior.rsample_from_base_samples(torch.Size([2]), torch.zeros(2))
    model.enn.add(np.ones((1, 2)), np.ones((1, 2)), np.full((1, 2), 0.01))
    with pytest.raises(RuntimeError, match="fresh posterior"):
        posterior.rsample()


def test_rng(model):
    x = torch.ones(3, 2, dtype=torch.float32)
    before = torch.random.get_rng_state()
    sampler = Sampler(torch.Size([4]), seed=17)
    values = sampler(model.posterior(x))
    assert torch.equal(before, torch.random.get_rng_state())
    assert values.dtype == torch.float32
    assert torch.equal(values, Sampler(torch.Size([4]), seed=17)(model.posterior(x)))
