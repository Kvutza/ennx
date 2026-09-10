from __future__ import annotations

import numpy as np
import pytest


def test_drawf1dandmultid():
    from ops.stress import DRAW_F_CENTER, draw_f

    x1 = np.array([[DRAW_F_CENTER], [0.5]])
    y1 = draw_f(x1)
    assert y1.shape == (2, 1)
    np.testing.assert_allclose(y1, [[0.0], [0.04]])

    x2 = np.array([[DRAW_F_CENTER, DRAW_F_CENTER], [0.4, 0.2]])
    y2 = draw_f(x2)
    assert y2.shape == (2, 1)
    np.testing.assert_allclose(y2, [[0.0], [0.01 + 0.01]])


def test_001():
    from ops.stress import DRAW_F_CENTER, argmin_rms

    # Two test points in 2D; three samples pick indices 0, 1, 0.
    x_test = np.array([[0.3, 0.3], [0.5, 0.1]], dtype=float)
    # draws (B=2, M=1, S=3): sample0 min at i=0, sample1 at i=1, sample2 at i=0
    draws = np.array(
        [
            [[0.0, 2.0, 0.1]],
            [[1.0, 0.0, 0.2]],
        ],
        dtype=float,
    )
    # residuals: [0,0], [0.2,-0.2], [0,0] vs center 0.3
    # ||eps||^2: 0, 0.08, 0 -> mean 0.08/3 -> rms sqrt(0.08/3)
    expected = float(np.sqrt(0.08 / 3.0))
    assert argmin_rms(x_test, draws) == pytest.approx(expected)
    # sanity: center constant matches DRAW_F_CENTER
    assert DRAW_F_CENTER == pytest.approx(0.3)


def test_002():
    from ops.stress import argmin_rate

    # True f-argmin is index 0 (at center); draws pick 0, 1, 0 -> hit rate 2/3.
    x_test = np.array([[0.3, 0.3], [0.5, 0.1]], dtype=float)
    draws = np.array(
        [
            [[0.0, 2.0, 0.1]],
            [[1.0, 0.0, 0.2]],
        ],
        dtype=float,
    )
    assert argmin_rate(x_test, draws) == pytest.approx(2.0 / 3.0)


def test_003():
    from ops.stress import make_observations2

    rng = np.random.default_rng(7)
    x, y = make_observations2(20, num_dim=3, rng=rng)
    assert x.shape == (20, 3)
    assert y.shape == (20, 1)
    assert np.all(x >= 0.0) and np.all(x <= 1.0)

    x_a, y_a = make_observations2(15, num_dim=2, rng=np.random.default_rng(3))
    x_b, y_b = make_observations2(15, num_dim=2, rng=np.random.default_rng(3))
    np.testing.assert_allclose(x_a, x_b)
    np.testing.assert_allclose(y_a, y_b)


def test_004():
    from ops.stress import average_likelihood, gaussian_likelihood

    y = np.array([[0.0], [1.0]])
    mu = np.array([[0.0], [1.0]])
    se = np.array([[1.0], [1.0]])
    lik = gaussian_likelihood(y, mu, se)
    expected = 1.0 / np.sqrt(2.0 * np.pi)
    np.testing.assert_allclose(lik, [[expected], [expected]])
    assert average_likelihood(y, mu, se) == pytest.approx(expected)


def test_005():
    from ops.stress import average_draws

    y = np.array([[0.0], [1.0]])
    # (batch, metrics, num_samples)
    draws = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0]],
        ],
        dtype=float,
    )
    # empirical se is 0 -> floored; density is large but finite
    avg = average_draws(y, draws)
    assert np.isfinite(avg)
    assert avg > 0.0


def test_006():
    from ops.stress import DrawStressConfig, run_stress2

    result = run_stress2(
        DrawStressConfig(
            num_obs=40,
            num_test=20,
            num_dim=2,
            seed=0,
            k=5,
            num_candidates=8,
            num_samples=5,
            num_draws=4,
        )
    )
    assert np.isfinite(result.posterior.avg_likelihood)
    assert np.isfinite(result.posterior_draw.avg_likelihood)
    assert np.isfinite(result.posterior.argmin_rms)
    assert np.isfinite(result.posterior_draw.argmin_rms)
    assert result.posterior.argmin_rms >= 0.0
    assert result.posterior_draw.argmin_rms >= 0.0
    assert 0.0 <= result.posterior.argmin_rate <= 1.0
    assert 0.0 <= result.posterior_draw.argmin_rate <= 1.0
    assert result.posterior.method == "posterior"
    assert result.posterior_draw.method == "posterior_draw"
    assert result.posterior.all_finite
    assert result.posterior_draw.all_finite
    assert result.posterior.draws_shape == (20, 1, 4)
    assert result.posterior_draw.draws_shape == (20, 1, 4)
    assert result.epistemic_scale > 0.0
    assert result.aleatoric_scale >= 0.0
    assert result.num_obs == 40
    assert result.num_test == 20
    assert result.num_dim == 2
    assert result.num_draws == 4


def test_meanseandformat():
    from ops.stress import MeanSE, format_se, mean_se

    one = mean_se([2.0])
    assert one.mean == pytest.approx(2.0)
    assert not np.isfinite(one.se)
    assert format_se(one) == "2"

    two = mean_se([1.0, 3.0])
    assert two.mean == pytest.approx(2.0)
    assert two.se == pytest.approx(
        1.0
    )  # std=sqrt(2)/1? ddof=1: std=sqrt(2), se=sqrt(2)/sqrt(2)=1
    assert format_se(two) == "2 ± 1"
    assert format_se(MeanSE(0.2193, 0.0123), fmt="0.4f") == "0.2193 ± 0.0123"


def test_007():
    from ops.stress import (
        DURATION_S_FMT,
        DrawMethodAggregate,
        DrawMethodResult,
        DrawStressAggregate,
        DrawStressResult,
        MeanSE,
        SampleStressResult,
        format_header3,
        format_aggregate,
        format_summary2,
        format_aggregate2,
        format_summary,
        format_row,
    )

    assert DURATION_S_FMT == ".4f"
    # Gate-visible cover for enn-add rows (testopsstress.py is ignored by make test).
    assert format_row(10, 1.2345, 0.0567, n_width=6) == "    10 1.2345 0.0567"
    assert format_row(100_000, 0.5, 12.3, n_width=6) == "100000 0.5000 12.3000"

    sample = SampleStressResult(
        num_dim=2,
        num_obs=5,
        num_samples=3,
        seed=1,
        num_function_seeds=1,
        draws_shape=(3, 1, 1),
        all_finite=True,
        init_s=0.12345,
        sample_s=0.01,
    )
    summary = format_summary(sample)
    assert "init_s=0.1235" in summary
    assert "sample_s=0.0100" in summary

    method = DrawMethodResult(
        method="posterior",
        avg_likelihood=1.0,
        argmin_rms=0.1,
        argmin_rate=0.5,
        draws_shape=(2, 1, 2),
        all_finite=True,
        eval_s=0.12345,
    )
    assert "eval_s=0.1235" in format_summary2(method)

    result = DrawStressResult(
        num_obs=4,
        num_test=2,
        num_dim=2,
        seed=0,
        k=5,
        num_candidates=8,
        num_samples=5,
        num_draws=4,
        epistemic_scale=1.0,
        aleatoric_scale=0.1,
        fit_s=0.98765,
        posterior=method,
        posterior_draw=method,
    )
    assert "fit_s=0.9877" in format_header3(result)

    method_agg = DrawMethodAggregate(
        method="posterior",
        avg_likelihood=MeanSE(1.0, 0.1),
        argmin_rms=MeanSE(0.1, 0.01),
        argmin_rate=MeanSE(0.5, 0.05),
        eval_s=MeanSE(0.12345, 0.00678),
    )
    assert "eval_s=0.1235 ± 0.0068" in format_aggregate2(method_agg)

    agg = DrawStressAggregate(
        num_obs=4,
        num_test=2,
        num_dim=2,
        seed=0,
        num_seeds=2,
        k=5,
        num_candidates=8,
        num_samples=5,
        num_draws=4,
        epistemic_scale=MeanSE(1.0, 0.1),
        aleatoric_scale=MeanSE(0.1, 0.01),
        fit_s=MeanSE(0.98765, 0.00123),
        posterior=method_agg,
        posterior_draw=method_agg,
    )
    assert "fit_s=0.9877 ± 0.0012" in format_aggregate(agg)


def test_008():
    from ops.stress import DrawStressConfig, run_seeds

    agg = run_seeds(
        DrawStressConfig(
            num_obs=40,
            num_test=20,
            num_dim=2,
            seed=0,
            k=5,
            num_candidates=8,
            num_samples=5,
            num_draws=4,
        ),
        num_seeds=3,
    )
    assert agg.num_seeds == 3
    assert agg.seed == 0
    assert np.isfinite(agg.posterior.avg_likelihood.mean)
    assert np.isfinite(agg.posterior.avg_likelihood.se)
    assert np.isfinite(agg.posterior_draw.argmin_rate.mean)
    assert np.isfinite(agg.posterior_draw.argmin_rate.se)
    assert agg.posterior.argmin_rate.se >= 0.0


def _cliargs(*, num_seeds: int | None = None) -> list[str]:
    args = [
        "draw",
        "40",
        "20",
        "--num-dim",
        "2",
        "--seed",
        "0",
        "--k",
        "5",
        "--num-fit-candidates",
        "8",
        "--num-fit-samples",
        "5",
        "--num-draws",
        "4",
    ]
    if num_seeds is not None:
        args.extend(["--num-seeds", str(num_seeds)])
    return args


def _cliline(line: str, method: str, *, with_se: bool) -> None:
    assert line.startswith(f"{method} avg_likelihood=")
    assert "argmin_rms=" in line
    assert "argmin_rate=" in line
    if with_se:
        assert " ± " in line
    else:
        assert " ± " not in line
        assert "draws_shape=" not in line
        assert "all_finite=" not in line
    avg = float(line.split("avg_likelihood=")[1].split()[0])
    rms = float(line.split("argmin_rms=")[1].split()[0])
    hit_tok = line.split("argmin_rate=")[1].split()[0]
    if with_se:
        hit_tok = line.split("argmin_rate=")[1].split("eval_s=")[0].strip()
        mean_s, se_s = hit_tok.split(" ± ")
        hit = float(mean_s)
        assert mean_s == f"{hit:0.4f}"
        assert se_s == f"{float(se_s):0.4f}"
    else:
        hit = float(hit_tok)
        assert hit_tok == f"{hit:0.4f}"
    assert np.isfinite(avg)
    assert np.isfinite(rms)
    assert rms >= 0.0
    assert 0.0 <= hit <= 1.0


def test_009():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, _cliargs())
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("num_dim=2 num_obs=40 num_test=20 seed=0")
    assert "num_seeds=1" in lines[0]
    assert "num_draws=4" in lines[0]
    assert "epistemic_scale=" in lines[0]
    assert "aleatoric_scale=" in lines[0]
    _cliline(lines[1], "posterior", with_se=False)
    _cliline(lines[2], "posterior_draw", with_se=False)


def test_010():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, _cliargs(num_seeds=3))
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 3
    assert "num_seeds=3" in lines[0]
    assert " ± " in lines[0]
    _cliline(lines[1], "posterior", with_se=True)
    _cliline(lines[2], "posterior_draw", with_se=True)


def test_011():
    from click.testing import CliRunner

    from ops.stress import DEFAULT_DRAW_NUM_DRAWS, DEFAULT_DRAW_NUM_SEEDS, cli

    assert DEFAULT_DRAW_NUM_DRAWS == 100
    assert DEFAULT_DRAW_NUM_SEEDS == 1
    result = CliRunner().invoke(cli, ["draw", "--help"])
    assert result.exit_code == 0, result.output
    assert "100" in result.output
    assert "--num-draws" in result.output
    assert "--num-seeds" in result.output
    assert "--num-samples" not in result.output


def test_012():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["draw", "0", "10"])
    assert result.exit_code != 0
    assert "num_obs must be >= 1" in result.output


def test_013():
    from click.testing import CliRunner

    from ops.stress import cli

    result = CliRunner().invoke(cli, ["draw"])
    assert result.exit_code != 0
