from __future__ import annotations

from .parity_types import ParityCase, ParityReport


def _posteriorfields(out, rs_se, rs_se_epi, rs_se_ale, *, rtol: float, atol: float):
    import numpy as np

    return (
        np.allclose(out.se, rs_se, rtol=rtol, atol=atol)
        and np.allclose(out.se_epi, rs_se_epi, rtol=rtol, atol=atol)
        and np.allclose(out.se_ale, rs_se_ale, rtol=rtol, atol=atol)
    )


def _recordcase(
    report: ParityReport, name: str, endpoint: str, passed: bool, error: str | None
) -> None:
    report.cases.append(
        ParityCase(
            name=name,
            endpoint=endpoint,
            passed=bool(passed),
            error=error,
            backend="rust_vs_python",
        )
    )
    if passed:
        report.passed += 1
    else:
        report.failed += 1
    report.total += 1


def run_simple(report: ParityReport) -> None:
    import numpy as np

    from ennx import ENN
    from ennx.ennx.enn_params import ENNParams, PosteriorFlags

    try:
        from ennx._rust import ENN as RustENN
    except ImportError:
        report.cases.append(
            ParityCase(
                name="posterior_simple",
                endpoint="ENN.posterior",
                passed=False,
                error="Rust not available",
                backend="rust_vs_python",
            )
        )
        report.skipped += 1
        report.total += 1
        return

    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    params = ENNParams(
        k_neighbors=2,
        epistemic_scale=1.0,
        aleatoric_scale=0.1,
    )
    flags = PosteriorFlags(exclude_nearest=False, observation_noise=False)

    try:
        model = ENN(train_x, train_y, scale_x=False)
        out = model.posterior(query, params=params, flags=flags)
    except Exception as e:
        report.cases.append(
            ParityCase(
                name="posterior_simple",
                endpoint="ENN.posterior",
                passed=False,
                error=str(e),
                backend="rust_vs_python",
            )
        )
        report.failed += 1
        report.total += 1
        return

    try:
        rs_model = RustENN(train_x, train_y, scale_x=False, index_driver="Exact")
        rs_mu, rs_se, rs_se_epi, rs_se_ale, _ = rs_model.posterior(
            query,
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
            exclude_nearest=False,
            observation_noise=False,
        )
    except Exception as e:
        report.cases.append(
            ParityCase(
                name="posterior_simple",
                endpoint="ENN.posterior",
                passed=False,
                error=str(e),
                backend="rust_vs_python",
            )
        )
        report.failed += 1
        report.total += 1
        return

    mu_ok = np.allclose(out.mu, rs_mu, rtol=1e-12, atol=1e-12)
    se_ok = _posteriorfields(out, rs_se, rs_se_epi, rs_se_ale, rtol=1e-12, atol=1e-12)
    passed = mu_ok and se_ok and out.idx is not None

    _recordcase(
        report,
        "posterior_simple",
        "ENN.posterior",
        passed,
        None if passed else f"mu_ok={mu_ok} se_ok={se_ok} idx={out.idx is not None}",
    )


def run_noise(report: ParityReport) -> None:
    import numpy as np

    from ennx import ENN
    from ennx.ennx.enn_params import ENNParams, PosteriorFlags

    try:
        from ennx._rust import ENN as RustENN
    except ImportError:
        report.cases.append(
            ParityCase(
                name="posterior_observation_noise",
                endpoint="ENN.posterior",
                passed=False,
                error="Rust not available",
                backend="rust_vs_python",
            )
        )
        report.skipped += 1
        report.total += 1
        return

    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    train_yvar = np.array([[0.01], [0.01], [0.01], [0.01]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    params = ENNParams(
        k_neighbors=2,
        epistemic_scale=1.0,
        aleatoric_scale=0.1,
    )
    flags = PosteriorFlags(exclude_nearest=False, observation_noise=True)

    try:
        model = ENN(train_x, train_y, train_yvar=train_yvar, scale_x=False)
        out = model.posterior(query, params=params, flags=flags)
    except Exception as e:
        report.cases.append(
            ParityCase(
                name="posterior_observation_noise",
                endpoint="ENN.posterior",
                passed=False,
                error=str(e),
                backend="rust_vs_python",
            )
        )
        report.failed += 1
        report.total += 1
        return

    try:
        rs_model = RustENN(
            train_x, train_y, train_yvar=train_yvar, scale_x=False, index_driver="Exact"
        )
        rs_mu, rs_se, rs_se_epi, rs_se_ale, _ = rs_model.posterior(
            query,
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
            exclude_nearest=False,
            observation_noise=True,
        )
    except Exception as e:
        report.cases.append(
            ParityCase(
                name="posterior_observation_noise",
                endpoint="ENN.posterior",
                passed=False,
                error=str(e),
                backend="rust_vs_python",
            )
        )
        report.failed += 1
        report.total += 1
        return

    mu_ok = np.allclose(out.mu, rs_mu, rtol=1e-12, atol=1e-12)
    se_ok = _posteriorfields(out, rs_se, rs_se_epi, rs_se_ale, rtol=1e-10, atol=1e-10)
    passed = mu_ok and se_ok

    _recordcase(
        report,
        "posterior_observation_noise",
        "ENN.posterior",
        passed,
        None if passed else f"mu_ok={mu_ok} se_ok={se_ok}",
    )
