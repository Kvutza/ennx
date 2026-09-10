from __future__ import annotations

from .parity_types import ParityCase, ParityReport


def run_parity(
    report: ParityReport,
    name: str,
    config,
) -> None:
    import numpy as np

    from .parchksob import append_case

    try:
        from ennx._rust import Optimizer  # noqa: F401
    except ImportError:
        append_case(
            report,
            name=name,
            endpoint="create_optimizer",
            backend="rust",
            reason="Rust not available",
        )
        return

    def _run() -> None:
        from .optimizer_checks import check_opt, make_optimizer

        bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        opt = make_optimizer(bounds, config, seed=47)
        check_opt(opt, bounds)

    try:
        _run()
    except Exception as e:
        report.cases.append(
            ParityCase(
                name=name,
                endpoint="create_optimizer",
                passed=False,
                error=str(e),
                backend="rust",
            )
        )
        report.failed += 1
    else:
        report.cases.append(
            ParityCase(
                name=name, endpoint="create_optimizer", passed=True, backend="rust"
            )
        )
        report.passed += 1
    report.total += 1
