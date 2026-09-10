from __future__ import annotations

import sys
from pathlib import Path

from ._lazy import module_attr


def _extendextension() -> None:
    package_dir = Path(__file__).resolve().parent
    package_paths = globals().get("__path__")
    if package_paths is None:
        return

    for entry in sys.path:
        try:
            candidate = Path(entry) / "ennx"
        except TypeError:
            continue
        if candidate == package_dir:
            continue
        if not any(candidate.glob("ennx_rust*.so")) and not any(
            candidate.glob("ennx_rust*.pyd")
        ):
            continue
        candidate_str = str(candidate)
        if candidate_str not in package_paths:
            package_paths.append(candidate_str)
        break


_extendextension()

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ENN": (".ennx.enn_class", "ENN"),
    "ENNStatefulFitter": (".ennx.enn_fitter", "ENNStatefulFitter"),
    "experimental": (".experimental", "experimental"),
    "create_optimizer": (".turbo.optimizer", "create_optimizer"),
    "enn_optimizer": ("._rust", "enn_optimizer"),
    "create_zero": ("._rust", "create_zero"),
    "create_lhd": ("._rust", "create_lhd"),
    "Telemetry": (".turbo.optimizer", "Telemetry"),
    "OptimizerConfig": (".turbo.optimizer_config", "OptimizerConfig"),
    "turbo_one": (".turbo.optimizer_config", "turbo_one"),
    "turbo_zero": (".turbo.optimizer_config", "turbo_zero"),
    "turbo_enn": (".turbo.optimizer_config", "turbo_enn"),
    "lhd_only": (".turbo.optimizer_config", "lhd_only"),
    "TurboTRConfig": (".turbo.config.trust_region", "TurboTRConfig"),
    "MorboTRConfig": (".turbo.config.trust_region", "MorboTRConfig"),
    "NoTRConfig": (".turbo.config.trust_region", "NoTRConfig"),
    "CandidateRV": (".turbo.optimizer_config", "CandidateRV"),
    "InitStrategy": (".turbo.optimizer_config", "InitStrategy"),
    "AcqType": (".turbo.optimizer_config", "AcqType"),
}


def __getattr__(name: str):
    return module_attr(name, globals())


__all__: list[str] = [
    "AcqType",
    "CandidateRV",
    "ENNStatefulFitter",
    "ENN",
    "InitStrategy",
    "MorboTRConfig",
    "NoTRConfig",
    "OptimizerConfig",
    "Telemetry",
    "TurboTRConfig",
    "create_optimizer",
    "enn_optimizer",
    "create_zero",
    "create_lhd",
    "experimental",
    "lhd_only",
    "turbo_enn",
    "turbo_one",
    "turbo_zero",
]
