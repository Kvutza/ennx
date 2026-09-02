from __future__ import annotations

import importlib
import sys
from pathlib import Path

from ._lazy import lazy_getattr


def _extend_path_with_installed_extension() -> None:
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


_extend_path_with_installed_extension()

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "EpistemicNearestNeighbors": (".ennx.enn_class", "EpistemicNearestNeighbors"),
    "ENNStatefulFitter": (".ennx.enn_fitter", "ENNStatefulFitter"),
    "experimental": (".experimental", "experimental"),
    "create_optimizer": (".turbo.optimizer", "create_optimizer"),
    "create_optimizer_enn": ("._rust", "create_optimizer_enn"),
    "create_optimizer_zero": ("._rust", "create_optimizer_zero"),
    "create_optimizer_lhd": ("._rust", "create_optimizer_lhd"),
    "Telemetry": (".turbo.optimizer", "Telemetry"),
    "OptimizerConfig": (".turbo.optimizer_config", "OptimizerConfig"),
    "turbo_one_config": (".turbo.optimizer_config", "turbo_one_config"),
    "turbo_zero_config": (".turbo.optimizer_config", "turbo_zero_config"),
    "turbo_enn_config": (".turbo.optimizer_config", "turbo_enn_config"),
    "lhd_only_config": (".turbo.optimizer_config", "lhd_only_config"),
    "TurboTRConfig": (".turbo.config.trust_region", "TurboTRConfig"),
    "MorboTRConfig": (".turbo.config.trust_region", "MorboTRConfig"),
    "NoTRConfig": (".turbo.config.trust_region", "NoTRConfig"),
    "CandidateRV": (".turbo.optimizer_config", "CandidateRV"),
    "InitStrategy": (".turbo.optimizer_config", "InitStrategy"),
    "AcqType": (".turbo.optimizer_config", "AcqType"),
}


def __getattr__(name: str):
    if name in _DEPRECATED_ATTRS:
        import warnings

        rel_module, attr_name = _DEPRECATED_ATTRS[name]
        module = importlib.import_module(rel_module, __package__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        warnings.warn(
            f"ennx.{name} is experimental; use ennx.experimental.{name}",
            DeprecationWarning,
            stacklevel=2,
        )
        return attr
    return lazy_getattr(
        name=name,
        module_name=__name__,
        package=__package__,
        mapping=_LAZY_ATTRS,
        extra="`pip install 'ennx[with-deps]'`",
    )


__all__: list[str] = [
    "AcqType",
    "CandidateRV",
    "ENNStatefulFitter",
    "EpistemicNearestNeighbors",
    "InitStrategy",
    "MorboTRConfig",
    "NoTRConfig",
    "OptimizerConfig",
    "Telemetry",
    "TurboTRConfig",
    "create_optimizer",
    "create_optimizer_enn",
    "create_optimizer_lhd",
    "create_optimizer_zero",
    "experimental",
    "lhd_only_config",
    "turbo_enn_config",
    "turbo_one_config",
    "turbo_zero_config",
]

_DEPRECATED_ATTRS: dict[str, tuple[str, str]] = {
    "quantize_int4": (".quantization", "quantize_int4"),
    "quantize_fp4_e2m1": (".quantization", "quantize_fp4_e2m1"),
}
