from __future__ import annotations

import importlib
from typing import Any


def module_attr(name: str, namespace: dict[str, Any]) -> Any:
    spec = namespace["_LAZY_ATTRS"].get(name)
    if spec is None:
        raise AttributeError(
            f"module {namespace['__name__']!r} has no attribute {name!r}"
        )
    rel_module, attr_name = spec
    try:
        module = importlib.import_module(rel_module, namespace["__package__"])
        return getattr(module, attr_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}. Install extras via `pip install 'ennx[with-deps]'`."
        ) from e
