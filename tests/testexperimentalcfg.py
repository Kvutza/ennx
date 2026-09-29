from dataclasses import asdict

import pytest

from ennx import experimental


@pytest.mark.parametrize("overrides", [{}, {"sampler": "gaussian", "length_init": 0.1}])
def test_settings(monkeypatch, overrides):
    calls = []

    def constructor(*args, **kwargs):
        calls.append((args, kwargs))
        return "search"

    monkeypatch.setattr(experimental, "__getattr__", lambda _: constructor)
    config = experimental.TurboEnnConfig(reference_seed=23)
    assert (
        experimental.turbo_enn("base", -1.0, [], 4, config=config, **overrides)
        == "search"
    )
    assert calls == [(("base", -1.0, [], 4), asdict(config) | overrides)]
    assert config.reference_seed == 23 and config.length_init == 0.8


def test_unknown():
    with pytest.raises(TypeError, match="unexpected keyword"):
        experimental.turbo_enn("base", -1.0, [], 4, unknown_scale=0.5)
