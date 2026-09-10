import os
import subprocess
import sys

import numpy as np
import pytest

from ennx._trials import Loop, Space


def test_space():
    space = Space({"z": (-2, 2), "a": (0.001, 1)}, logs=["a"])
    assert space.names == ("a", "z")
    point = {"z": 0.5, "a": 0.1}
    np.testing.assert_allclose(space.encode(point), [np.log(0.1), 0.5])
    decoded = space.decode(space.encode(point))
    assert decoded == pytest.approx(point)
    for bad in [{"a": 1}, {"a": 0, "z": 1}, {"a": 1, "z": float("nan")}]:
        with pytest.raises(ValueError):
            space.encode(bad)
    for bounds in [{}, {"a": (1, 1)}, {"a": (2, 1)}, {"a": (0, float("inf"))}]:
        with pytest.raises(ValueError):
            Space(bounds)
    with pytest.raises(ValueError, match="positive"):
        Space({"a": (-1, 1)}, ["a"])


def test_history(config):
    loop = Loop(Space({"x": (0, 1)}), config, seed=4)
    records = {"0": ({"x": 0.1}, 1.0, 0.04)}
    loop.update(records)
    loop.update(records)
    assert len(loop._seen) == 1
    for bad in [
        {},
        {"0": ({"x": 0.1}, 2.0, 0.04)},
        {**records, "1": ({"x": 0.2}, float("inf"), None)},
        {**records, "1": ({"x": 0.2}, 1.0, -0.1)},
        {**records, "1": ({"x": 0.2}, 1.0, None)},
    ]:
        with pytest.raises(ValueError):
            loop.update(bad)
        assert len(loop._seen) == 1


def test_plain_import_is_light():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, ennx; assert not {'torch', 'botorch', 'optuna', 'ax'} & sys.modules.keys()",
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
