import ast

from scripts.check_names import _overrides


def test_overrides():
    tree = ast.parse("""
from optuna.samplers import BaseSampler
from typing import override
class Sampler(BaseSampler):
    @override
    def infer_relative_search_space(self): pass
    def locally_named_long_method(self): pass
""")
    methods = _overrides(tree)
    names = {node.name for node in ast.walk(tree) if id(node) in methods}
    assert names == {"infer_relative_search_space"}


def test_local():
    tree = ast.parse("""
from typing import override
from ennx.example import Base
class Local(Base):
    @override
    def locally_named_long_method(self): pass
@override
def another_local_long_function(): pass
""")
    assert not _overrides(tree)
