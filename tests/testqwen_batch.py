"""Architecture and numerical smoke tests for the dense Qwen control path."""

import json

import pytest

from ops.qwen import checkpoint
from ops.qwen.config import Config


def test_benchbatch(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from ops.qwen import bench

    batch_calls = 0

    class FakeEncoding:
        ids = [5, 6]

    class FakeTokenizer:
        def encode(self, prompt, add_special_tokens):
            return FakeEncoding()

    class FakeEvaluator:
        def __init__(self, directory, *, max_tokens):
            self.weights = object()

        def bench_generate(self, weights, tokens, max_new_tokens, *, mode):
            raise AssertionError("batch benchmarks must use native generate_batch")

        def generate_batch(self, weights, prompts, max_new_tokens):
            nonlocal batch_calls
            batch_calls += 1
            assert len(prompts) == 3
            return [prompt + [99, 100] for prompt in prompts]

    monkeypatch.setattr(bench, "load", lambda directory: FakeTokenizer())
    monkeypatch.setattr(bench, "Evaluator", FakeEvaluator)

    result = CliRunner().invoke(
        bench.main,
        [
            str(tmp_path),
            "--prompt",
            "batched",
            "--max-new-tokens",
            "2",
            "--batch-size",
            "3",
            "--repeats",
            "1",
            "--warmups",
            "0",
        ],
    )

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert batch_calls == 1
    assert report["batch_size"] == 3
    assert report["generated_tokens"] == 6
    assert report["native_streams_per_repeat"] == 1
    assert report["native_batch_profile"] is True
    assert report["python_token_loop"] is False
    assert report["decode_bottleneck_candidates"] == ["qwen_batched_generation_total"]
