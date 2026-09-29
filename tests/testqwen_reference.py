"""Architecture and numerical smoke tests for the dense Qwen control path."""

import json

import pytest

from ops.qwen import checkpoint
from ops.qwen.config import Config


def test_refcap(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from ops.qwen import bench

    class FakeEncoding:
        ids = [1]

    class FakeTokenizer:
        def encode(self, prompt, add_special_tokens):
            return FakeEncoding()

    class FakeEvaluator:
        def __init__(self, directory, *, max_tokens):
            self.weights = object()

        def bench_generate(self, weights, tokens, max_new_tokens, *, mode):
            return {
                "device_name": "Fake Metal",
                "prompt_tokens": len(tokens),
                "generated_tokens": 1,
                "requested_generated_tokens": max_new_tokens,
                "kv_cache_bytes": 1,
                "logits_read_to_cpu": False,
                "token_selection_on_gpu": True,
                "command_buffer_count": 1,
                "prefill_ms": 1.0,
                "first_token_ms": 1.0,
                "decode_ms_after_first": 0.0,
                "steady_decode_tokens_per_second": 0.0,
                "total_ms": 2.0,
                "end_to_end_generated_tokens_per_second": 500.0,
                "host_overhead_ms": 0.0,
                "decode_kernel_trace": ["qwen_dattn", "qwen_argmax"],
                "decode_kernel_times_ms": [],
                "decode_bottleneck_candidates": [],
                "lm_head_argmax_decision": "measure before fusion",
            }

    monkeypatch.setattr(bench, "load", lambda directory: FakeTokenizer())
    monkeypatch.setattr(bench, "Evaluator", FakeEvaluator)

    result = CliRunner().invoke(
        bench.main,
        [
            str(tmp_path),
            "--prompt",
            "x",
            "--max-new-tokens",
            "1",
            "--context-length",
            "4096",
            "--reference",
            "--repeats",
            "1",
            "--warmups",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "limited to 256 tokens" in result.output
