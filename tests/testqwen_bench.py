"""Architecture and numerical smoke tests for the dense Qwen control path."""

import json

import pytest

from ops.qwen import checkpoint
from ops.qwen.config import Config


def test_benchopts(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from ops.qwen import bench

    class FakeEncoding:
        ids = [11, 12]

    class FakeTokenizer:
        def encode(self, prompt, add_special_tokens):
            assert prompt == "hi"
            assert add_special_tokens is False
            return FakeEncoding()

    class FakeEvaluator:
        def __init__(self, directory, *, max_tokens):
            assert directory == tmp_path
            assert max_tokens == 4096
            self.weights = object()

        def bench_generate(self, weights, tokens, max_new_tokens, *, mode):
            assert weights is self.weights
            assert len(tokens) == 4094
            assert max_new_tokens == 2
            assert mode == "greedy"
            return {
                "device_name": "Fake Metal",
                "prompt_tokens": len(tokens),
                "generated_tokens": 2,
                "requested_generated_tokens": max_new_tokens,
                "kv_cache_bytes": 123,
                "logits_read_to_cpu": False,
                "token_selection_on_gpu": True,
                "command_buffer_count": 18,
                "prefill_ms": 10.0,
                "first_token_ms": 2.0,
                "decode_ms_after_first": 3.0,
                "steady_decode_tokens_per_second": 333.0,
                "total_ms": 15.0,
                "end_to_end_generated_tokens_per_second": 133.0,
                "host_overhead_ms": 0.0,
                "decode_kernel_trace": ["qwen_dattn", "qwen_argmax"],
                "decode_kernel_times_ms": [("qwen_decode_token_total", 3.0)],
                "decode_bottleneck_candidates": ["qwen_decode_token_total"],
                "lm_head_argmax_decision": "measure before fusion",
            }

    monkeypatch.setattr(bench, "load", lambda directory: FakeTokenizer())
    monkeypatch.setattr(bench, "Evaluator", FakeEvaluator)

    result = CliRunner().invoke(
        bench.main,
        [
            str(tmp_path),
            "--prompt",
            "hi",
            "--max-new-tokens",
            "2",
            "--context-length",
            "4096",
            "--repeats",
            "1",
            "--warmups",
            "0",
        ],
    )

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["python_token_loop"] is False
    assert report["logits_read_to_cpu"] is False
    assert report["decode_kernel_trace"] == ["qwen_dattn", "qwen_argmax"]
    assert report["decode_kernel_times_ms"] == [["qwen_decode_token_total", 3.0]]
    assert report["decode_bottleneck_candidates"] == ["qwen_decode_token_total"]
    assert report["target_context_status"]["4096"] == {
        "requested": True,
        "passed": True,
    }
    assert report["target_context_status"]["16384"] == {
        "requested": False,
        "passed": False,
    }
