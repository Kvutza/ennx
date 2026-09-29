"""Architecture and numerical smoke tests for the dense Qwen control path."""

import json

import pytest

from ops.qwen import checkpoint
from ops.qwen.config import Config


def test_contexts(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from ops.qwen import bench

    seen_contexts = []

    class FakeEncoding:
        ids = [7, 8, 9]

    class FakeTokenizer:
        def encode(self, prompt, add_special_tokens):
            return FakeEncoding()

    class FakeEvaluator:
        def __init__(self, directory, *, max_tokens):
            self.max_tokens = max_tokens
            self.weights = object()
            seen_contexts.append(max_tokens)

        def bench_generate(self, weights, tokens, max_new_tokens, *, mode):
            assert len(tokens) == self.max_tokens - max_new_tokens
            return {
                "device_name": "Fake Metal",
                "prompt_tokens": len(tokens),
                "generated_tokens": max_new_tokens,
                "requested_generated_tokens": max_new_tokens,
                "kv_cache_bytes": self.max_tokens * 1024,
                "logits_read_to_cpu": False,
                "token_selection_on_gpu": True,
                "command_buffer_count": self.max_tokens // 256,
                "prefill_ms": float(self.max_tokens),
                "first_token_ms": 1.0,
                "decode_ms_after_first": 2.0,
                "steady_decode_tokens_per_second": 500.0,
                "total_ms": float(self.max_tokens + 3),
                "end_to_end_generated_tokens_per_second": 100.0,
                "host_overhead_ms": 0.0,
                "decode_kernel_trace": ["qwen_dattn", "qwen_argmax"],
                "decode_kernel_times_ms": [("qwen_decode_token_total", 2.0)],
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
            "sweep",
            "--max-new-tokens",
            "2",
            "--context-length",
            "all",
            "--output",
            str(tmp_path / "reports" / "qwen-sweep.json"),
            "--repeats",
            "1",
            "--warmups",
            "0",
        ],
    )

    assert result.exit_code == 0
    report = json.loads(result.output)
    saved = json.loads((tmp_path / "reports" / "qwen-sweep.json").read_text())
    assert saved == report
    assert bench.TARGET_CONTEXT_LENGTHS == (4096, 16384, 32768)
    assert seen_contexts == [4096, 16384, 32768]
    assert [row["target_context_length"] for row in report["results"]] == [
        4096,
        16384,
        32768,
    ]
    assert report["target_context_status"] == {
        "4096": {"requested": True, "passed": True},
        "16384": {"requested": True, "passed": True},
        "32768": {"requested": True, "passed": True},
    }
