"""Architecture and numerical smoke tests for the dense Qwen control path."""

import json

import pytest

from ops.qwen import checkpoint
from ops.qwen.config import Config


def test_pins():
    config = Config.from_json(
        {
            "model_type": "qwen2",
            "vocab_size": 151936,
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "bos_token_id": 151643,
            "eos_token_id": 151643,
            "tie_word_embeddings": True,
        }
    )
    assert config == Config()
    assert config.head_dim == 128
    assert config.kv_repeat == 6


def test_drift():
    data = Config().__dict__ | {"model_type": "qwen2", "hidden_size": 1152}
    with pytest.raises(ValueError, match="not Qwen2.5-Coder-1.5B"):
        Config.from_json(data)


def test_resume(monkeypatch, tmp_path):
    config_data = Config().__dict__ | {"model_type": "qwen2"}
    (tmp_path / "config.json").write_text(json.dumps(config_data))
    manifest = {
        "model_id": checkpoint.MODEL_ID,
        "revision": checkpoint.REVISION,
        "config": Config().__dict__,
        "weights_downloaded": False,
        "files": {
            name: {"sha256": "old", "bytes": 1} for name in checkpoint.SMALL_FILES
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    downloaded = []

    def fake_download(name, output, *, force):
        downloaded.append((name, force))
        (output / name).write_bytes(b"weights")
        return "new"

    monkeypatch.setattr(checkpoint, "_download", fake_download)
    result = checkpoint.fetch_model(tmp_path, weights=True)
    assert downloaded == [("model.safetensors", False)]
    assert result["weights_downloaded"] is True
    assert set(result["files"]) == set(checkpoint.SMALL_FILES) | {"model.safetensors"}


def test_exportbest(tmp_path):
    np = pytest.importorskip("numpy")
    from ops.qwen.bo import _ckptheader, _exportbest

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    header = {
        "z": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
        "a": {"dtype": "BF16", "shape": [2], "data_offsets": [8, 12]},
    }
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    padding = b" " * ((8 - len(encoded) % 8) % 8)
    (source / "model.safetensors").write_bytes(
        len(encoded + padding).to_bytes(8, "little") + encoded + padding + b"\0" * 12
    )
    for name in checkpoint.SMALL_FILES:
        (source / name).write_bytes(b"metadata")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "test/model",
                "revision": "test",
                "config": {},
            }
        )
    )

    best = np.arange(6, dtype=np.uint16)
    output.mkdir()
    _exportbest(source, output, best)

    written = output / "model.safetensors"
    result_header = _ckptheader(written)
    with written.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
    assert written.stat().st_size == 8 + header_size + 12
    assert result_header["a"]["data_offsets"] == [0, 4]
    assert result_header["z"]["data_offsets"] == [4, 12]
    assert written.read_bytes()[-12:] == best.tobytes()
    assert not (output / ".model.safetensors.tmp").exists()
    assert not (output / ".manifest.json.tmp").exists()


def test_tinyloss():
    torch = pytest.importorskip("torch")
    from ops.qwen.model import forward, loss

    config = Config(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
    )
    params = {
        "model.embed_tokens.weight": torch.randn(17, 8, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        params.update(
            {
                prefix + "input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
                prefix + "post_attention_layernorm.weight": torch.ones(
                    8, dtype=torch.bfloat16
                ),
                prefix + "self_attn.q_proj.weight": torch.randn(
                    8, 8, dtype=torch.bfloat16
                ),
                prefix + "self_attn.q_proj.bias": torch.zeros(8, dtype=torch.bfloat16),
                prefix + "self_attn.k_proj.weight": torch.randn(
                    4, 8, dtype=torch.bfloat16
                ),
                prefix + "self_attn.k_proj.bias": torch.zeros(4, dtype=torch.bfloat16),
                prefix + "self_attn.v_proj.weight": torch.randn(
                    4, 8, dtype=torch.bfloat16
                ),
                prefix + "self_attn.v_proj.bias": torch.zeros(4, dtype=torch.bfloat16),
                prefix + "self_attn.o_proj.weight": torch.randn(
                    8, 8, dtype=torch.bfloat16
                ),
                prefix + "mlp.gate_proj.weight": torch.randn(
                    16, 8, dtype=torch.bfloat16
                ),
                prefix + "mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
                prefix + "mlp.down_proj.weight": torch.randn(
                    8, 16, dtype=torch.bfloat16
                ),
            }
        )
    tokens = torch.tensor([[1, 3, 4, 2]], dtype=torch.int64)
    logits = forward(params, tokens, config)
    assert logits.shape == (1, 4, 17)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss(logits, tokens))


def test_tokrows(monkeypatch, tmp_path):
    tokenizer = pytest.importorskip("tokenizers")
    from ops.qwen import tokenizer as qwen_tokenizer

    vocab = {"a": 0}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": {}})
    )

    class SmallConfig:
        vocab_size = 2

    monkeypatch.setattr(qwen_tokenizer, "Config", SmallConfig)
    loaded = qwen_tokenizer.load(tmp_path)
    assert loaded.get_vocab_size(with_added_tokens=True) == 1


def test_decode():
    from ops.qwen.joint import DecoderPolicy, _decoderpolicy

    assert _decoderpolicy([0.2, 0.9]) == DecoderPolicy(0.2, 0.9)
    with pytest.raises(ValueError, match="valid sampling domain"):
        _decoderpolicy([0.0, 0.9])


def test_fimmeta():
    from ops.qwen.eval import validate_obj

    base = {
        "provenance": {"prompt_template": "ennx.qwen_fim_code.v1"},
        "examples": [{"setup": "", "tests": ["assert True"]}],
    }
    validate_obj(base)
    with pytest.raises(ValueError, match="FIM objective"):
        validate_obj(
            base | {"provenance": {"prompt_template": "ennx.qwen_chat_code.v1"}}
        )
    with pytest.raises(ValueError, match="checker setup/tests"):
        validate_obj(base | {"examples": [{"setup": None, "tests": []}]})


def test_coverage():
    from ops.qwen.joint import ScoreTiming
    from ops.qwen.metal import proposal_stats

    class FakeProposals:
        def describe(self):
            return [(7, 0.25, 0.01, [(3, 1.5), (0, 0.0), (2, 0.5)])]

        def geometry(self):
            return [(0, 0.75)]

        def base_id(self):
            return 5

        def history_dists(self):
            return [(11, 0.75), (19, 1.25)]

        def pool(self):
            return [
                (0, 7, 0.01, 0.75, [(11, 0.75), (19, 1.25)]),
                (1, 7, 0.04, 0.75, [(11, 0.8), (19, 1.1)]),
                (2, 9, 0.01, 0.0, [(11, 0.9), (19, 1.0)]),
                (3, 9, 0.04, 0.0, [(11, 1.0), (19, 0.9)]),
            ]

        def pool_geometry(self):
            return (
                [0.011, 0.039, 0.0105, 0.041],
                [
                    (0, 1, 0.99),
                    (0, 2, 0.02),
                    (0, 3, 0.01),
                    (1, 2, 0.03),
                    (1, 3, 0.02),
                    (2, 3, 0.98),
                ],
                [0.76, 0.74, 0.01, -0.02],
            )

    metrics = proposal_stats(FakeProposals(), 10)
    assert metrics["changed_bf16_elements"] == 5
    assert metrics["changed_bf16_fraction"] == 0.5
    assert metrics["changed_blocks"] == 2
    assert metrics["total_blocks"] == 3
    assert metrics["base_observation_id"] == 5
    assert metrics["candidate_index"] == 0
    assert metrics["reference_correlation"] == 0.75
    assert metrics["realized_reference_cosine"] == 0.76
    assert metrics["reference_normalization"] == "per_tensor_bf16_rms"
    assert metrics["realized_radius"] == 0.011
    assert metrics["history_distances"] == [
        {"observation_id": 11, "squared_distance": 0.75},
        {"observation_id": 19, "squared_distance": 1.25},
    ]
    assert metrics["candidate_pool"][2]["reference_correlation"] == 0.0
    assert metrics["candidate_pool"][2]["realized_reference_cosine"] == 0.01
    assert metrics["candidate_pool"][3]["realized_radius"] == 0.041
    assert metrics["pairwise_cosines"][0] == {
        "left": 0,
        "right": 1,
        "cosine": 0.99,
    }

    timing = ScoreTiming(
        elapsed_seconds=2.0,
        generation_seconds=1.5,
        decode_seconds=0.1,
        checker_seconds=0.2,
        batches=2,
        prompt_tokens=8,
        generated_tokens=16,
    )
    assert timing.as_dict()["python_overhead_seconds"] == pytest.approx(0.2)

    class BadGeometry(FakeProposals):
        def geometry(self):
            return [("persistent", 0.75)]

    with pytest.raises(RuntimeError, match="geometry is invalid"):
        proposal_stats(BadGeometry(), 10)

    class ZeroGeometry(FakeProposals):
        def pool_geometry(self):
            return (
                [0.0] * 4,
                [
                    (0, 1, None),
                    (0, 2, None),
                    (0, 3, None),
                    (1, 2, None),
                    (1, 3, None),
                    (2, 3, None),
                ],
                [None] * 4,
            )

    zero = proposal_stats(ZeroGeometry(), 10)
    assert zero["realized_radius"] == 0.0
    assert all(row["cosine"] is None for row in zero["pairwise_cosines"])
    assert zero["realized_reference_cosine"] is None


def test_losspipe():
    from ops.qwen.metal import Evaluator

    calls = []

    class FakeSearch:
        def ask_losses(self, engine, tokens, masks, **kwargs):
            calls.append((engine, tokens, masks, kwargs))
            return "proposal", [0.25, 0.5]

    evaluator = object.__new__(Evaluator)
    evaluator.engine = object()
    tokens = [[1, 2], [3, 4]]
    masks = [[False, True], [False, True]]

    assert evaluator.ask_losses(
        FakeSearch(),
        tokens,
        masks,
        seed=11,
        draw_seed=13,
        arms=1,
        candidates=4,
        neighbors=7,
    ) == ("proposal", [0.25, 0.5])
    assert calls == [
        (
            evaluator.engine,
            tokens,
            masks,
            {"arms": 1, "candidates": 4, "neighbors": 7, "seed": 11, "draw_seed": 13},
        )
    ]


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
