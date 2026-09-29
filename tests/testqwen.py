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
