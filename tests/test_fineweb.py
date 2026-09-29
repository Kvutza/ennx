import tomllib
from pathlib import Path

import pytest

from ops import fineweb, pretrain


def host(split):
    return next(
        f"host-{index}.example"
        for index in range(1000)
        if fineweb.partition(f"https://host-{index}.example/a")[1] == split
    )


def test_dedup():
    profile = fineweb.recipe()
    profile["required_characters"] = dict.fromkeys(profile["sequences"], 64)
    rows = [
        {"url": f"https://{host(split)}/document", "text": split * 100}
        for split in profile["sequences"]
    ]
    # A duplicate on another host cannot be admitted to a different split.
    rows.insert(
        1, {"url": f"https://{host('validation')}/duplicate", "text": rows[0]["text"]}
    )
    documents, characters = fineweb.collect(rows, profile)
    assert all(count >= 64 for count in characters.values())
    hashes = [
        doc.content_id for buckets in documents.values() for doc in buckets["web"]
    ]
    assert len(hashes) == len(set(hashes)) == 3
    for split in profile["sequences"]:
        name = host(split)
        assert fineweb.partition(f"https://www.{name}/other")[1] == split
        assert fineweb.partition(f"http://{name}/different")[1] == split


def test_exhaustion():
    with pytest.raises(ValueError, match="exhausted"):
        fineweb.collect([], fineweb.recipe())


def test_corruption(tmp_path):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    stream = tmp_path / "train.ennxptn"
    stream.write_bytes(b"original")
    profile = fineweb.recipe()
    splits = {}
    for split, count in profile["sequences"].items():
        path = tmp_path / f"{split}.ennxptn"
        path.write_bytes(b"original")
        splits[split] = {"sha256": pretrain.file_digest(path), "sequences": count}
    manifest = {
        "recipe": profile,
        "tokenizer_sha256": pretrain.file_digest(tokenizer),
        "splits": splits,
    }
    fineweb.validate_cache(tmp_path, manifest, profile)
    stream.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="corrupt"):
        fineweb.validate_cache(tmp_path, manifest, profile)
    del manifest["splits"]["validation"]
    with pytest.raises(ValueError, match="missing or unexpected split sizes"):
        fineweb.validate_cache(tmp_path, manifest, profile)


def test_packing(tmp_path):
    profile = fineweb.recipe()
    tokens = list(range(16)) * 512
    path = tmp_path / "web.u16"
    with path.open("xb") as stream:
        count = pretrain.write_values(stream, tokens)
    output = tmp_path / "train.ennxptn"
    seed = int(pretrain.canonical_id("test-fineweb-pack", profile)[:16], 16)
    digest, mixture = pretrain.pack_stream(
        output, {"web": path}, {"web": count}, 2, seed, {"web": 4096}
    )
    assert output.read_bytes()[24:] == path.read_bytes()
    assert mixture == {"web": 8192}
    assert digest == pretrain.file_digest(output)


def test_pairing(tmp_path):
    # Rust supplies this validated concrete request, not the authored schema.
    base = {
        "version": 1,
        "study": "pretrain",
        "model": pretrain.MODEL_PRESET,
        "corpus": fineweb.PRESET,
        "rounds": 12,
    }
    configs = [
        pretrain.lower_study({**base, "selection": selection})
        for selection in ("enn", "random")
    ]
    first, second = [
        pretrain.plan_study(config, tmp_path, "source") for config in configs
    ]
    assert first["corpus"] == second["corpus"]
    assert first["output"] != second["output"]
    assert {key for key in configs[0] if configs[0][key] != configs[1][key]} == {
        "selection"
    }
    resolved = tomllib.loads(pretrain.resolved_toml(configs[0], first))
    assert "generation" not in resolved
    assert resolved["validation_dataset"] == str(first["corpus"] / "validation.ennxptn")
    assert not any(key.endswith("seed") for key in configs[0])


def test_labeling(tmp_path):
    config = {
        "version": 1,
        "study": "pretrain",
        "model": pretrain.MODEL_PRESET,
        "corpus": fineweb.PRESET,
        "generation": {
            "max_tokens": 4096,
            "corpus_prompt_tokens": 128,
            "reward": {"kind": "code_reconstruction"},
        },
    }
    with pytest.raises(ValueError, match="corpus-prefix NLL"):
        pretrain.plan_study(config, tmp_path, "source")
