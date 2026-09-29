import pytest

from ops import pretrain


def test_tampering(tmp_path):
    recipe = pretrain.corpus_recipe()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    tokenizer = corpus / "tokenizer.json"
    tokenizer.write_text("tokenizer")
    splits = {}
    for split, count in recipe["sequences"].items():
        path = corpus / f"{split}.ennxptn"
        path.write_bytes(split.encode())
        splits[split] = {
            "sequences": count,
            "sha256": pretrain.file_digest(path),
        }
    manifest = {
        "format": "ennx.pretraining.v1",
        "source": recipe["source"],
        "split_policy": recipe["split_policy"],
        "tokenizer": {"sha256": pretrain.file_digest(tokenizer)},
        "splits": splits,
    }
    pretrain.validate_cache(corpus, manifest, recipe)
    (corpus / "validation.ennxptn").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="missing or corrupt"):
        pretrain.validate_cache(corpus, manifest, recipe)
