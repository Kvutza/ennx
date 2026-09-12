"""Minimal tokenizer loader for the pinned Qwen base checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config


def load(directory: Path):
    try:
        from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    except ImportError as error:
        raise RuntimeError(
            "Qwen generation requires tokenizers; install ops/qwen/requirements.txt"
        ) from error
    directory = Path(directory)
    tokenizer_json = directory / "tokenizer.json"
    if tokenizer_json.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
    else:
        vocab = directory / "vocab.json"
        merges = directory / "merges.txt"
        if not vocab.exists() or not merges.exists():
            raise ValueError(
                "Qwen tokenizer requires tokenizer.json or vocab.json and merges.txt"
            )
        tokenizer = Tokenizer(
            models.BPE.from_file(str(vocab), str(merges), unk_token=None)
        )
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer_config = directory / "tokenizer_config.json"
        if not tokenizer_config.exists():
            raise ValueError("Qwen tokenizer_config.json is required with vocab/merges")
        data = json.loads(tokenizer_config.read_text())
        added = data.get("added_tokens_decoder", {})
        if not isinstance(added, dict):
            raise ValueError("Qwen tokenizer added-token metadata is invalid")
        for raw_id, record in sorted(added.items(), key=lambda item: int(item[0])):
            if not isinstance(record, dict) or not isinstance(
                record.get("content"), str
            ):
                raise ValueError("Qwen tokenizer added-token record is invalid")
            expected_id = int(raw_id)
            added_token = AddedToken(
                record["content"],
                single_word=bool(record.get("single_word", False)),
                lstrip=bool(record.get("lstrip", False)),
                rstrip=bool(record.get("rstrip", False)),
                normalized=bool(record.get("normalized", True)),
                special=bool(record.get("special", False)),
            )
            if record.get("special", False):
                tokenizer.add_special_tokens([added_token])
            else:
                tokenizer.add_tokens([added_token])
            if tokenizer.token_to_id(record["content"]) != expected_id:
                raise ValueError("Qwen tokenizer added-token IDs are not contiguous")
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tokenizer_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_size > Config().vocab_size:
        raise ValueError("Qwen tokenizer vocabulary exceeds the model config")
    if any(
        token_id >= Config().vocab_size for token_id in tokenizer.get_vocab().values()
    ):
        raise ValueError("Qwen tokenizer contains an out-of-range token ID")
    return tokenizer
