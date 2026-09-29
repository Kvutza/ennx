"""Bounded, immutable FineWeb pilot for the native corpus-prefix NLL scorer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

PRESET = "fineweb_10bt_pilot_v1"


def recipe():
    return {
        "preset": PRESET,
        "format": "ennx.pretraining.v1",
        "source": {
            "id": "HuggingFaceFW/fineweb",
            "revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
            "subset": "sample-10BT",
            "directory": "sample/10BT",
        },
        "split_policy": "sha256_lowercase_url_host_mod20_v1",
        "deduplication": "exact_utf8_sha256_across_splits",
        "filter": {"min_characters": 64, "max_characters": 262144},
        "tokenizer": {
            "kind": "byte_level_bpe",
            "vocabulary": 8192,
            "package": "tokenizers==0.21.4",
        },
        "datasets_package": "datasets==4.1.1",
        "reader": "parquet_file_projected_single_thread_v1",
        "context": 4096,
        "sequences": {"train": 20, "validation": 16, "test": 16},
        "required_characters": {
            "train": 8_000_000,
            "validation": 393216,
            "test": 393216,
        },
        "seed_policy": "sha256_corpus_recipe",
    }


def partition(url):
    host = urlsplit(url).hostname
    if not host:
        raise ValueError("FineWeb document requires a URL host")
    identity = host.lower().removeprefix("www.")
    digest = hashlib.sha256(identity.encode()).digest()
    bucket = int.from_bytes(digest[:8], "little") % 20
    return (
        identity,
        "train" if bucket < 18 else "validation" if bucket == 18 else "test",
    )


def collect(source, profile):
    from ops.pretrain import Document, LOG

    targets = profile["required_characters"]
    documents = {split: {"web": []} for split in targets}
    characters = dict.fromkeys(targets, 0)
    seen = set()
    for row in source:
        text = row.get("text")
        if (
            not isinstance(text, str)
            or "\0" in text
            or not profile["filter"]["min_characters"]
            <= len(text)
            <= profile["filter"]["max_characters"]
        ):
            continue
        host, split = partition(row["url"])
        identity = hashlib.sha256(text.encode()).hexdigest()
        if identity in seen or characters[split] >= targets[split]:
            continue
        seen.add(identity)
        documents[split]["web"].append(
            Document(
                host, profile["source"]["revision"], identity, row["url"], "web", text
            )
        )
        characters[split] += len(text)
        if all(characters[key] >= target for key, target in targets.items()):
            LOG.info(
                "FineWeb | collected characters %s | documents %d",
                characters,
                len(seen),
            )
            return documents, characters
    raise ValueError(f"FineWeb source exhausted before character quotas: {characters}")


def stream(profile):
    import pyarrow.parquet as parquet
    from huggingface_hub import HfApi, HfFileSystem

    source = profile["source"]
    paths = HfApi().list_repo_files(
        source["id"], repo_type="dataset", revision=source["revision"]
    )
    filesystem = HfFileSystem()
    for path in sorted(paths):
        if not path.startswith(source["directory"] + "/") or not path.endswith(
            ".parquet"
        ):
            continue
        name = f"datasets/{source['id']}@{source['revision']}/{path}"
        with filesystem.open(name, "rb") as stream:
            reader = parquet.ParquetFile(stream, pre_buffer=False)
            for batch in reader.iter_batches(
                batch_size=4096, columns=["text", "url"], use_threads=False
            ):
                yield from batch.to_pylist()


def write_corpus(directory, profile, documents, characters, identity):
    from ops import pretrain

    tokenizer = pretrain.train_tokenizer(documents, ("web",))
    tokenizer.save(str(directory / "tokenizer.json"))
    paths, counts = pretrain.spool_tokens(
        tokenizer, documents, directory / "pools", ("web",)
    )
    manifest = {
        "format": profile["format"],
        "recipe_id": identity,
        "recipe": profile,
        "objective": "corpus_prefix_next_token_cross_entropy",
        "teacher_forcing": True,
        "generation_in_loop": False,
        "official_nanogpt_comparable": False,
        "tokenizer_sha256": pretrain.file_digest(directory / "tokenizer.json"),
        "splits": {},
    }
    seed = int(pretrain.canonical_id("ennx-fineweb-packing-v1", profile)[:16], 16)
    for split, count in profile["sequences"].items():
        split_seed = int.from_bytes(
            hashlib.sha256(f"{seed}/{split}".encode()).digest()[:8], "little"
        )
        digest, mixture = pretrain.pack_stream(
            directory / f"{split}.ennxptn",
            paths[split],
            counts[split],
            count,
            split_seed,
            {"web": profile["context"]},
        )
        manifest["splits"][split] = {
            "sequences": count,
            "processed_tokens": count * profile["context"],
            "causal_targets": count * (profile["context"] - 1),
            "sha256": digest,
            "documents": len(documents[split]["web"]),
            "hosts": len({doc.repository for doc in documents[split]["web"]}),
            "collected_characters": characters[split],
            "mixture_tokens": mixture,
        }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def prepare(output: Path, profile, identity):
    from ops.pretrain import phase

    output.parent.mkdir(parents=True, exist_ok=True)
    with phase("stream bounded FineWeb corpus"):
        iterator = stream(profile)
        try:
            documents, characters = collect(iterator, profile)
        finally:
            iterator.close()
    directory = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        with phase("tokenize and pack FineWeb"):
            manifest = write_corpus(directory, profile, documents, characters, identity)
        if output.exists():
            raise ValueError(f"output already exists: {output}")
        os.replace(directory, output)
        return manifest
    finally:
        if directory.exists():
            shutil.rmtree(directory)


def validate_cache(directory, manifest, profile):
    from ops.pretrain import file_digest

    if manifest.get("recipe") != profile:
        raise ValueError("FineWeb cached recipe does not match the pinned study")
    splits = manifest.get("splits", {})
    if set(splits) != set(profile["sequences"]) or any(
        splits[split].get("sequences") != count
        for split, count in profile["sequences"].items()
    ):
        raise ValueError("FineWeb cache has missing or unexpected split sizes")
    files = {"tokenizer.json": manifest["tokenizer_sha256"]}
    files.update(
        {f"{split}.ennxptn": record["sha256"] for split, record in splits.items()}
    )
    for name, expected in files.items():
        path = directory / name
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"FineWeb cache is missing or corrupt: {path}")
