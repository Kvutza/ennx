"""Build immutable Python-centered Stack v3 pretraining streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from array import array
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

SOURCE_ID = "HuggingFaceCode/stack-v3-train"
SOURCE_REVISION = "1f61b735bc0a5698345ce2196730f24bfa467f33"
CORPUS_PRESET = "stack_v3_python_pilot_v1"
MODEL_PRESET = "fbt_pisa1_moe_v1"
MAGIC = b"ENNXPTN1"
VOCAB = 8192
CONTEXT = 4096
SEED = 0x454E4E58
SPECIALS = ("<|unknown|>", "<|endoftext|>", "<|endrepository|>")
QUOTAS = {
    "implementation": 2867,
    "tests": 615,
    "documentation": 410,
    "configuration": 204,
}
MAX_FILE_CHARS = 262_144
MAX_REPOSITORY_CHARS = 2_097_152
CHARS_PER_TOKEN_RESERVE = 6
SHUFFLE_REPOSITORIES = 1
PARQUET_BATCH = 1
COLLECTOR_VERSION = 1
PRESET_SEQUENCES = {"train": 20, "validation": 16, "test": 16}
LOG = logging.getLogger("ennx.pretrain")


@contextmanager
def phase(name: str):
    started = time.monotonic()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(5):
            LOG.info("%s | working | elapsed %.1fs", name, time.monotonic() - started)

    LOG.info("%s | start", name)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException as error:
        LOG.error(
            "%s | failed | elapsed %.1fs | %s",
            name,
            time.monotonic() - started,
            error,
        )
        raise
    else:
        LOG.info("%s | complete | elapsed %.1fs", name, time.monotonic() - started)
    finally:
        stopped.set()
        thread.join()


@dataclass(frozen=True)
class Document:
    repository: str
    commit: str
    content_id: str
    path: str
    bucket: str
    text: str


def repository_split(repository: str, commit: str) -> str:
    digest = hashlib.sha256(f"{repository}\0{commit}".encode()).digest()
    partition = int.from_bytes(digest[:8], "little") % 20
    if partition < 18:
        return "train"
    return "validation" if partition == 18 else "test"


def classify(path: str, language: str) -> str | None:
    lowered = "/" + path.lower().lstrip("/")
    basename = lowered.rsplit("/", 1)[-1]
    if language == "Python":
        if (
            "/test/" in lowered
            or "/tests/" in lowered
            or basename.startswith("test_")
            or basename.endswith("_test.py")
        ):
            return "tests"
        return "implementation"
    if language in {"Markdown", "reStructuredText"} or basename.startswith(
        ("readme", "contributing", "architecture")
    ):
        return "documentation"
    if language in {"Dockerfile", "INI", "JSON", "Makefile", "Shell", "TOML", "YAML"}:
        return "configuration"
    return None


def required_characters(sequences: dict[str, int]) -> dict[str, dict[str, int]]:
    result = {}
    for split, count in sequences.items():
        result[split] = {
            bucket: math.ceil(count * tokens * CHARS_PER_TOKEN_RESERVE)
            for bucket, tokens in QUOTAS.items()
        }
    minimum_train = 8_000_000
    current_train = sum(result["train"].values())
    if current_train < minimum_train:
        for bucket, quota in QUOTAS.items():
            result["train"][bucket] = max(
                result["train"][bucket],
                math.ceil(minimum_train * quota / CONTEXT),
            )
    return result


def collect(source, targets: dict[str, dict[str, int]]):
    documents = {split: defaultdict(list) for split in targets}
    characters = {split: defaultdict(int) for split in targets}
    seen_content = set()
    reported = time.monotonic()
    for scanned, record in enumerate(source, 1):
        if time.monotonic() - reported >= 5:
            quota = sum(
                target for split in targets.values() for target in split.values()
            )
            filled = sum(
                min(characters[split][bucket], target)
                for split, buckets in targets.items()
                for bucket, target in buckets.items()
            )
            LOG.info(
                "collect | repositories scanned %d | documents kept %d | character quota %.1f%%",
                scanned,
                len(seen_content),
                100 * filled / quota,
            )
            reported = time.monotonic()
        repository = record["repo_path"]
        commit = record["commit_id"]
        split = repository_split(repository, commit)
        if split not in targets or record.get("github_metadata", {}).get("is_fork"):
            continue
        repository_characters = 0
        for file in record["files"]:
            if repository_characters >= MAX_REPOSITORY_CHARS:
                break
            if file.get("is_vendor") or file.get("license_type") != "permissive":
                continue
            bucket = classify(file["file_path"], file["language"])
            content_id = file["content_id"]
            text = file["content"]
            if (
                bucket is None
                or characters[split][bucket] >= targets[split][bucket]
                or content_id in seen_content
                or not isinstance(text, str)
                or not 64 <= len(text) <= MAX_FILE_CHARS
                or "\0" in text
            ):
                continue
            remaining = MAX_REPOSITORY_CHARS - repository_characters
            if len(text) > remaining:
                continue
            seen_content.add(content_id)
            documents[split][bucket].append(
                Document(
                    repository, commit, content_id, file["file_path"], bucket, text
                )
            )
            characters[split][bucket] += len(text)
            repository_characters += len(text)
        if all(
            characters[part][bucket] >= targets[part][bucket]
            for part in targets
            for bucket in QUOTAS
        ):
            break
    missing = {
        f"{split}/{bucket}": targets[split][bucket] - characters[split][bucket]
        for split in targets
        for bucket in QUOTAS
        if characters[split][bucket] < targets[split][bucket]
    }
    if missing:
        raise ValueError(
            f"Stack v3 stream ended before corpus quotas were met: {missing}"
        )
    LOG.info(
        "collect | documents kept %d | all character quotas met", len(seen_content)
    )
    return documents, characters


def train_tokenizer(documents):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=SPECIALS[0]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB,
        min_frequency=2,
        show_progress=True,
        special_tokens=list(SPECIALS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        (document.text for bucket in QUOTAS for document in documents["train"][bucket]),
        trainer=trainer,
    )
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if tokenizer.get_vocab_size(with_added_tokens=True) != VOCAB:
        raise ValueError(
            "training corpus did not produce the required 8192-token vocabulary"
        )
    if tuple(tokenizer.token_to_id(token) for token in SPECIALS) != (0, 1, 2):
        raise ValueError("pretraining special-token IDs changed")
    return tokenizer


def write_values(stream, values) -> int:
    if any(not 0 <= token < VOCAB for token in values):
        raise ValueError("tokenizer emitted an out-of-range token")
    encoded = array("H", values)
    if sys.byteorder != "little":
        encoded.byteswap()
    encoded.tofile(stream)
    return len(encoded)


def spool_tokens(tokenizer, documents, directory: Path):
    eot = tokenizer.token_to_id(SPECIALS[1])
    eor = tokenizer.token_to_id(SPECIALS[2])
    paths = {split: {} for split in documents}
    counts = {split: {} for split in documents}
    directory.mkdir()
    for split, buckets in documents.items():
        for bucket in QUOTAS:
            path = directory / f"{split}-{bucket}.u16"
            count = 0
            previous_repository = None
            with path.open("xb") as stream:
                for document in buckets[bucket]:
                    if previous_repository not in (None, document.repository):
                        count += write_values(stream, [eor])
                    encoding = tokenizer.encode(document.text, add_special_tokens=False)
                    count += write_values(stream, encoding.ids)
                    count += write_values(stream, [eot])
                    previous_repository = document.repository
            paths[split][bucket] = path
            counts[split][bucket] = count
            LOG.info("tokenize | %s/%s | %s tokens", split, bucket, f"{count:,}")
    return paths, counts


def pack(
    pools: dict[str, list[int]], sequences: int, seed: int
) -> tuple[list[int], dict]:
    needed = {bucket: sequences * quota for bucket, quota in QUOTAS.items()}
    short = {
        bucket: needed[bucket] - len(pools[bucket])
        for bucket in QUOTAS
        if len(pools[bucket]) < needed[bucket]
    }
    if short:
        raise ValueError(f"token pools are smaller than the requested mixture: {short}")
    offsets = {bucket: 0 for bucket in QUOTAS}
    packed = []
    for sequence in range(sequences):
        order = list(QUOTAS)
        random.Random(seed + sequence).shuffle(order)
        for bucket in order:
            start = offsets[bucket]
            end = start + QUOTAS[bucket]
            packed.extend(pools[bucket][start:end])
            offsets[bucket] = end
    assert len(packed) == sequences * CONTEXT
    return packed, {bucket: offsets[bucket] for bucket in QUOTAS}


def write_stream(path: Path, tokens: list[int], sequences: int) -> str:
    header = MAGIC + VOCAB.to_bytes(4, "little") + CONTEXT.to_bytes(4, "little")
    header += sequences.to_bytes(4, "little") + (0).to_bytes(4, "little")
    with path.open("xb") as stream:
        stream.write(header)
        write_values(stream, tokens)
    return file_digest(path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pack_stream(
    path: Path,
    paths: dict[str, Path],
    counts: dict[str, int],
    sequences: int,
    seed: int,
) -> tuple[str, dict[str, int]]:
    needed = {bucket: sequences * quota for bucket, quota in QUOTAS.items()}
    short = {
        bucket: needed[bucket] - counts[bucket]
        for bucket in QUOTAS
        if counts[bucket] < needed[bucket]
    }
    if short:
        raise ValueError(f"token pools are smaller than the requested mixture: {short}")
    header = MAGIC + VOCAB.to_bytes(4, "little") + CONTEXT.to_bytes(4, "little")
    header += sequences.to_bytes(4, "little") + (0).to_bytes(4, "little")
    streams = {bucket: paths[bucket].open("rb") for bucket in QUOTAS}
    offsets = {bucket: 0 for bucket in QUOTAS}
    try:
        with path.open("xb") as output:
            output.write(header)
            for sequence in range(sequences):
                order = list(QUOTAS)
                random.Random(seed + sequence).shuffle(order)
                for bucket in order:
                    size = QUOTAS[bucket] * 2
                    chunk = streams[bucket].read(size)
                    if len(chunk) != size:
                        raise ValueError(f"token pool ended early: {bucket}")
                    output.write(chunk)
                    offsets[bucket] += QUOTAS[bucket]
    finally:
        for stream in streams.values():
            stream.close()
    return file_digest(path), offsets


def canonical_id(namespace: str, value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(namespace.encode() + b"\0" + payload).hexdigest()[:20]


def corpus_recipe(rounds: int = 10) -> dict:
    return {
        "preset": CORPUS_PRESET,
        "format": "ennx.pretraining.v1",
        "source": {"id": SOURCE_ID, "revision": SOURCE_REVISION},
        "shuffle_repositories": SHUFFLE_REPOSITORIES,
        "parquet_batch_repositories": PARQUET_BATCH,
        "tokenizer": {
            "kind": "byte_level_bpe",
            "vocabulary": VOCAB,
            "package": "tokenizers==0.21.4",
        },
        "datasets_package": "datasets==4.1.1",
        "context": CONTEXT,
        "seed": SEED,
        "sequences": {
            **PRESET_SEQUENCES,
        },
        "mixture_tokens_per_sequence": QUOTAS,
    }


def token_stage(characters: dict) -> dict:
    return {
        "format": "ennx.token-pools.v1",
        "collector_version": COLLECTOR_VERSION,
        "source": {"id": SOURCE_ID, "revision": SOURCE_REVISION},
        "filter": {
            "license_type": "permissive",
            "exclude_forks": True,
            "exclude_vendor": True,
            "max_file_characters": MAX_FILE_CHARS,
            "max_repository_characters": MAX_REPOSITORY_CHARS,
        },
        "required_characters": characters,
        "tokenizer": {
            "kind": "byte_level_bpe",
            "vocabulary": VOCAB,
            "package": "tokenizers",
            "package_version": "0.21.4",
            "special_tokens": list(SPECIALS),
        },
        "seed": SEED,
        "shuffle_repositories": SHUFFLE_REPOSITORIES,
        "parquet_batch_repositories": PARQUET_BATCH,
        "mixture_tokens_per_sequence": QUOTAS,
    }


def plan_study(document: dict, root: Path, source: str) -> dict:
    document = lower_study(document)
    if document.get("version") != 1:
        raise ValueError("version must be 1")
    if document.get("study") != "pretrain":
        raise ValueError("resolved corpus studies require study = 'pretrain'")
    if document.get("model") != MODEL_PRESET:
        raise ValueError(f"model must be {MODEL_PRESET!r}")
    if document.get("corpus") != CORPUS_PRESET:
        raise ValueError(f"corpus must be {CORPUS_PRESET!r}")
    forbidden = sorted({"output", "dataset"}.intersection(document))
    if forbidden:
        raise ValueError(
            f"pretrain paths are generated automatically; remove {', '.join(forbidden)}"
        )
    rounds = document.get("rounds", 3)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    structured = {
        key: value for key, value in document.items() if isinstance(value, (dict, list))
    }
    if set(structured) - {"reliability_controller"} or any(
        not isinstance(value, dict)
        or any(isinstance(field, (dict, list)) for field in value.values())
        for value in structured.values()
    ):
        raise ValueError(
            "pretrain version 1 accepts scalars and the reliability_controller table"
        )

    recipe = corpus_recipe(rounds)
    corpus_id = canonical_id("ennx-corpus-v1", recipe)
    characters = required_characters(recipe["sequences"])
    stage_recipe = token_stage(characters)
    stage_id = canonical_id("ennx-token-pools-v1", stage_recipe)
    experiment = {
        "config": document,
        "corpus_id": corpus_id,
        "stage_id": stage_id,
        "source": source,
    }
    experiment_id = canonical_id("ennx-experiment-v1", experiment)
    cache = root / ".cache" / "ennx"
    return {
        "corpus_id": corpus_id,
        "experiment_id": experiment_id,
        "corpus": cache / "corpora" / corpus_id,
        "stage": cache / "token-pools" / stage_id,
        "output": cache / "runs" / "pretrain" / experiment_id,
        "resolved": cache / "studies" / f"{experiment_id}.toml",
        "recipe": recipe,
    }


def scalar_section(document: dict, section: str) -> dict:
    value = document.pop(section, None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        document[section] = value
        return {}
    return dict(value)


def move_absent(document: dict, section: str, key: str, value) -> None:
    if key in document:
        raise ValueError(f"{section}.{key} conflicts with top-level {key}")
    document[key] = value


def single_subsection(section: dict, name: str):
    subtables = [key for key, value in section.items() if isinstance(value, dict)]
    if not subtables:
        return None
    if len(subtables) != 1:
        raise ValueError(f"{name} must select exactly one primitive")
    key = subtables[0]
    value = section.pop(key)
    return key, dict(value)


def normalize_primitive(name: str) -> str:
    return name.replace("-", "_")


def scalar_section_alias(document: dict, public: str, legacy: str) -> dict:
    public_value = scalar_section(document, public)
    legacy_value = scalar_section(document, legacy)
    if public_value and legacy_value:
        raise ValueError(f"{public} conflicts with {legacy}")
    return public_value or legacy_value


def lower_study(raw: dict) -> dict:
    document = dict(raw)
    pretrain = scalar_section(document, "pretrain")
    if pretrain:
        move_absent(document, "pretrain", "study", "pretrain")
        for key, value in pretrain.items():
            move_absent(document, "pretrain", key, value)
    study = scalar_section(document, "study")
    if study:
        kind = study.pop("kind", None)
        if kind is not None:
            move_absent(document, "study", "study", kind)
        for key, value in study.items():
            move_absent(document, "study", key, value)
    rounds = scalar_section(document, "rounds")
    if "count" in rounds:
        move_absent(document, "rounds", "rounds", rounds.pop("count"))
    if "target_ms" in rounds:
        move_absent(document, "rounds", "target_round_ms", rounds.pop("target_ms"))
    for key, value in rounds.items():
        move_absent(document, "rounds", key, value)
    experiment = scalar_section(document, "experiment")
    for key, value in experiment.items():
        move_absent(document, "experiment", key, value)
    perturbation = scalar_section(document, "perturbation")
    distribution = perturbation.pop("distribution", None)
    if distribution is not None:
        move_absent(document, "perturbation", "perturbation", distribution)
    selected = single_subsection(perturbation, "perturbation")
    if selected is not None:
        name, fields = selected
        if fields:
            raise ValueError(f"perturbation.{name} does not accept fields")
        move_absent(document, "perturbation", "perturbation", name)
    for key, value in perturbation.items():
        move_absent(document, "perturbation", key, value)
    proposal = scalar_section(document, "proposal")
    distribution = proposal.pop("distribution", None)
    if distribution is not None:
        move_absent(document, "proposal", "perturbation", distribution)
    for key, value in proposal.items():
        move_absent(document, "proposal", key, value)
    acquisition = scalar_section(document, "acquisition")
    method = acquisition.pop("method", None)
    if method is not None:
        move_absent(document, "acquisition", "acquisition", method)
    selected = single_subsection(acquisition, "acquisition")
    if selected is not None:
        name, fields = selected
        move_absent(document, "acquisition", "acquisition", name)
        for key, value in fields.items():
            move_absent(document, "acquisition", key, value)
    kind = acquisition.pop("kind", None)
    if kind is not None:
        move_absent(document, "acquisition", "acquisition", kind)
    for key, value in acquisition.items():
        move_absent(document, "acquisition", key, value)
    surrogate = scalar_section(document, "surrogate")
    method = surrogate.pop("method", None)
    if method is not None and method not in {"enn", "resident_enn"}:
        raise ValueError("surrogate.method must be 'enn'")
    if "fit_candidates" in surrogate:
        move_absent(
            document,
            "surrogate",
            "num_candidates",
            surrogate.pop("fit_candidates"),
        )
    if "fit_samples" in surrogate:
        move_absent(document, "surrogate", "num_samples", surrogate.pop("fit_samples"))
    selected = single_subsection(surrogate, "surrogate")
    if selected is not None:
        name, fields = selected
        if normalize_primitive(name) not in {"enn", "resident_enn"}:
            raise ValueError("surrogate must select method = 'enn'")
        fit = scalar_section(fields, "fit")
        for key, value in fit.items():
            target = {"candidates": "num_candidates", "samples": "num_samples"}.get(key)
            if target is None:
                raise ValueError(f"unknown field surrogate.enn.fit.{key}")
            move_absent(document, "surrogate.enn.fit", target, value)
        for key, target in {
            "neighbors": "k_neighbors",
            "candidates": "num_candidates",
            "samples": "num_samples",
        }.items():
            if key in fields:
                move_absent(document, "surrogate", target, fields.pop(key))
        for key, value in fields.items():
            move_absent(document, "surrogate", key, value)
    kind = surrogate.pop("kind", None)
    if kind is not None and kind != "resident_enn":
        raise ValueError("surrogate.kind must be 'resident_enn'")
    for key, target in {
        "neighbors": "k_neighbors",
        "candidates": "num_candidates",
        "samples": "num_samples",
    }.items():
        if key in surrogate:
            move_absent(document, "surrogate", target, surrogate.pop(key))
    for key, value in surrogate.items():
        move_absent(document, "surrogate", key, value)
    trust_region = scalar_section_alias(document, "trust-region", "trust_region")
    reliability = trust_region.pop("reliability", None)
    if reliability is not None:
        if not isinstance(reliability, dict):
            raise ValueError("trust-region.reliability must be a table")
        move_absent(
            document,
            "trust-region.reliability",
            "reliability_controller",
            reliability,
        )
    method = trust_region.pop("method", None)
    if method is not None:
        move_absent(document, "trust_region", "trust_region_kind", method)
    selected = single_subsection(trust_region, "trust_region")
    if selected is not None:
        name, fields = selected
        move_absent(
            document, "trust_region", "trust_region_kind", normalize_primitive(name)
        )
        shape = fields.pop("shape", None)
        if shape is not None:
            move_absent(document, "trust_region", "trust_region_shape", shape)
        for key, value in fields.items():
            move_absent(document, "trust_region", key, value)
    kind = trust_region.pop("kind", None)
    if kind is not None:
        move_absent(document, "trust_region", "trust_region_kind", kind)
    shape = trust_region.pop("shape", None)
    if shape is not None:
        move_absent(document, "trust_region", "trust_region_shape", shape)
    if trust_region.pop("tensor_family", None) is not None:
        raise ValueError(
            "trust_region.tensor_family is reserved for learned/static shape values but is not configurable yet"
        )
    for key, value in trust_region.items():
        move_absent(document, "trust_region", key, value)
    return document


def source_identity(root: Path) -> str:
    result = subprocess.run(
        ["jj", "log", "-r", "@", "--no-graph", "-T", "commit_id"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("jj could not determine the working-copy source identity")
    return result.stdout.strip()


def toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, dict):
        fields = ", ".join(f"{key} = {toml_value(value[key])}" for key in sorted(value))
        return f"{{ {fields} }}"
    raise ValueError(f"unsupported pretrain configuration value: {value!r}")


def resolved_toml(document: dict, plan: dict) -> str:
    resolved = dict(document)
    resolved["dataset"] = str((plan["corpus"] / "train.ennxptn").resolve())
    resolved["output"] = str(plan["output"].resolve())
    keys = ["version"] + sorted(key for key in resolved if key != "version")
    return "".join(f"{key} = {toml_value(resolved[key])}\n" for key in keys)


def resolve_study(config: Path, root: Path) -> Path:
    document = lower_study(tomllib.loads(config.read_text()))
    plan = plan_study(document, root.resolve(), source_identity(root))
    plan["resolved"].parent.mkdir(parents=True, exist_ok=True)
    log_path = plan["resolved"].with_suffix(".prepare.log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s [pretrain] %(message)s"))
    LOG.addHandler(handler)
    try:
        LOG.info("corpus %s | preparation log %s", plan["corpus_id"], log_path)
        return resolve_plan(document, plan)
    except Exception as error:
        LOG.error("resolution failed | %s", error)
        raise
    finally:
        LOG.removeHandler(handler)
        handler.close()


def resolve_plan(document: dict, plan: dict) -> Path:
    corpus = plan["corpus"]
    manifest_path = corpus / "manifest.json"
    if corpus.exists():
        LOG.info("cache | hit | %s", corpus)
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid cached corpus {corpus}: {error}") from error
        if manifest.get("recipe_id") != plan["corpus_id"]:
            raise ValueError(
                f"cached corpus recipe does not match its identity: {corpus}"
            )
    else:
        sequences = plan["recipe"]["sequences"]
        LOG.info(
            "cache | miss | training batches %d | context %d",
            sequences["train"] // 2,
            CONTEXT,
        )
        prepare(
            corpus,
            sequences["train"],
            sequences["validation"],
            sequences["test"],
            recipe_id=plan["corpus_id"],
            stage=plan["stage"],
        )
    plan["resolved"].parent.mkdir(parents=True, exist_ok=True)
    temporary = plan["resolved"].with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(resolved_toml(document, plan))
    os.replace(temporary, plan["resolved"])
    LOG.info(
        "ready | rounds %d | study %s", document.get("rounds", 3), plan["resolved"]
    )
    return plan["resolved"]


def prepare(
    output: Path,
    train_sequences: int,
    validation_sequences: int,
    test_sequences: int,
    recipe_id: str | None = None,
    stage: Path | None = None,
):
    sequences = {
        "train": train_sequences,
        "validation": validation_sequences,
        "test": test_sequences,
    }
    if any(count < 2 or count % 2 for count in sequences.values()):
        raise ValueError("every split requires a positive even sequence count")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    characters_required = required_characters(sequences)
    stage_recipe = token_stage(characters_required)
    stage_id = canonical_id("ennx-token-pools-v1", stage_recipe)
    stage = (stage or output.parent / "token-pools" / stage_id).resolve()
    stage.parent.mkdir(parents=True, exist_ok=True)

    def load_stage():
        metadata_path = stage / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid token-pool cache {stage}: {error}") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid token-pool cache metadata: {stage}")
        if (
            metadata.get("stage_id") != stage_id
            or metadata.get("recipe") != stage_recipe
        ):
            raise ValueError(f"token-pool cache identity does not match: {stage}")
        tokenizer_path = stage / "tokenizer.json"
        if not tokenizer_path.is_file() or file_digest(tokenizer_path) != metadata.get(
            "tokenizer_sha256"
        ):
            raise ValueError(
                f"tokenizer is missing or corrupt in token-pool cache: {stage}"
            )
        pool_records = metadata.get("pools")
        split_records = metadata.get("splits")
        if not isinstance(pool_records, dict) or not isinstance(split_records, dict):
            raise ValueError(f"invalid token-pool cache metadata: {stage}")
        paths, counts = (
            {split: {} for split in sequences},
            {split: {} for split in sequences},
        )
        for split in sequences:
            for bucket in QUOTAS:
                key = f"{split}/{bucket}"
                record = pool_records.get(key)
                path = stage / "pools" / f"{split}-{bucket}.u16"
                if (
                    not isinstance(record, dict)
                    or type(record.get("count")) is not int
                    or record["count"] < 0
                    or not path.is_file()
                    or path.stat().st_size != record["count"] * 2
                    or file_digest(path) != record.get("sha256")
                ):
                    raise ValueError(f"token pool is missing or corrupt: {path}")
                paths[split][bucket] = path
                counts[split][bucket] = record["count"]
            if not isinstance(split_records.get(split), dict):
                raise ValueError(f"invalid token-pool split metadata: {stage}/{split}")
        return metadata, paths, counts

    def create_stage():
        from ops.corpus import collect_native

        with phase("collect and filter"):
            documents, characters = collect_native(required_characters(sequences))
        with phase("train tokenizer"):
            tokenizer = train_tokenizer(documents)
        stage_temporary = Path(
            tempfile.mkdtemp(prefix=f".{stage.name}.", dir=stage.parent)
        )
        try:
            with phase("tokenize"):
                pool_paths, pool_counts = spool_tokens(
                    tokenizer, documents, stage_temporary / "pools"
                )
            tokenizer_path = stage_temporary / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            pool_metadata = {}
            for split in sequences:
                for bucket in QUOTAS:
                    path = pool_paths[split][bucket]
                    pool_metadata[f"{split}/{bucket}"] = {
                        "count": pool_counts[split][bucket],
                        "sha256": file_digest(path),
                    }
            split_metadata = {}
            for split in sequences:
                rows = [
                    document
                    for bucket in QUOTAS
                    for document in documents[split][bucket]
                ]
                split_metadata[split] = {
                    "collected_characters": dict(characters[split]),
                    "documents": len(rows),
                    "repositories": len({document.repository for document in rows}),
                }
            metadata = {
                "stage_id": stage_id,
                "recipe": stage_recipe,
                "tokenizer_sha256": file_digest(tokenizer_path),
                "tokenizer_package_version": version("tokenizers"),
                "collector_package_version": "native",
                "pools": pool_metadata,
                "splits": split_metadata,
            }
            (stage_temporary / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            try:
                os.replace(stage_temporary, stage)
            except OSError:
                if not stage.exists():
                    raise
            finally:
                if stage_temporary.exists():
                    shutil.rmtree(stage_temporary)
        except BaseException:
            shutil.rmtree(stage_temporary, ignore_errors=True)
            raise

    if stage.exists():
        LOG.info("token pools | hit | %s", stage)
    else:
        LOG.info("token pools | miss | %s", stage)
        create_stage()
    stage_metadata, pool_paths, pool_counts = load_stage()

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copyfile(stage / "tokenizer.json", temporary / "tokenizer.json")
        manifest = {
            "format": "ennx.pretraining.v1",
            "recipe_id": recipe_id,
            "stage_id": stage_id,
            "source": {"id": SOURCE_ID, "revision": SOURCE_REVISION},
            "filter": {
                "license_type": "permissive",
                "exclude_forks": True,
                "exclude_vendor": True,
                "max_file_characters": MAX_FILE_CHARS,
                "max_repository_characters": MAX_REPOSITORY_CHARS,
            },
            "split": "sha256(repository + NUL + commit) modulo 20: train 0..17, validation 18, test 19",
            "mixture_tokens_per_sequence": QUOTAS,
            "tokenizer": {
                "kind": "byte_level_bpe",
                "vocabulary": VOCAB,
                "special_tokens": dict(zip(SPECIALS, range(len(SPECIALS)))),
                "package_version": stage_metadata["tokenizer_package_version"],
                "sha256": stage_metadata["tokenizer_sha256"],
            },
            "seed": SEED,
            "splits": {},
        }
        for index, (split, count) in enumerate(sequences.items()):
            path = temporary / f"{split}.ennxptn"
            with phase(f"pack {split} ({count} sequences)"):
                digest, mixture = pack_stream(
                    path,
                    pool_paths[split],
                    pool_counts[split],
                    count,
                    SEED + index * 1_000_000,
                )
            split_metadata = stage_metadata["splits"][split]
            manifest["splits"][split] = {
                "sequences": count,
                "batches": count // 2,
                "processed_tokens": count * CONTEXT,
                "causal_targets": count * (CONTEXT - 1),
                "mixture_tokens": mixture,
                **split_metadata,
                "sha256": digest,
            }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            raise ValueError(f"output already exists: {output}")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--train-sequences", type=int, default=4096)
    result.add_argument("--validation-sequences", type=int, default=256)
    result.add_argument("--test-sequences", type=int, default=256)
    return result


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="[pretrain] %(message)s")
    LOG.setLevel(logging.INFO)
    if len(sys.argv) >= 2 and sys.argv[1] == "resolve":
        resolve = argparse.ArgumentParser(
            description="Resolve a path-free pretrain study."
        )
        resolve.add_argument("config", type=Path)
        arguments = resolve.parse_args(sys.argv[2:])
        try:
            path = resolve_study(arguments.config, Path.cwd())
        except (OSError, ValueError) as error:
            resolve.error(str(error))
        print(path)
        return
    arguments = parser().parse_args()
    try:
        manifest = prepare(
            arguments.output,
            arguments.train_sequences,
            arguments.validation_sequences,
            arguments.test_sequences,
        )
    except (OSError, ValueError) as error:
        parser().error(str(error))
    print(json.dumps(manifest["splits"], sort_keys=True))


if __name__ == "__main__":
    main()
