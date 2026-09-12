"""Prepare a fixed MBPP TRAIN solution-token objective; never execute references.

Run: python -m ops.flame.coding --output objective.json --count 8 --max-tokens 256
Dependencies: existing click/requests plus tokenizers==0.21.4 (no transformers,
datasets, remote Python, Hugging Face credentials, or model weights required).

Tokenizer evidence, inspected without unpickling the checkpoint:
https://huggingface.co/CMU-FLAME/FLAME-MoE-290M-1.3B/blob/04bded20e1eafa97c5c84cee9522798a2e83606f/iter_0005473/common.pt
records tokenizer_type=HuggingFaceTokenizer, tokenizer_model=EleutherAI/pythia-12b,
and padded_vocab_size=50304. Its SHA256 is CHECKPOINT_ARGS_SHA256 below.
The pinned Megatron implementation (REFERENCE in config.py),
megatron/training/tokenizer/tokenizer.py::_HuggingFaceTokenizer, delegates to
AutoTokenizer and takes eod from eos_token_id. Pythia's pinned tokenizer_config
names GPTNeoXTokenizer, add_prefix_space=false, and EOS=<|endoftext|>; tokenizer.json
assigns EOS ID 0. The checkpoint names the tokenizer but saves no tokenizer
revision: TOKENIZER_REVISION is our explicit reproducibility pin, not a claim
that a tokenizer artifact hash was saved during FLAME training.

The pinned HF MBPP 1.0.2/full loader defines TRAIN as task_id 601..974. Its
dataset_infos.json records the exact upstream JSON checksum enforced here.
We fetch that JSON at an immutable Google source revision, implement only the
published TRAIN filter, and never execute the HF loader. References are the
dataset's supplied code, not execution-verified solutions. BigCodeBench-Hard
is reserved for evaluation and is never downloaded or used for selection.
This separation is not a claim about FLAME pretraining contamination.

Prompt and unmodified reference code are concatenated and encoded together.
Only offset trimming is disabled (this does not change token IDs). Reject
tokens crossing the character boundary and require the standalone prompt IDs
to match the concatenated prefix. No BOS, padding, or truncation is applied.
Append exactly one EOS, scored as part of the solution; token zero and all
prompt tokens are unscored. Length limits include EOS. Token counts report both
stored tokens and shifted model inputs (len(tokens)-1 per example).
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import click
import requests

from .config import ITERATION, MODEL_ID, REFERENCE, REVISION, Config

DATASET_ID = "google-research-datasets/mbpp"
DATASET_REVISION = "5a8a3b632e28582ab85087da984bef822e34e415"
DATASET_SOURCE_REVISION = "f82046ba5aabbbb427dbfd38a254d26bff08b533"
DATASET_SHA256 = "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"
DATASET_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    f"{DATASET_SOURCE_REVISION}/mbpp/mbpp.jsonl"
)
DATASET_INFO_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
    f"{DATASET_REVISION}/dataset_infos.json"
)
TOKENIZER_ID = "EleutherAI/pythia-12b"
TOKENIZER_REVISION = "bb1e3e710cdf6b524461d543cfb5ba773f0a81b6"
TOKENIZER_SHA256 = "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"
TOKENIZER_URL = (
    f"https://huggingface.co/{TOKENIZER_ID}/resolve/{TOKENIZER_REVISION}/tokenizer.json"
)
CHECKPOINT_ARGS_SHA256 = (
    "c21cde6bd4b53d4ef143826a68f151327b7a8474b2bec50a07be6bfc169ffdee"
)
EOS_ID = 0
TRAIN_IDS = range(601, 975)
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    return sha256(text.encode("utf-8"))


def read_public(url: str) -> bytes:
    """Bound public downloads and ignore ambient netrc authentication."""
    with requests.Session() as session:
        session.trust_env = False
        with session.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            chunks = []
            size = 0
            for chunk in response.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Public artifact exceeds download size limit")
                chunks.append(chunk)
            return b"".join(chunks)


def load_train() -> list[dict]:
    info = json.loads(read_public(DATASET_INFO_URL))["full"]
    checksum = next(iter(info["download_checksums"].values()))["checksum"]
    if checksum != DATASET_SHA256 or info["splits"]["train"]["num_examples"] != 374:
        raise ValueError("Pinned HF MBPP metadata does not match the TRAIN contract")
    data = read_public(DATASET_URL)
    if sha256(data) != DATASET_SHA256:
        raise ValueError("MBPP source SHA256 mismatch")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    train = [row for row in rows if row["task_id"] in TRAIN_IDS]
    if sorted(row["task_id"] for row in train) != list(TRAIN_IDS):
        raise ValueError("MBPP TRAIN IDs differ from the pinned split")
    return train


def load_tokenizer():
    try:
        from tokenizers import Tokenizer, processors
    except ImportError as error:
        raise ValueError(
            "Install tokenizers==0.21.4 to prepare coding tokens"
        ) from error
    data = read_public(TOKENIZER_URL)
    if sha256(data) != TOKENIZER_SHA256:
        raise ValueError("Pythia tokenizer SHA256 mismatch")
    tokenizer = Tokenizer.from_str(data.decode("utf-8"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    if tokenizer.token_to_id("<|endoftext|>") != EOS_ID:
        raise ValueError("Pythia EOS ID does not match checkpoint tokenizer")
    if tokenizer.get_vocab_size() != 50277:
        raise ValueError("Pythia tokenizer vocabulary differs from the pinned artifact")
    return tokenizer


def prompt_for(row: dict) -> str:
    def comments(text):
        return "\n".join("# " + line for line in text.splitlines())

    return (
        "# Task\n"
        + comments(row["text"])
        + "\n# Test setup\n"
        + comments(row["test_setup_code"])
        + "\n# Tests\n"
        + comments("\n".join(row["test_list"]))
        + "\n# Solution\n"
    )


class BoundaryError(ValueError):
    """A token cannot be assigned wholly to prompt or solution."""


def solution_tokens(tokenizer, prompt: str, solution: str):
    if not prompt or not solution.strip():
        raise ValueError("Prompt and solution must be nonempty")
    text = prompt + solution
    boundary = len(prompt)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    tokens = list(encoded.ids)
    if not tokens or len(tokens) != len(encoded.offsets):
        raise ValueError("Tokenizer returned missing IDs or offsets")
    if any(type(t) is not int or not 0 <= t < Config().vocab for t in tokens):
        raise ValueError("Token ID is outside the FLAME vocabulary")
    mask = []
    previous_start = previous_end = 0
    for start, end in encoded.offsets:
        if not (0 <= start < end <= len(text)):
            raise BoundaryError("Tokenizer returned empty or invalid offsets")
        if start < previous_start or end < previous_end:
            raise BoundaryError("Tokenizer offsets are not monotonic")
        if start < boundary < end:
            raise BoundaryError("Token straddles the prompt/solution boundary")
        mask.append(start >= boundary)
        previous_start, previous_end = start, end
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    n_prompt = sum(not scored for scored in mask)
    if (
        not n_prompt
        or n_prompt == len(tokens)
        or tokens[:n_prompt] != list(prompt_ids)
        or encoded.offsets[0][0] != 0
        or encoded.offsets[n_prompt - 1][1] != boundary
        or encoded.offsets[n_prompt][0] != boundary
        or encoded.offsets[-1][1] != len(text)
    ):
        raise BoundaryError("Tokenization does not preserve the exact text boundary")
    tokens.append(EOS_ID)
    mask.append(True)
    mask[0] = False
    return tokens, mask


def prepare(rows: list[dict], tokenizer, *, count: int = 8, max_tokens: int = 256):
    """Select ascending TRAIN IDs by length/boundary eligibility, without scores."""
    if type(count) is not int or not 1 <= count <= len(TRAIN_IDS):
        raise ValueError("count must be between 1 and 374")
    if type(max_tokens) is not int or not 2 <= max_tokens <= Config().context:
        raise ValueError("max_tokens must be between 2 and 2048")
    by_id = {}
    for row in rows:
        task_id = row.get("task_id")
        if type(task_id) is not int or task_id not in TRAIN_IDS:
            raise ValueError("Only MBPP TRAIN task IDs 601..974 are permitted")
        if task_id in by_id:
            raise ValueError(f"Duplicate MBPP task ID {task_id}")
        if any(
            not isinstance(row.get(key), str)
            for key in ("text", "code", "test_setup_code")
        ):
            raise ValueError(f"Invalid MBPP text fields for {task_id}")
        if not row["text"].strip() or not row["code"].strip():
            raise ValueError(f"Empty MBPP prompt or reference for {task_id}")
        if not isinstance(row.get("test_list"), list) or not all(
            isinstance(test, str) for test in row["test_list"]
        ):
            raise ValueError(f"Invalid MBPP test_list for {task_id}")
        by_id[task_id] = row
    ordered_ids = sorted(by_id)
    examples = []
    oversize = []
    boundaries = []
    eligible = 0
    for task_id in ordered_ids:
        row = by_id[task_id]
        prompt, solution = prompt_for(row), row["code"]
        try:
            tokens, mask = solution_tokens(tokenizer, prompt, solution)
        except BoundaryError as error:
            boundaries.append({"task_id": task_id, "reason": str(error)})
            continue
        if len(tokens) > max_tokens:
            oversize.append({"task_id": task_id, "tokens": len(tokens)})
            continue
        eligible += 1
        if len(examples) == count:
            continue
        examples.append(
            {
                "id": f"mbpp/train/{task_id}",
                "task_id": task_id,
                "tokens": tokens,
                "loss_mask": mask,
                "prompt": prompt,
                "solution": solution,
                "prompt_sha256": text_hash(prompt),
                "reference_sha256": text_hash(solution),
                "prompt_solution_sha256": text_hash(prompt + solution),
                "reference_source": "MBPP full code field (dataset supplied)",
                "execution_verified": False,
            }
        )
    stats = {
        "candidate_count": len(ordered_ids),
        "eligible_count": eligible,
        "excluded_oversize_count": len(oversize),
        "excluded_boundary_count": len(boundaries),
        "example_count": len(examples),
        "total_tokens": sum(len(ex["tokens"]) for ex in examples),
        "total_input_tokens": sum(len(ex["tokens"]) - 1 for ex in examples),
        "total_scored_tokens": sum(sum(ex["loss_mask"]) for ex in examples),
    }
    if len(examples) < count:
        raise ValueError(
            f"Requested {count} examples, only {eligible} eligible; "
            f"excluded oversize={len(oversize)}, boundary={len(boundaries)}. "
            "No truncation or partial objective was written."
        )
    return {
        "format": "ennx.solution_tokens.v1",
        "provenance": {
            "model": {"id": MODEL_ID, "revision": REVISION, "iteration": ITERATION},
            "dataset": {
                "id": DATASET_ID,
                "revision": DATASET_REVISION,
                "config": "full",
                "split": "train",
                "license": "CC-BY-4.0",
                "metadata_url": DATASET_INFO_URL,
                "source_url": DATASET_URL,
                "source_revision": DATASET_SOURCE_REVISION,
                "source_sha256": DATASET_SHA256,
                "train_task_id_range_inclusive": [601, 974],
            },
            "tokenizer": {
                "id": TOKENIZER_ID,
                "revision": TOKENIZER_REVISION,
                "artifact_url": TOKENIZER_URL,
                "sha256": TOKENIZER_SHA256,
                "checkpoint_args_sha256": CHECKPOINT_ARGS_SHA256,
                "checkpoint_tokenizer_type": "HuggingFaceTokenizer",
                "megatron_revision": REFERENCE,
                "checkpoint_records_tokenizer_revision": False,
                "offset_trim": False,
                "add_special_tokens": False,
                "normalization": "NFC (from pinned tokenizer.json)",
                "eos_id": EOS_ID,
            },
            "selection": {
                "rule": "ascending numeric TRAIN task_id, first count eligible; no model scores",
                "candidate_task_ids": ordered_ids,
                "selected_task_ids": [ex["task_id"] for ex in examples],
                "requested_count": count,
                "max_tokens_including_eos": max_tokens,
                "excluded_oversize": oversize,
                "excluded_boundary": boundaries,
            },
            "reference_verification": "dataset-supplied references; no programs executed",
            "evaluation_separation": "BigCodeBench-Hard is evaluation-only; no data loaded or used",
            "prompt_template": "ennx.mbpp_comments.v1",
            "loss_mask_semantics": "mask[i] scores tokens[i] from tokens[:i]; mask[0]=false",
            "eos_policy": "append exactly one EOS (ID 0) after reference; score it; no BOS",
            "text_hash_encoding": "SHA256 of exact UTF-8 text before tokenizer NFC normalization",
            "statistics": stats,
        },
        "examples": examples,
    }


@click.command(help=__doc__, context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
@click.option("--count", default=8, show_default=True, type=click.IntRange(1, 374))
@click.option(
    "--max-tokens", default=256, show_default=True, type=click.IntRange(2, 2048)
)
def main(output: Path, count: int, max_tokens: int):
    try:
        rows = load_train()
        tokenizer = load_tokenizer()
        result = prepare(rows, tokenizer, count=count, max_tokens=max_tokens)
        result["provenance"]["tokenizer"]["package_version"] = version("tokenizers")
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                result,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
    except (OSError, ValueError, KeyError, requests.RequestException) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result["provenance"]["statistics"], sort_keys=True), err=True)


if __name__ == "__main__":
    main()
