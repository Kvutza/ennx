"""Build contiguous prompt/continuation episodes from immutable token pools."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from array import array
from pathlib import Path


def documents(path: Path):
    """Recover file boundaries from the EOT markers retained in legacy pools."""
    tokens = array("H")
    with path.open("rb") as stream:
        tokens.fromfile(stream, path.stat().st_size // 2)
    if sys.byteorder != "little":
        tokens.byteswap()
    index_path = path.parent / "document-index.json"
    if index_path.exists():
        records = json.loads(index_path.read_text())[path.name]
        previous_end = 0
        for record in records:
            first = record["offset"]
            end = first + record["length"]
            if first < previous_end or end >= len(tokens) or tokens[end] != 1:
                raise ValueError(f"invalid document index for {path}")
            yield first, tokens[first:end].tolist()
            previous_end = end + 1
        return
    first = 0
    for offset, token in enumerate(tokens):
        if token not in (1, 2):
            continue
        if offset > first:
            yield first, tokens[first:offset].tolist()
        first = offset + 1
    if first != len(tokens):
        raise ValueError(f"token pool has an unterminated document: {path}")


def build_episodes(
    paths,
    counts,
    quotas,
    prompt_tokens,
    generated_tokens,
    seed,
    *,
    sources=None,
    buckets=None,
):
    if prompt_tokens < 1 or generated_tokens < 1:
        raise ValueError("episode prompt and continuation lengths must be positive")
    sources = sources or {split: [split] for split in counts}
    buckets = buckets or list(quotas)
    result = {}
    for split, count in counts.items():
        candidates = {}
        for bucket in buckets:
            rows = []
            for source in sources[split]:
                path = paths[source][bucket]
                index_path = path.parent / "document-index.json"
                provenance = (
                    {
                        record["offset"]: record
                        for record in json.loads(index_path.read_text()).get(
                            path.name, []
                        )
                    }
                    if index_path.exists()
                    else {}
                )
                for offset, tokens in documents(path):
                    window = prompt_tokens + generated_tokens
                    stride = max(1, window // 4)
                    document_rows = []
                    for start in range(0, len(tokens) - window + 1, stride):
                        values = tokens[start : start + window]
                        encoded = array("H", values)
                        if sys.byteorder != "little":
                            encoded.byteswap()
                        identity = hashlib.sha256(encoded.tobytes()).hexdigest()
                        document_rows.append(
                            {
                                "id": identity,
                                "bucket": bucket,
                                "pool": path.name,
                                "source_split": source,
                                "document_offset": offset,
                                "window_offset": start,
                                "document_tokens": len(tokens),
                                "source_document": provenance.get(offset),
                                "prompt": values[:prompt_tokens],
                                "expected": values[prompt_tokens:],
                            }
                        )
                    if document_rows:
                        # One episode per file prevents overlapping windows from
                        # masquerading as independent evaluation examples.
                        chooser = random.Random(
                            f"{seed}/{split}/{bucket}/{source}/{offset}"
                        )
                        rows.append(chooser.choice(document_rows))
            random.Random(f"{seed}/{split}/{bucket}").shuffle(rows)
            candidates[bucket] = rows
        selected = []
        used = dict.fromkeys(buckets, 0)
        while len(selected) < count:
            available = [bucket for bucket in buckets if candidates.get(bucket)]
            if not available:
                raise ValueError(
                    f"{split} has only {len(selected)} document-aligned episodes "
                    f"with {prompt_tokens} prompt + {generated_tokens} continuation tokens; "
                    f"requires {count}"
                )
            # Weighted fair selection balances whole episodes, never token slices.
            bucket = min(available, key=lambda bucket: used[bucket] / quotas[bucket])
            selected.append(candidates[bucket].pop())
            used[bucket] += 1
        result[split] = {
            "episodes": selected,
            "mixture_episodes": used,
            "eligible_remaining": {
                bucket: len(rows) for bucket, rows in candidates.items()
            },
        }
    return result


def write_episodes(
    path,
    paths,
    counts,
    quotas,
    prompt_tokens,
    generated_tokens,
    seed,
    *,
    sources=None,
    buckets=None,
):
    splits = build_episodes(
        paths,
        counts,
        quotas,
        prompt_tokens,
        generated_tokens,
        seed,
        sources=sources,
        buckets=buckets,
    )
    missing = [
        row["id"]
        for split in splits.values()
        for row in split["episodes"]
        if not isinstance(row.get("source_document"), dict)
        or any(
            not row["source_document"].get(field)
            for field in ("repository", "commit", "path", "content_id")
        )
    ]
    if missing:
        raise ValueError(
            f"episode provenance is missing for {len(missing)} selected documents"
        )
    document = {
        "schema": "ennx.document_episodes.v2",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "window_stride": max(1, (prompt_tokens + generated_tokens) // 4),
        "boundaries": "end-of-text-delimited files; no cross-file windows",
        "provenance": "repository, commit, path, content ID, source split, immutable token-pool offset",
        "selection": "one deterministic window per source document",
        "splits": splits,
    }
    with path.open("x") as stream:
        json.dump(document, stream, separators=(",", ":"))
        stream.write("\n")
    return document
