"""Persistent, model-independent tokenization for the frozen Qwen evaluator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ops.qwen.config import Config
from ops.qwen.tokenizer import load


def encode(tokenizer, prompt: str, completion: str, limit: int):
    # Explicit segmentation makes every scored token belong to the completion;
    # a BPE merge across the prompt boundary cannot silently score prompt bytes.
    prefix = tokenizer.encode(prompt, add_special_tokens=False).ids
    suffix = tokenizer.encode(completion, add_special_tokens=False).ids
    tokens = [Config().bos_token_id, *prefix, *suffix]
    if not suffix or len(tokens) > limit:
        raise ValueError("completion is empty or exceeds frozen evaluator context")
    return {
        "tokens": tokens,
        "mask": [False] * (1 + len(prefix)) + [True] * len(suffix),
        "completion_tokens": len(suffix),
        "completion_bytes": len(completion.encode()),
    }


def main():
    setup = json.loads(sys.stdin.readline())
    checkpoint = Path(setup["checkpoint"])
    Config.from_file(checkpoint / "config.json")
    tokenizer = load(checkpoint)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            rows = [
                encode(tokenizer, row["prompt"], row["completion"], setup["max_tokens"])
                for row in request["rows"]
            ]
            result = {"rows": rows}
        except (ValueError, KeyError, TypeError) as error:
            result = {"error": str(error)}
        print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
