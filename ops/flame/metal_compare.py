"""Compare original and final Metal BO checkpoints after training, on its TRAIN corpus."""

import gc
import hashlib
import json
import time
from pathlib import Path

import click
import numpy as np

from .config import Config
from .layout import Block, Layout
from .metal import NativeEvaluator, upload_weights
from .native import load_checkpoint
from .objective import SolutionObjective


def compare(original, run):
    import ennx.ennx_rust as extension

    record = json.loads((run / "run.json").read_text())
    if record["status"] != "complete" or record["backend"] != "metal":
        raise ValueError("Comparison requires a completed Metal run")
    extension_hash = hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
    if extension_hash != record["extension_sha256"]:
        raise ValueError("Use the exact extension binary that produced the run")
    if (
        hashlib.sha256((original / "manifest.json").read_bytes()).hexdigest()
        != record["source_manifest_sha256"]
    ):
        raise ValueError("Original checkpoint manifest does not match the run")
    layout = Layout(
        tuple(
            Block(
                name=b["name"],
                key=b["key"],
                offset=b["offset"],
                shape=tuple(b["shape"]),
                scale=b["scale"],
                weight=b["weight"],
            )
            for b in record["blocks"]
        )
    )
    objective = SolutionObjective.parse(record["tokens"], Config())
    engine = NativeEvaluator(objective, Config())
    events = [
        json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
    ]
    if not events or events[0]["event"] != "baseline":
        raise ValueError("Run has no baseline event")
    expected = [events[0]["minibatch"], record["final_minibatch"]]
    values = []
    elapsed = []
    for checkpoint, check in zip((original, run / "best"), expected, strict=True):
        config, params = load_checkpoint(checkpoint)
        if config != Config():
            raise ValueError("Unexpected checkpoint architecture")
        flat = layout.flatten_torch(params)
        del params
        gc.collect()
        weights = upload_weights(flat)
        del flat
        gc.collect()
        actual = engine.losses(weights, check["indices"])
        np.testing.assert_array_equal(actual, check["losses"])
        start = time.perf_counter()
        losses = engine.losses(weights, np.arange(len(engine.tokens)))
        elapsed.append(time.perf_counter() - start)
        if not np.isfinite(losses).all():
            raise ValueError("Nonfinite full-corpus losses")
        values.append(losses)
        del weights
        gc.collect()
    before, after = values
    return {
        "scope": "TRAIN corpus; not held-out validation or generated-code accuracy",
        "problems": len(before),
        "original_mean_loss": float(before.mean()),
        "final_mean_loss": float(after.mean()),
        "change_final_minus_original": float((after - before).mean()),
        "improved_problems": int(np.count_nonzero(after < before)),
        "worsened_problems": int(np.count_nonzero(after > before)),
        "unchanged_problems": int(np.count_nonzero(after == before)),
        "original_losses": before.tolist(),
        "final_losses": after.tolist(),
        "evaluation_seconds": elapsed,
        "checkpoint_reload_losses_exact": True,
        "post_run_problem_forwards": len(before) * 2
        + sum(len(c["indices"]) for c in expected),
        "extension_sha256": extension_hash,
        "run_sha256": hashlib.sha256((run / "run.json").read_bytes()).hexdigest(),
        "final_manifest_sha256": hashlib.sha256(
            (run / "best/manifest.json").read_bytes()
        ).hexdigest(),
    }


@click.command(help=__doc__)
@click.argument(
    "original", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("run", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
def main(original, run, output):
    if output.exists():
        raise click.ClickException("Refusing to overwrite an existing comparison")
    result = compare(original, run)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    click.echo(
        json.dumps(
            {k: v for k, v in result.items() if not k.endswith("_losses")},
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
