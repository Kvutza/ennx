"""Inspect, download, and run the dense Qwen control model."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .checkpoint import fetch_model, load_checkpoint
from .config import MODEL_ID, REVISION, Config
from . import bench
from . import bo, coding, eval, joint


@click.group()
def main():
    """Use the pinned dense Qwen2.5-Coder-1.5B control path."""


@main.command()
@click.argument("directory", type=click.Path(file_okay=False, path_type=Path))
def inspect(directory: Path):
    """Validate and print architecture metadata without loading weights."""
    config = Config.from_file(directory / "config.json")
    click.echo(
        json.dumps(
            {"model_id": MODEL_ID, "revision": REVISION, "config": config.__dict__},
            indent=2,
            sort_keys=True,
        )
    )


@main.command()
@click.option(
    "--output", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option(
    "--weights/--no-weights",
    default=False,
    help="Also download the explicit 3.1 GB weight file.",
)
@click.option(
    "--force", is_flag=True, help="Replace files already in the output directory."
)
def download(output: Path, weights: bool, force: bool):
    """Download the pinned Qwen metadata, optionally including weights."""
    try:
        manifest = fetch_model(output, weights=weights, force=force)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(manifest, indent=2, sort_keys=True))


@main.command()
@click.argument("directory", type=click.Path(file_okay=False, path_type=Path))
@click.option("--prompt", required=True)
@click.option(
    "--max-new-tokens",
    default=128,
    show_default=True,
    type=click.IntRange(1, 1024),
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    show_default=True,
)
def generate(directory: Path, prompt: str, max_new_tokens: int, device: str):
    """Greedily generate a completion from a local dense checkpoint."""
    try:
        import torch

        from .model import greedy_generate
        from .tokenizer import load

        config, params = load_checkpoint(directory)
        tokenizer = load(directory)
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise click.ClickException("CUDA was requested but is unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise click.ClickException("MPS was requested but is unavailable")
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        if not encoded.ids:
            raise click.ClickException("Prompt tokenized to an empty sequence")
        tokens = torch.tensor([encoded.ids], dtype=torch.int64, device=device)
        params = {name: value.to(device) for name, value in params.items()}
        with torch.inference_mode():
            result = greedy_generate(params, tokens, max_new_tokens, config)
        completion = result[0, tokens.shape[1] :].tolist()
        click.echo(tokenizer.decode(completion, skip_special_tokens=False))
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error


main.add_command(bench.main, name="bench")
main.add_command(bo.main, name="bo")
main.add_command(eval.main, name="eval")
main.add_command(joint.main, name="joint")
main.add_command(coding.main, name="prepare")


if __name__ == "__main__":
    main()
