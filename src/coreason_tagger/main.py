# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import asyncio
import json
import os
from typing import List, Optional

import typer
from transformers import logging as transformers_logging
from typing_extensions import Annotated

from coreason_tagger import __version__
from coreason_tagger.schema import ExtractionStrategy
from coreason_tagger.tagger import CoreasonTaggerAsync
from coreason_tagger.utils.logger import logger, setup_logger

transformers_logging.set_verbosity_error()  # type: ignore
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = typer.Typer(help="CoReason Tagger CLI")


def get_tagger() -> CoreasonTaggerAsync:
    """Factory function to initialize the full tagger pipeline.

    In a real app, this might rely on Dependency Injection containers.

    Returns:
        CoreasonTaggerAsync: The initialized tagger instance.
    """
    logger.info("Initializing Tagger Pipeline...")
    return CoreasonTaggerAsync()


@app.command()
def version() -> None:
    """Print the version of the package."""
    typer.echo(f"coreason-tagger version: {__version__}")


async def _tag_async(text: str, labels: List[str], strategy: ExtractionStrategy) -> None:
    """Async helper for the tag command.

    Args:
        text: The input text.
        labels: The labels to extract.
        strategy: The extraction strategy.
    """
    try:
        async with get_tagger() as tagger:
            results = await tagger.tag(text, labels, strategy=strategy)
            # Convert Pydantic models to list of dicts
            output = [entity.model_dump() for entity in results]
            typer.echo(json.dumps(output, indent=2))
    except Exception as e:
        logger.exception("Failed to process text")
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e


@app.command()
def tag(
    text: str,
    labels: Annotated[Optional[List[str]], typer.Option("--label", "-l", help="Entity labels to extract")] = None,
    strategy: Annotated[
        ExtractionStrategy, typer.Option("--strategy", "-s", help="Extraction strategy to use")
    ] = ExtractionStrategy.SPEED_GLINER,
) -> None:
    """Tag a single string of text and output JSON.

    Args:
        text: The text to tag.
        labels: Optional list of labels to extract. Defaults to ["Symptom", "Drug", "Condition"].
        strategy: The extraction strategy to use. Defaults to SPEED_GLINER.
    """
    if labels is None:
        # Default labels if none provided
        labels = ["Symptom", "Drug", "Condition"]

    asyncio.run(_tag_async(text, labels, strategy))


if __name__ == "__main__":  # pragma: no cover
    setup_logger()
    app()
