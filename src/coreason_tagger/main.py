# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import json
from typing import List, Optional

import typer
from typing_extensions import Annotated

from coreason_tagger import __version__
from coreason_tagger.assertion_detector import RegexBasedAssertionDetector
from coreason_tagger.codex_mock import MockCoreasonCodex
from coreason_tagger.linker import VectorLinker
from coreason_tagger.ner import GLiNERExtractor
from coreason_tagger.tagger import CoreasonTagger
from coreason_tagger.utils.logger import logger

app = typer.Typer(help="CoReason Tagger CLI")


def get_tagger() -> CoreasonTagger:
    """
    Factory function to initialize the full tagger pipeline.
    In a real app, this might rely on Dependency Injection containers.
    """
    logger.info("Initializing Tagger Pipeline...")
    ner = GLiNERExtractor()
    assertion = RegexBasedAssertionDetector()
    # TODO: Replace MockCoreasonCodex with real client when available
    codex_client = MockCoreasonCodex()
    linker = VectorLinker(codex_client=codex_client)
    return CoreasonTagger(ner=ner, assertion=assertion, linker=linker)


@app.command()
def version() -> None:
    """Print the version of the package."""
    typer.echo(f"coreason-tagger version: {__version__}")


@app.command()
def tag(
    text: str,
    labels: Annotated[Optional[List[str]], typer.Option("--label", "-l", help="Entity labels to extract")] = None,
) -> None:
    """
    Tag a single string of text and output JSON.
    """
    if labels is None:
        # Default labels if none provided
        labels = ["Symptom", "Drug", "Condition"]

    tagger = get_tagger()
    try:
        results = tagger.tag(text, labels)
        # Convert Pydantic models to list of dicts
        output = [entity.model_dump() for entity in results]
        typer.echo(json.dumps(output, indent=2))
    except Exception as e:
        logger.exception("Failed to process text")
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()  # pragma: no cover
