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
from unittest.mock import MagicMock, patch

from coreason_tagger import __version__
from coreason_tagger.main import app, get_tagger
from coreason_tagger.schema import AssertionStatus, TaggedEntity
from coreason_tagger.tagger import CoreasonTagger
from typer.testing import CliRunner

runner = CliRunner()


def test_version_command() -> None:
    """Test the version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"coreason-tagger version: {__version__}" in result.stdout


@patch("coreason_tagger.main.get_tagger")
def test_tag_command_success(mock_get_tagger: MagicMock) -> None:
    """Test the tag command with a valid input."""
    # Mock the tagger instance and its tag method
    mock_tagger_instance = MagicMock()
    mock_get_tagger.return_value = mock_tagger_instance

    # Return a dummy entity
    mock_entity = TaggedEntity(
        span_text="fever",
        label="Symptom",
        concept_id="123",
        concept_name="Fever",
        link_confidence=0.99,
        assertion=AssertionStatus.PRESENT,
    )
    mock_tagger_instance.tag.return_value = [mock_entity]

    result = runner.invoke(app, ["tag", "Patient has fever"])

    assert result.exit_code == 0
    # Verify JSON output
    output = json.loads(result.stdout)
    assert len(output) == 1
    assert output[0]["span_text"] == "fever"
    assert output[0]["assertion"] == "PRESENT"

    # Verify calls
    mock_tagger_instance.tag.assert_called_once()


@patch("coreason_tagger.main.get_tagger")
def test_tag_command_custom_labels(mock_get_tagger: MagicMock) -> None:
    """Test the tag command with custom labels."""
    mock_tagger_instance = MagicMock()
    mock_get_tagger.return_value = mock_tagger_instance
    mock_tagger_instance.tag.return_value = []

    result = runner.invoke(app, ["tag", "Patient text", "--label", "Custom1", "-l", "Custom2"])

    assert result.exit_code == 0
    mock_tagger_instance.tag.assert_called_once()
    # Check that labels were passed correctly
    call_args = mock_tagger_instance.tag.call_args
    assert set(call_args[0][1]) == {"Custom1", "Custom2"}


@patch("coreason_tagger.main.get_tagger")
def test_tag_command_error(mock_get_tagger: MagicMock) -> None:
    """Test error handling in tag command."""
    mock_tagger_instance = MagicMock()
    mock_get_tagger.return_value = mock_tagger_instance
    mock_tagger_instance.tag.side_effect = Exception("Pipeline failure")

    result = runner.invoke(app, ["tag", "Crash me"])

    assert result.exit_code == 1
    assert "Error: Pipeline failure" in result.stderr


def test_get_tagger_factory() -> None:
    """
    Test the factory function creates a valid CoreasonTagger instance.
    We mock the heavy dependencies to make this unit test fast.
    """
    with patch("coreason_tagger.main.GLiNERExtractor"), patch("coreason_tagger.main.VectorLinker"):
        tagger = get_tagger()
        assert isinstance(tagger, CoreasonTagger)
        assert tagger.ner is not None
        assert tagger.assertion is not None
        assert tagger.linker is not None
