# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import os
from unittest.mock import patch

from coreason_tagger.config import Settings


def test_empty_api_key_override() -> None:
    """
    Edge Case: Verify that setting CODEX_API_KEY to an empty string in the environment
    results in an empty string in the settings, NOT None (the default).
    This ensures we can explicitly 'unset' or blank out the key if needed.
    """
    env_vars = {"CODEX_API_KEY": ""}
    with patch.dict(os.environ, env_vars):
        settings = Settings()
        assert settings.CODEX_API_KEY == ""
        assert settings.CODEX_API_KEY is not None


def test_numeric_api_key_casting() -> None:
    """
    Edge Case: Verify that if CODEX_API_KEY is provided as a numeric string,
    it is correctly cast to a string. Pydantic handles this, but we verify it.
    """
    # Environment variables are always strings in os.environ, but let's simulate the string "12345"
    env_vars = {"CODEX_API_KEY": "12345"}
    with patch.dict(os.environ, env_vars):
        settings = Settings()
        assert settings.CODEX_API_KEY == "12345"
        assert isinstance(settings.CODEX_API_KEY, str)


def test_complex_env_isolation() -> None:
    """
    Complex Scenario: Verify that setting unrelated environment variables does NOT affect
    the core settings, and that we can mix defaults with overrides seamlessly.
    """
    env_vars = {
        "CODEX_API_URL": "http://custom-url.com",
        "SOME_RANDOM_VAR": "should_be_ignored",
        "NER_MODEL_NAME": "custom-ner",
        # CODEX_API_KEY is missing, should remain default (None)
    }
    with patch.dict(os.environ, env_vars):
        settings = Settings()

        # Overrides
        assert settings.CODEX_API_URL == "http://custom-url.com"
        assert settings.NER_MODEL_NAME == "custom-ner"

        # Defaults
        assert settings.CODEX_API_KEY is None
        assert settings.LINKER_CANDIDATE_TOP_K == 10  # Default preserved

        # Verify strictness (pydantic-settings ignores extras by default, which is good)
        # We just verify it didn't crash.
