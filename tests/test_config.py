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
from coreason_tagger.linker import VectorLinker
from coreason_tagger.ner import GLiNERExtractor


def test_default_settings() -> None:
    """Test that default settings match requirements."""
    settings = Settings()
    assert settings.NER_MODEL_NAME == "urchade/gliner_small-v2.1"
    assert settings.LINKER_MODEL_NAME == "all-MiniLM-L6-v2"
    assert settings.LINKER_CANDIDATE_TOP_K == 10
    assert settings.LINKER_WINDOW_SIZE == 50
    assert settings.LOG_LEVEL == "INFO"


def test_env_override() -> None:
    """Test that environment variables override settings."""
    env_vars = {
        "NER_MODEL_NAME": "test-ner-model",
        "LINKER_MODEL_NAME": "test-linker-model",
        "LOG_LEVEL": "DEBUG",
        "LINKER_CANDIDATE_TOP_K": "20",
    }
    with patch.dict(os.environ, env_vars):
        # We must re-instantiate Settings to pick up new env vars
        # because the global instance is created at import time
        settings = Settings()
        assert settings.NER_MODEL_NAME == "test-ner-model"
        assert settings.LINKER_MODEL_NAME == "test-linker-model"
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.LINKER_CANDIDATE_TOP_K == 20


@patch("coreason_tagger.ner.GLiNER.from_pretrained")
def test_ner_uses_config(mock_load: object) -> None:
    """Test that NER extractor uses configured model name."""
    with patch("coreason_tagger.ner.settings") as mock_settings:
        mock_settings.NER_MODEL_NAME = "configured-model"
        extractor = GLiNERExtractor()
        assert extractor.model_name == "configured-model"


@patch("coreason_tagger.linker.SentenceTransformer")
def test_linker_uses_config(mock_st: object) -> None:
    """Test that VectorLinker uses configured settings."""
    with patch("coreason_tagger.linker.settings") as mock_settings:
        mock_settings.LINKER_MODEL_NAME = "configured-linker"
        mock_settings.LINKER_WINDOW_SIZE = 99
        mock_settings.LINKER_CANDIDATE_TOP_K = 5

        # We pass a mock client just to satisfy init
        linker = VectorLinker(codex_client=None)  # type: ignore

        assert linker.model_name == "configured-linker"
        assert linker.window_size == 99
        assert linker.candidate_top_k == 5
