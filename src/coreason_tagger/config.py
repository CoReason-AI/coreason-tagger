# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings.

    Values can be overridden by environment variables (e.g., APP_ENV=production).

    Attributes:
        APP_ENV: The application environment (development/production).
        DEBUG: Whether to enable debug mode.
        LOG_LEVEL: Logging level (INFO, DEBUG, etc.).
        NER_MODEL_NAME: Default GLiNER model name.
        NUNER_MODEL_NAME: Default NuNER model name.
        ASSERTION_MODEL_NAME: Default Assertion model name.
        LLM_MODEL_NAME: Default LLM model name for reasoning.
        LLM_API_KEY: API key for the LLM service.
        CODEX_API_URL: URL for the Codex API.
        CODEX_API_KEY: API key for the Codex API.
        LINKER_MODEL_NAME: Default Sentence Transformer model for linking.
        LINKER_CANDIDATE_TOP_K: Number of candidates to retrieve.
        LINKER_WINDOW_SIZE: Context window size for linking.
        REDIS_URL: Redis connection URL.
        REDIS_TTL: Redis cache time-to-live in seconds.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Core
    APP_ENV: str = "development"
    DEBUG: bool = False

    # Logging
    LOG_LEVEL: str = "INFO"

    # NER Configuration
    NER_MODEL_NAME: str = "urchade/gliner_small-v2.1"
    NUNER_MODEL_NAME: str = "numind/NuNER-Zero"

    # Assertion Configuration
    ASSERTION_MODEL_NAME: str = "distilbert-base-uncased"

    # LLM Configuration
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    LLM_API_KEY: str | None = None

    # Codex Configuration
    CODEX_API_URL: str = "http://localhost:8000"
    CODEX_API_KEY: str | None = None

    # Linker Configuration
    LINKER_MODEL_NAME: str = "all-MiniLM-L6-v2"
    LINKER_CANDIDATE_TOP_K: int = 10
    LINKER_WINDOW_SIZE: int = 50

    # Redis Configuration
    REDIS_URL: str | None = None
    REDIS_TTL: int = 86400  # 24 hours


settings = Settings()
