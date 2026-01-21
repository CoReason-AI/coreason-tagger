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

from async_lru import alru_cache
from gliner import GLiNER
from loguru import logger
from sentence_transformers import SentenceTransformer


@alru_cache(maxsize=1)  # type: ignore[misc]
async def get_gliner_model(model_name: str) -> GLiNER:
    """
    Load the GLiNER model. Caches the result to ensure singleton behavior per model name.
    """
    logger.info(f"Loading GLiNER model: {model_name}")
    loop = asyncio.get_running_loop()
    # GLiNER.from_pretrained can trigger downloads, so run in executor
    model = await loop.run_in_executor(None, GLiNER.from_pretrained, model_name)
    return model


@alru_cache(maxsize=1)  # type: ignore[misc]
async def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    """
    Load the SentenceTransformer model. Caches the result to ensure singleton behavior per model name.
    """
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(None, SentenceTransformer, model_name)
    return model
