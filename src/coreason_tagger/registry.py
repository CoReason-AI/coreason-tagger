# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from __future__ import annotations

import asyncio
from typing import Any, Optional

import redis.asyncio as redis
from async_lru import alru_cache
from gliner import GLiNER
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import pipeline


@alru_cache(maxsize=1)
async def get_gliner_model(model_name: str) -> GLiNER:
    """Load the GLiNER model. Caches the result to ensure singleton behavior per model name.

    Args:
        model_name: The name or path of the GLiNER model.

    Returns:
        GLiNER: The loaded GLiNER model instance.
    """
    logger.info(f"Loading GLiNER model: {model_name}")
    # GLiNER.from_pretrained can trigger downloads, so run in executor
    model = await asyncio.to_thread(GLiNER.from_pretrained, model_name)
    return model


@alru_cache(maxsize=1)
async def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Load the SentenceTransformer model. Caches the result to ensure singleton behavior per model name.

    Args:
        model_name: The name or path of the SentenceTransformer model.

    Returns:
        SentenceTransformer: The loaded SentenceTransformer model instance.
    """
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    model = await asyncio.to_thread(SentenceTransformer, model_name)
    return model


@alru_cache(maxsize=1)
async def get_nuner_pipeline(model_name: str) -> Any:
    """Load the NuNER pipeline (token-classification).

    Caches the result to ensure singleton behavior per model name.

    Args:
        model_name: The name or path of the NuNER model.

    Returns:
        Any: The loaded transformers pipeline for token classification.
    """
    logger.info(f"Loading NuNER pipeline: {model_name}")
    # transformers.pipeline can trigger downloads, so run in executor
    pipe = await asyncio.to_thread(
        pipeline,
        "token-classification",
        model=model_name,
        aggregation_strategy="simple",
        device_map="auto",
    )
    return pipe


@alru_cache(maxsize=1)
async def get_assertion_pipeline(model_name: str) -> Any:
    """Load the Assertion pipeline (text-classification).

    Caches the result to ensure singleton behavior per model name.

    Args:
        model_name: The name or path of the assertion model.

    Returns:
        Any: The loaded transformers pipeline for text classification.
    """
    logger.info(f"Loading Assertion pipeline: {model_name}")
    # transformers.pipeline can trigger downloads, so run in executor
    pipe = await asyncio.to_thread(
        pipeline,
        "text-classification",
        model=model_name,
        device_map="auto",
    )
    return pipe


@alru_cache(maxsize=1)
async def get_redis_client(redis_url: str) -> Optional[redis.Redis[Any]]:
    """Get a Redis client instance. Caches the result to ensure singleton behavior per URL.

    Args:
        redis_url: The URL connection string for Redis.

    Returns:
        Optional[redis.Redis[Any]]: The Redis client instance, or None if redis_url is empty.
    """
    if not redis_url:
        return None

    logger.info(f"Connecting to Redis at {redis_url}")
    try:
        client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        # Verify connection
        await client.ping()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None
