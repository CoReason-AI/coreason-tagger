# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from coreason_tagger.config import settings
from coreason_tagger.ner import ExtractorFactory
from coreason_tagger.schema import ExtractionStrategy, TaggingRequest, LinkedEntity
from coreason_tagger.tagger import CoreasonTaggerAsync


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize the tagger and load models."""
    logger.info("Initializing CoreasonTaggerAsync...")
    tagger = CoreasonTaggerAsync()
    await tagger.__aenter__()

    # Pre-load the default GLiNER model to prevent latency on first request
    logger.info(f"Pre-loading default model: {settings.NER_MODEL_NAME}")
    if isinstance(tagger.ner_or_factory, ExtractorFactory):
        # We assume the default strategy (SPEED_GLINER) uses the default model
        extractor = tagger.ner_or_factory.get_extractor(ExtractionStrategy.SPEED_GLINER)
        if hasattr(extractor, "load_model"):
            await extractor.load_model()
            logger.info("Model loaded successfully.")

    app.state.tagger = tagger
    yield
    logger.info("Shutting down CoreasonTaggerAsync...")
    await tagger.__aexit__(None, None, None)


app = FastAPI(title="Coreason Tagger NER Service", lifespan=lifespan)


@app.post("/tag", response_model=list[LinkedEntity])
async def tag_text(request: TaggingRequest):
    """Extract entities from text."""
    tagger: CoreasonTaggerAsync = app.state.tagger
    return await tagger.tag(request.text, request.labels, request.strategy)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ready", "model": settings.NER_MODEL_NAME}
