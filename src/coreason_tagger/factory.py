# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from coreason_tagger.assertion_detector import RegexBasedAssertionDetector
from coreason_tagger.codex_mock import MockCoreasonCodex
from coreason_tagger.linker import VectorLinker
from coreason_tagger.ner import GLiNERExtractor
from coreason_tagger.tagger import CoreasonTagger
from coreason_tagger.utils.logger import logger


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
