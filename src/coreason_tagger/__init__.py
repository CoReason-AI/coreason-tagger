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
from coreason_tagger.interfaces import BaseAssertionDetector, BaseExtractor, BaseLinker
from coreason_tagger.ner import GLiNERExtractor
from coreason_tagger.schema import (
    AssertionStatus,
    EntityCandidate,
    ExtractionStrategy,
    LinkedEntity,
)
from coreason_tagger.tagger import CoreasonTagger, CoreasonTaggerAsync

__version__ = "0.2.0"

__all__ = [
    "BaseAssertionDetector",
    "BaseExtractor",
    "BaseLinker",
    "GLiNERExtractor",
    "RegexBasedAssertionDetector",
    "CoreasonTagger",
    "CoreasonTaggerAsync",
    "AssertionStatus",
    "LinkedEntity",
    "EntityCandidate",
    "ExtractionStrategy",
    "__version__",
]
