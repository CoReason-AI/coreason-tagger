# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Any, Dict, List

from coreason_tagger.interfaces import BaseAssertionDetector, BaseLinker, BaseNERExtractor
from coreason_tagger.schema import AssertionStatus, ExtractedSpan


class ConcreteExtractor(BaseNERExtractor):
    def extract(self, text: str, labels: List[str]) -> List[ExtractedSpan]:
        return [ExtractedSpan(text="test", label=labels[0], start=0, end=4, score=0.99)]


class ConcreteAssertionDetector(BaseAssertionDetector):
    def detect(self, text: str, span_text: str, span_start: int, span_end: int) -> AssertionStatus:
        return AssertionStatus.PRESENT


class ConcreteLinker(BaseLinker):
    def link(self, span: ExtractedSpan, context: str) -> Dict[str, Any]:
        return {"concept_id": "1", "concept_name": "Test", "link_confidence": 1.0}


def test_extractor_interface() -> None:
    extractor = ConcreteExtractor()
    result = extractor.extract("sample text", ["TestLabel"])
    assert len(result) == 1
    assert isinstance(result[0], ExtractedSpan)
    assert result[0].text == "test"
    assert result[0].label == "TestLabel"


def test_assertion_detector_interface() -> None:
    detector = ConcreteAssertionDetector()
    result = detector.detect("sample text", "sample", 0, 6)
    assert result == AssertionStatus.PRESENT


def test_linker_interface() -> None:
    linker = ConcreteLinker()
    span = ExtractedSpan(text="sample", label="Label", start=0, end=6, score=1.0)
    result = linker.link(span, "sample text")
    assert result == {"concept_id": "1", "concept_name": "Test", "link_confidence": 1.0}
