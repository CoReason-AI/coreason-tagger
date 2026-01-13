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
from coreason_tagger.schema import AssertionStatus, ExtractedSpan, TaggedEntity


class ConcreteExtractor(BaseNERExtractor):
    def extract(self, text: str, labels: List[str]) -> List[ExtractedSpan]:
        if not text:
            return []
        return [ExtractedSpan(text="test", label=labels[0], start=0, end=4, score=0.99)]


class ConcreteAssertionDetector(BaseAssertionDetector):
    def detect(self, text: str, span_text: str, span_start: int, span_end: int) -> AssertionStatus:
        if "not" in text:
            return AssertionStatus.ABSENT
        return AssertionStatus.PRESENT


class ConcreteLinker(BaseLinker):
    def link(self, span: ExtractedSpan, context: str) -> Dict[str, Any]:
        if span.text == "unknown":
            return {}
        return {"concept_id": "1", "concept_name": "Test", "link_confidence": 1.0}


def test_extractor_interface() -> None:
    extractor = ConcreteExtractor()
    result = extractor.extract("sample text", ["TestLabel"])
    assert len(result) == 1
    assert isinstance(result[0], ExtractedSpan)
    assert result[0].text == "test"
    assert result[0].label == "TestLabel"


def test_extractor_empty_text() -> None:
    """Edge case: Empty text should return empty list."""
    extractor = ConcreteExtractor()
    result = extractor.extract("", ["TestLabel"])
    assert result == []


def test_assertion_detector_interface() -> None:
    detector = ConcreteAssertionDetector()
    result = detector.detect("sample text", "sample", 0, 6)
    assert result == AssertionStatus.PRESENT


def test_linker_interface() -> None:
    linker = ConcreteLinker()
    span = ExtractedSpan(text="sample", label="Label", start=0, end=6, score=1.0)
    result = linker.link(span, "sample text")
    assert result == {"concept_id": "1", "concept_name": "Test", "link_confidence": 1.0}


def test_pipeline_simulation_complex() -> None:
    """
    Complex Scenario: Simulate a full pipeline flow (Extract -> Assert -> Link)
    ensuring data types flow correctly between components.
    """
    text = "Patient does not have test."
    labels = ["Symptom"]

    # 1. Extract
    extractor = ConcreteExtractor()
    spans = extractor.extract(text, labels)
    assert len(spans) == 1
    span = spans[0]

    # 2. Assert
    detector = ConcreteAssertionDetector()
    status = detector.detect(text, span.text, span.start, span.end)
    # The concrete mock detects "not" -> ABSENT
    assert status == AssertionStatus.ABSENT

    # 3. Link
    linker = ConcreteLinker()
    link_info = linker.link(span, text)
    assert link_info["concept_id"] == "1"

    # 4. Construct Final Entity
    entity = TaggedEntity(
        span_text=span.text,
        label=span.label,
        concept_id=link_info["concept_id"],
        concept_name=link_info["concept_name"],
        link_confidence=link_info["link_confidence"],
        assertion=status,
    )

    assert entity.assertion == AssertionStatus.ABSENT
    assert entity.concept_id == "1"
