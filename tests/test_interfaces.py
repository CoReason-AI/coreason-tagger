# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import List

from coreason_tagger.interfaces import BaseAssertionDetector, BaseLinker, BaseNERExtractor
from coreason_tagger.schema import (
    AssertionStatus,
    EntityCandidate,
    ExtractionStrategy,
    LinkedEntity,
)


class ConcreteExtractor(BaseNERExtractor):
    def extract(self, text: str, labels: List[str], threshold: float = 0.5) -> List[EntityCandidate]:
        return [
            EntityCandidate(
                text="test",
                label=labels[0],
                start=0,
                end=4,
                confidence=1.0,
                source_model="mock",
            )
        ]

    def extract_batch(self, texts: List[str], labels: List[str], threshold: float = 0.5) -> List[List[EntityCandidate]]:
        return [
            [
                EntityCandidate(
                    text="test",
                    label=labels[0],
                    start=0,
                    end=4,
                    confidence=1.0,
                    source_model="mock",
                )
            ]
            for _ in texts
        ]


class ConcreteAssertionDetector(BaseAssertionDetector):
    def detect(self, text: str, span_text: str, span_start: int, span_end: int) -> AssertionStatus:
        return AssertionStatus.PRESENT


class ConcreteLinker(BaseLinker):
    def resolve(self, entity: EntityCandidate, context: str, strategy: ExtractionStrategy) -> LinkedEntity:
        return LinkedEntity(
            **entity.model_dump(),
            strategy_used=strategy,
            concept_id="1",
            concept_name="Test",
            link_score=1.0,
        )


def test_extractor_interface() -> None:
    extractor = ConcreteExtractor()
    result = extractor.extract("sample text", ["TestLabel"])
    assert len(result) == 1
    assert isinstance(result[0], EntityCandidate)
    assert result[0].text == "test"
    assert result[0].label == "TestLabel"

    batch_result = extractor.extract_batch(["sample1", "sample2"], ["TestLabel"])
    assert len(batch_result) == 2
    assert len(batch_result[0]) == 1
    assert batch_result[0][0].text == "test"


def test_assertion_detector_interface() -> None:
    detector = ConcreteAssertionDetector()
    result = detector.detect("sample text", "sample", 0, 6)
    assert result == AssertionStatus.PRESENT


def test_linker_interface() -> None:
    linker = ConcreteLinker()
    candidate = EntityCandidate(
        text="sample",
        label="Label",
        start=0,
        end=6,
        confidence=1.0,
        source_model="mock",
    )
    result = linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)
    assert isinstance(result, LinkedEntity)
    assert result.concept_id == "1"
    assert result.concept_name == "Test"
