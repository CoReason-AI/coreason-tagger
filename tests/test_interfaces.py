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

from coreason_tagger.interfaces import BaseAssertionDetector, BaseExtractor, BaseLinker
from coreason_tagger.schema import AssertionStatus


class ConcreteExtractor(BaseExtractor):
    def extract(self, text: str, labels: List[str]) -> List[Dict[str, Any]]:
        return [{"text": "test", "label": labels[0]}]


class ConcreteAssertionDetector(BaseAssertionDetector):
    def detect(self, text: str, entity_span: Dict[str, Any]) -> AssertionStatus:
        return AssertionStatus.PRESENT


class ConcreteLinker(BaseLinker):
    def link(self, text: str, label: str) -> Dict[str, Any]:
        return {"concept_id": "1", "concept_name": "Test", "confidence": 1.0}


def test_extractor_interface() -> None:
    extractor = ConcreteExtractor()
    result = extractor.extract("sample text", ["TestLabel"])
    assert result == [{"text": "test", "label": "TestLabel"}]


def test_assertion_detector_interface() -> None:
    detector = ConcreteAssertionDetector()
    result = detector.detect("sample text", {"text": "sample"})
    assert result == AssertionStatus.PRESENT


def test_linker_interface() -> None:
    linker = ConcreteLinker()
    result = linker.link("sample", "Label")
    assert result == {"concept_id": "1", "concept_name": "Test", "confidence": 1.0}
