# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import pytest

from coreason_tagger.assertion_detector import RegexBasedAssertionDetector
from coreason_tagger.schema import AssertionStatus


@pytest.fixture
def detector() -> RegexBasedAssertionDetector:
    return RegexBasedAssertionDetector()


async def test_present_simple(detector: RegexBasedAssertionDetector) -> None:
    text = "Patient has diabetes."
    span = "diabetes"
    start = text.find(span)
    end = start + len(span)
    assert await detector.detect(text, span, start, end) == AssertionStatus.PRESENT


async def test_absent_simple(detector: RegexBasedAssertionDetector) -> None:
    examples = [
        "Patient denies diabetes.",
        "No signs of diabetes.",
        "Negative for diabetes.",
        "Patient is without diabetes.",
        "Diabetes is unlikely.",
    ]
    for text in examples:
        span = "diabetes"
        start = text.find(span)
        end = start + len(span)
        assert await detector.detect(text, span, start, end) == AssertionStatus.ABSENT


async def test_possible_simple(detector: RegexBasedAssertionDetector) -> None:
    examples = [
        "Possible diabetes.",
        "Rule out diabetes.",
        "Suspected diabetes.",
        "Question of diabetes.",
        "Diabetes is likely.",
    ]
    for text in examples:
        span = "diabetes"
        start = text.find(span)
        end = start + len(span)
        assert await detector.detect(text, span, start, end) == AssertionStatus.POSSIBLE


async def test_family_history(detector: RegexBasedAssertionDetector) -> None:
    examples = [
        "Mother had diabetes.",
        "Family history of diabetes.",
        "Father died of diabetes.",
        "Paternal aunt has diabetes.",
    ]
    for text in examples:
        span = "diabetes"
        start = text.find(span)
        end = start + len(span)
        assert await detector.detect(text, span, start, end) == AssertionStatus.FAMILY


async def test_history_personal(detector: RegexBasedAssertionDetector) -> None:
    examples = [
        "History of diabetes.",
        "Past medical history: diabetes.",
        "Status post diabetes treatment.",
        "h/o diabetes",
    ]
    for text in examples:
        span = "diabetes"
        start = text.find(span)
        end = start + len(span)
        assert await detector.detect(text, span, start, end) == AssertionStatus.HISTORY


async def test_conditional(detector: RegexBasedAssertionDetector) -> None:
    examples = [
        "Return if diabetes worsens.",
        "Monitor for diabetes.",
        "Unless diabetes is present.",
    ]
    for text in examples:
        span = "diabetes"
        start = text.find(span)
        end = start + len(span)
        assert await detector.detect(text, span, start, end) == AssertionStatus.CONDITIONAL


async def test_edge_cases_double_negation(detector: RegexBasedAssertionDetector) -> None:
    # "not ruled out" contains "not" (Absent) and "rule out" (Possible)
    # Correct logic should be POSSIBLE
    text = "Diabetes not ruled out."
    span = "Diabetes"
    start = text.find(span)
    end = start + len(span)
    assert await detector.detect(text, span, start, end) == AssertionStatus.POSSIBLE

    text = "Cannot rule out diabetes."
    start = text.find("diabetes")
    end = start + len("diabetes")
    assert await detector.detect(text, "diabetes", start, end) == AssertionStatus.POSSIBLE


async def test_priority_overrides(detector: RegexBasedAssertionDetector) -> None:
    # Family history should override negation
    text = "Mother does not have diabetes."
    span = "diabetes"
    start = text.find(span)
    end = start + len(span)
    assert await detector.detect(text, span, start, end) == AssertionStatus.FAMILY
