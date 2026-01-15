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


def test_present_simple(detector: RegexBasedAssertionDetector) -> None:
    text = "Patient has diabetes."
    span = "diabetes"
    start = text.find(span)
    end = start + len(span)
    assert detector.detect(text, span, start, end) == AssertionStatus.PRESENT


def test_absent_simple(detector: RegexBasedAssertionDetector) -> None:
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
        assert detector.detect(text, span, start, end) == AssertionStatus.ABSENT


def test_possible_simple(detector: RegexBasedAssertionDetector) -> None:
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
        assert detector.detect(text, span, start, end) == AssertionStatus.POSSIBLE


def test_family_history(detector: RegexBasedAssertionDetector) -> None:
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
        assert detector.detect(text, span, start, end) == AssertionStatus.FAMILY


def test_conditional(detector: RegexBasedAssertionDetector) -> None:
    examples = [
        "Return if diabetes worsens.",
        "Monitor for diabetes.",
        "Unless diabetes is present.",
    ]
    for text in examples:
        span = "diabetes"
        start = text.find(span)
        end = start + len(span)
        assert detector.detect(text, span, start, end) == AssertionStatus.CONDITIONAL


def test_associated_with_someone_else(detector: RegexBasedAssertionDetector) -> None:
    examples = [
        "Spouse has diabetes.",
        "Husband diagnosed with diabetes.",
    ]
    for text in examples:
        span = "diabetes"
        start = text.find(span)
        end = start + len(span)
        assert detector.detect(text, span, start, end) == AssertionStatus.ASSOCIATED_WITH_SOMEONE_ELSE


def test_edge_cases_double_negation(detector: RegexBasedAssertionDetector) -> None:
    # "not ruled out" contains "not" (Absent) and "rule out" (Possible)
    # Correct logic should be POSSIBLE
    text = "Diabetes not ruled out."
    span = "Diabetes"
    start = text.find(span)
    end = start + len(span)
    assert detector.detect(text, span, start, end) == AssertionStatus.POSSIBLE

    text = "Cannot rule out diabetes."
    start = text.find("diabetes")
    end = start + len("diabetes")
    assert detector.detect(text, "diabetes", start, end) == AssertionStatus.POSSIBLE


def test_priority_overrides(detector: RegexBasedAssertionDetector) -> None:
    # Family history should override negation?
    # e.g., "Mother does not have diabetes" -> Technically Family + Absent.
    # The PRD says "Assertion detects 'mother' ... sets status: ASSOCIATED_WITH_SOMEONE_ELSE / FAMILY"
    # Usually FAMILY takes precedence over simply ABSENT in the context of the *patient*.
    # If the mother doesn't have it, it's still a FAMILY assertion (negative family history).
    # Current logic prioritizes FAMILY first.

    text = "Mother does not have diabetes."
    span = "diabetes"
    start = text.find(span)
    end = start + len(span)
    assert detector.detect(text, span, start, end) == AssertionStatus.FAMILY
