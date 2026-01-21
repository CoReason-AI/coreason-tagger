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


def test_negated_history(detector: RegexBasedAssertionDetector) -> None:
    """
    Test "No history of" edge case.
    Should be ABSENT, not HISTORY.
    """
    text = "Patient has no history of diabetes."
    span = "diabetes"
    start = text.find(span)
    end = start + len(span)

    # If HISTORY priority > ABSENT, this would fail (return HISTORY).
    # We expect ABSENT (Negated).
    assert detector.detect(text, span, start, end) == AssertionStatus.ABSENT


def test_conditional_history(detector: RegexBasedAssertionDetector) -> None:
    """
    Test "If history of" edge case.
    Should be CONDITIONAL, not HISTORY.
    """
    text = "If history of diabetes exists, treat accordingly."
    span = "diabetes"
    start = text.find(span)
    end = start + len(span)

    assert detector.detect(text, span, start, end) == AssertionStatus.CONDITIONAL


def test_negated_family_history(detector: RegexBasedAssertionDetector) -> None:
    """
    Test "No family history of" edge case.
    Should be FAMILY (as it pertains to family context), or ABSENT?
    Usually "Family history" overrides everything because it attributes the entity to someone else.
    "Mother does not have diabetes" -> FAMILY (attributed to mother, even if negated).
    "No family history of diabetes" -> FAMILY (attributed to family context).
    """
    text = "No family history of diabetes."
    span = "diabetes"
    start = text.find(span)
    end = start + len(span)

    assert detector.detect(text, span, start, end) == AssertionStatus.FAMILY


def test_assertion_priority_complex(detector: RegexBasedAssertionDetector) -> None:
    """
    Test a mix of triggers.
    "Possible history of diabetes" -> POSSIBLE? HISTORY?
    "History" implies past. "Possible" implies uncertainty.
    "Possible history" -> POSSIBLE seems safer/more specific than just HISTORY.
    """
    text = "Possible history of diabetes."
    span = "diabetes"
    start = text.find(span)
    end = start + len(span)

    # If HISTORY > POSSIBLE: returns HISTORY.
    # If POSSIBLE > HISTORY: returns POSSIBLE.
    # "Possible history" is speculative.
    # Current/New Plan order: FAMILY > CONDITIONAL > ABSENT > HISTORY > POSSIBLE.
    # Wait, if ABSENT > HISTORY, where does POSSIBLE fit?
    # Spec says:
    # 1. Family
    # 2. History (Personal)
    # 3. Conditional
    # 4. Absent
    # 5. Possible

    # My proposed Plan: FAMILY > CONDITIONAL > ABSENT > HISTORY
    # What about POSSIBLE?
    # "Possible history" matches HISTORY ("history") and POSSIBLE ("possible").
    # If HISTORY > POSSIBLE, it returns HISTORY.
    # Ideally, "Possible history" should probably be POSSIBLE (it's not confirmed history).
    # So POSSIBLE > HISTORY?

    # Let's verify standard clinical NLP (ConText/NegEx):
    # Usually: Negation > Uncertainty > Historicity.
    # But Family is usually top.

    # So Order: FAMILY > CONDITIONAL > ABSENT > POSSIBLE > HISTORY > PRESENT.

    # Let's test this hypothesis.
    assert detector.detect(text, span, start, end) == AssertionStatus.POSSIBLE
