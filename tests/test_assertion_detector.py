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


class TestRegexBasedAssertionDetector:
    @pytest.fixture  # type: ignore
    def detector(self) -> RegexBasedAssertionDetector:
        return RegexBasedAssertionDetector()

    def test_present(self, detector: RegexBasedAssertionDetector) -> None:
        text = "Patient has severe headaches."
        span = {"start": 12, "end": 28, "text": "severe headaches"}
        assert detector.detect(text, span) == AssertionStatus.PRESENT

    def test_absent_no(self, detector: RegexBasedAssertionDetector) -> None:
        text = "Patient has no headaches."
        span = {"start": 15, "end": 24, "text": "headaches"}
        assert detector.detect(text, span) == AssertionStatus.ABSENT

    def test_absent_denies(self, detector: RegexBasedAssertionDetector) -> None:
        text = "Patient denies any chest pain."
        span = {"start": 19, "end": 29, "text": "chest pain"}
        assert detector.detect(text, span) == AssertionStatus.ABSENT

    def test_family_mother(self, detector: RegexBasedAssertionDetector) -> None:
        text = "Mother had breast cancer."
        span = {"start": 11, "end": 24, "text": "breast cancer"}
        assert detector.detect(text, span) == AssertionStatus.FAMILY

    def test_family_history(self, detector: RegexBasedAssertionDetector) -> None:
        text = "Family history of diabetes."
        span = {"start": 18, "end": 26, "text": "diabetes"}
        assert detector.detect(text, span) == AssertionStatus.FAMILY

    def test_possible(self, detector: RegexBasedAssertionDetector) -> None:
        text = "Rule out pneumonia."
        span = {"start": 9, "end": 18, "text": "pneumonia"}
        assert detector.detect(text, span) == AssertionStatus.POSSIBLE

    def test_conditional(self, detector: RegexBasedAssertionDetector) -> None:
        text = "If symptoms persist, return immediately."
        span = {"start": 3, "end": 11, "text": "symptoms"}
        assert detector.detect(text, span) == AssertionStatus.CONDITIONAL

    def test_priority_family_over_absent(self, detector: RegexBasedAssertionDetector) -> None:
        # "Mother has no history of X" -> Still Family context usually,
        # but technically "Mother" implies it's about the mother.
        # However, "Mother has no cancer" -> FAMILY (Associated with someone else)
        # Our logic prioritizes FAMILY check first.
        text = "Mother has no diabetes."
        span = {"start": 14, "end": 22, "text": "diabetes"}
        assert detector.detect(text, span) == AssertionStatus.FAMILY

    def test_window_logic(self, detector: RegexBasedAssertionDetector) -> None:
        # "Patient has diabetes. No headache."
        # Testing "headache". "No" is in the same sentence/window. -> ABSENT
        text = "Patient has diabetes. No headache."
        span = {"start": 25, "end": 33, "text": "headache"}
        assert detector.detect(text, span) == AssertionStatus.ABSENT

        # Testing "diabetes". "No" is in the next sentence.
        # "Patient has diabetes. No headache."
        diabetes_span = {"start": 12, "end": 20, "text": "diabetes"}
        assert detector.detect(text, diabetes_span) == AssertionStatus.PRESENT

    def test_default(self, detector: RegexBasedAssertionDetector) -> None:
        text = "Just a random sentence with a finding."
        span = {"start": 28, "end": 35, "text": "finding"}
        assert detector.detect(text, span) == AssertionStatus.PRESENT
