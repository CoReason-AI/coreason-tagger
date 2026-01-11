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

    # --- Complex & Edge Cases ---

    def test_conditional_negation(self, detector: RegexBasedAssertionDetector) -> None:
        # "If no signs of infection" -> CONDITIONAL (not ABSENT)
        text = "If no signs of infection are present."
        span = {"start": 15, "end": 24, "text": "infection"}
        # Priority: CONDITIONAL > ABSENT
        assert detector.detect(text, span) == AssertionStatus.CONDITIONAL

    def test_not_ruled_out(self, detector: RegexBasedAssertionDetector) -> None:
        # "Pneumonia is not ruled out." -> POSSIBLE (Double negation / Override)
        text = "Pneumonia is not ruled out."
        span = {"start": 0, "end": 9, "text": "Pneumonia"}
        assert detector.detect(text, span) == AssertionStatus.POSSIBLE

    def test_ruled_out(self, detector: RegexBasedAssertionDetector) -> None:
        # "Pneumonia is ruled out." -> ABSENT
        text = "Pneumonia is ruled out."
        span = {"start": 0, "end": 9, "text": "Pneumonia"}
        assert detector.detect(text, span) == AssertionStatus.ABSENT

    def test_cannot_rule_out(self, detector: RegexBasedAssertionDetector) -> None:
        # "Cannot rule out tumor." -> POSSIBLE
        text = "Cannot rule out tumor."
        span = {"start": 16, "end": 21, "text": "tumor"}
        assert detector.detect(text, span) == AssertionStatus.POSSIBLE

    def test_unlikely(self, detector: RegexBasedAssertionDetector) -> None:
        # "It is unlikely to be cancer." -> ABSENT
        text = "It is unlikely to be cancer."
        span = {"start": 21, "end": 27, "text": "cancer"}
        assert detector.detect(text, span) == AssertionStatus.ABSENT

    def test_no_evidence_of(self, detector: RegexBasedAssertionDetector) -> None:
        # "No evidence of fracture." -> ABSENT
        text = "No evidence of fracture."
        span = {"start": 15, "end": 23, "text": "fracture"}
        assert detector.detect(text, span) == AssertionStatus.ABSENT

    def test_family_negation(self, detector: RegexBasedAssertionDetector) -> None:
        # "No family history of heart disease." -> FAMILY (It's a family history assertion, even if negative)
        text = "No family history of heart disease."
        span = {"start": 21, "end": 34, "text": "heart disease"}
        assert detector.detect(text, span) == AssertionStatus.FAMILY

    def test_conditional_family(self, detector: RegexBasedAssertionDetector) -> None:
        # "If mother has diabetes..." -> FAMILY (Context is mother, overrides conditional?)
        # Logic: FAMILY > CONDITIONAL.
        text = "If mother has diabetes check levels."
        span = {"start": 14, "end": 22, "text": "diabetes"}
        assert detector.detect(text, span) == AssertionStatus.FAMILY
