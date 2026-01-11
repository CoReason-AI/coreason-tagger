# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import re
from typing import Any, Dict, List

from coreason_tagger.interfaces import BaseAssertionDetector
from coreason_tagger.schema import AssertionStatus


class RegexBasedAssertionDetector(BaseAssertionDetector):
    """
    Rule-based implementation of Assertion Detection using regex patterns.
    Determines if an entity is Present, Absent, Possible, Conditional, or Family History.
    """

    # Pre-compiled regex patterns for performance
    PATTERNS: Dict[AssertionStatus, List[str]] = {
        AssertionStatus.FAMILY: [
            r"\bmother\b",
            r"\bfather\b",
            r"\bbrother\b",
            r"\bsister\b",
            r"\bfamily history\b",
            r"\buncle\b",
            r"\baunt\b",
            r"\bgrandfather\b",
            r"\bgrandmother\b",
        ],
        AssertionStatus.ABSENT: [
            r"\bno\b",
            r"\bnot\b",
            r"\bdenies\b",
            r"\bdenied\b",
            r"\bnegative for\b",
            r"\bfree of\b",
            r"\bwithout\b",
            r"\bruled out\b",
            r"\bunlikely\b",
            r"\bno evidence of\b",
        ],
        AssertionStatus.POSSIBLE: [
            r"\bpossible\b",
            r"\bprobable\b",
            r"\blikely\b",
            r"\bsuspect\b",
            r"\bsuspected\b",
            r"\brule out\b",  # ambiguous, but usually implies checking for it
            r"\bquestion of\b",
        ],
        AssertionStatus.CONDITIONAL: [
            r"\bif\b",
            r"\bunless\b",
            r"\bshould\b",  # e.g., "should symptoms worsen"
        ],
        AssertionStatus.ASSOCIATED_WITH_SOMEONE_ELSE: [
            r"\bpatient's\b\s+\w+",
        ],
    }

    # Override patterns that should be checked first and return a specific status
    # This helps with double negation or complex phrases like "not ruled out"
    OVERRIDES: Dict[str, AssertionStatus] = {
        r"\bnot ruled out\b": AssertionStatus.POSSIBLE,
        r"\bnot unlikely\b": AssertionStatus.POSSIBLE,  # "Not unlikely" -> Possible
        r"\bcannot rule out\b": AssertionStatus.POSSIBLE,
    }

    def detect(self, text: str, entity_span: Dict[str, Any]) -> AssertionStatus:
        """
        Determines the assertion status of an entity within the text.
        entity_span is a dict containing 'start', 'end', 'text'.

        Logic:
        1. Extract a context window around the entity.
        2. Check for Overrides first.
        3. Check for patterns in priority order: FAMILY > CONDITIONAL > ABSENT > POSSIBLE.
           (Changed order: Conditional > Absent to catch 'if no...')
        4. Default to PRESENT.
        """
        start = entity_span.get("start", 0)
        end = entity_span.get("end", len(text))

        # simple window: sentence boundary or N chars
        # For this atomic unit, we'll take the whole sentence if possible, or a window of +/- 50 chars
        # But to be robust, let's look at the preceding text mainly, and some following text.

        window_start = max(0, start - 60)
        window_end = min(len(text), end + 30)  # Look slightly ahead for "Condition if..."

        # A better approach for "sentence" might be finding the last period
        pre_text = text[:start]
        post_text = text[end:]

        last_period = pre_text.rfind(".")
        if last_period != -1 and last_period >= window_start:
            window_start = last_period + 1

        next_period = post_text.find(".")
        if next_period != -1 and (end + next_period) <= window_end:
            window_end = end + next_period

        context = text[window_start:window_end].lower()

        # 0. Check Overrides (Double Negations, etc.)
        for pattern, status in self.OVERRIDES.items():
            if re.search(pattern, context):
                return status

        # Priority Check (Updated)

        # 1. Family History (Context is someone else)
        for pattern in self.PATTERNS[AssertionStatus.FAMILY]:
            if re.search(pattern, context):
                return AssertionStatus.FAMILY

        # 2. Conditional (Overrides simple negation: "If no symptoms...")
        for pattern in self.PATTERNS[AssertionStatus.CONDITIONAL]:
            if re.search(pattern, context):
                return AssertionStatus.CONDITIONAL

        # 3. Absent (Negated)
        for pattern in self.PATTERNS[AssertionStatus.ABSENT]:
            if re.search(pattern, context):
                return AssertionStatus.ABSENT

        # 4. Possible
        for pattern in self.PATTERNS[AssertionStatus.POSSIBLE]:
            if re.search(pattern, context):
                return AssertionStatus.POSSIBLE

        # Default
        return AssertionStatus.PRESENT
