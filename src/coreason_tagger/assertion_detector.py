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
from typing import Any, Dict

from coreason_tagger.interfaces import BaseAssertionDetector
from coreason_tagger.schema import AssertionStatus


class RegexBasedAssertionDetector(BaseAssertionDetector):
    """
    Rule-based implementation of Assertion Detection using regex patterns.
    Determines if an entity is Present, Absent, Possible, Conditional, or Family History.
    """

    # Pre-compiled regex patterns for performance
    PATTERNS = {
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
            r"\bpatient's\b\s+\w+",  # Generic catch-all might be hard with regex without dependency parsing
            # We'll rely mostly on FAMILY for specific relatives
        ],
    }

    def detect(self, text: str, entity_span: Dict[str, Any]) -> AssertionStatus:
        """
        Determines the assertion status of an entity within the text.
        entity_span is a dict containing 'start', 'end', 'text'.

        Logic:
        1. Extract a context window around the entity.
        2. Check for patterns in priority order: FAMILY > ABSENT > POSSIBLE > CONDITIONAL.
        3. Default to PRESENT.
        """
        start = entity_span.get("start", 0)
        end = entity_span.get("end", len(text))

        # simple window: sentence boundary or N chars
        # For this atomic unit, we'll take the whole sentence if possible, or a window of +/- 50 chars
        # But to be robust, let's look at the preceding text mainly, and some following text.

        window_start = max(0, start - 50)
        window_end = min(len(text), end + 20)  # Look slightly ahead for "Condition if..."

        # A better approach for "sentence" might be finding the last period
        pre_text = text[:start]
        post_text = text[end:]

        last_period = pre_text.rfind(".")
        if last_period != -1:
            window_start = last_period + 1

        next_period = post_text.find(".")
        if next_period != -1:
            window_end = end + next_period

        context = text[window_start:window_end].lower()

        # Priority Check

        # 1. Family History
        for pattern in self.PATTERNS[AssertionStatus.FAMILY]:
            if re.search(pattern, context):
                return AssertionStatus.FAMILY

        # 2. Absent (Negated)
        for pattern in self.PATTERNS[AssertionStatus.ABSENT]:
            if re.search(pattern, context):
                return AssertionStatus.ABSENT

        # 3. Possible
        for pattern in self.PATTERNS[AssertionStatus.POSSIBLE]:
            if re.search(pattern, context):
                return AssertionStatus.POSSIBLE

        # 4. Conditional
        for pattern in self.PATTERNS[AssertionStatus.CONDITIONAL]:
            if re.search(pattern, context):
                return AssertionStatus.CONDITIONAL

        # Default
        return AssertionStatus.PRESENT
