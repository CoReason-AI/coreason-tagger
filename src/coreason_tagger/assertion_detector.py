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
from typing import List

from coreason_tagger.interfaces import BaseAssertionDetector
from coreason_tagger.schema import AssertionStatus


class RegexBasedAssertionDetector(BaseAssertionDetector):
    """
    A rule-based assertion detector using regular expressions.
    Prioritizes statuses in a specific order:
    FAMILY > ASSOCIATED > CONDITIONAL > ABSENT > POSSIBLE > PRESENT
    """

    def __init__(self) -> None:
        # Compile patterns for efficiency
        # Note: Order matters implicitly if we check sequentially, but we define explicit priority below.

        # Family History
        self.family_patterns = [
            r"\b(mother|father|brother|sister|grandmother|grandfather|aunt|uncle|parent|sibling)s?\b",
            r"\bfamily history\b",
            r"\bmaternal\b",
            r"\bpaternal\b",
        ]

        # Associated with someone else (non-family usually, or broad context)
        # "Patient's husband has..."
        self.associated_patterns = [
            r"\b(husband|wife|spouse|partner|friend|neighbor|colleague)s?\b",
            r"\bdaughter|son\b",  # Children are family, but often treated as "someone else" context in some schemas.
            # We'll map them to FAMILY if strictly genetic, but here we group "other people".
            # For this PRD, "Mother" -> Family. Let's stick to the PRD examples.
        ]

        # Conditional / Hypothetical
        self.conditional_patterns = [
            r"\bif\b",
            r"\bunless\b",
            r"\bshould\b",
            r"\breturn if\b",
            r"\bmonitor for\b",
        ]

        # Absent (Negation)
        self.absent_patterns = [
            r"\bno\b",
            r"\bnot\b",
            r"\bdenies\b",
            r"\bdenied\b",
            r"\bwithout\b",
            r"\bfree of\b",
            r"\bnegative for\b",
            r"\bunlikely\b",
            r"\brules out\b",  # "This rules out X" -> X is Absent
        ]

        # Possible (Uncertainty)
        self.possible_patterns = [
            r"\bpossible\b",
            r"\bprobable\b",
            r"\blikely\b",
            r"\brule out\b",  # "Rule out X" -> X is Possible/Hypothetical target
            r"\bsuspect\b",
            r"\bsuspected\b",
            r"\bquestion of\b",
            r"\bcannot rule out\b",
            r"\bnot ruled out\b",  # Double negation-ish context
        ]

    def _matches_any(self, text: str, patterns: List[str]) -> bool:
        """Check if any pattern matches the text (case-insensitive)."""
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                return True
        return False

    def detect(self, text: str, span_text: str, span_start: int, span_end: int) -> AssertionStatus:
        """
        Detect assertion status by analyzing the window of text around the entity.
        Currently analyzes the whole sentence or a reasonable window.
        """

        # Simple windowing strategy: look at the whole text provided (assuming it's a sentence)
        # In a real system, we might dependency parse, but regex looks at the linear context.

        # To avoid matching the entity itself (if it contains trigger words like "failure"?? unlikely),
        # we focus on the context. But for simple regex, checking the whole string is the starting point.
        # Ideally, we should check "preceding" and "following" context.

        # pre_context = text[:span_start]
        # post_context = text[span_end:]

        # We consider a window of roughly 5-10 words before the mention as the most potent area for assertion triggers.
        # But for this implementation, we search the full 'text' provided (assuming the caller passes a sentence).

        # Check specific edge cases first (override general patterns)

        # Double Negation / Complex Logic
        # "not ruled out" -> POSSIBLE (contains "not" (absent) and "rule out" (possible))
        if re.search(r"\bnot ruled out\b", text, re.IGNORECASE):
            return AssertionStatus.POSSIBLE

        if re.search(r"\bcannot rule out\b", text, re.IGNORECASE):
            return AssertionStatus.POSSIBLE

        # Priority Check

        # 1. Family History (Strongest override usually)
        if self._matches_any(text, self.family_patterns):
            return AssertionStatus.FAMILY

        # 2. Associated with someone else
        if self._matches_any(text, self.associated_patterns):
            return AssertionStatus.ASSOCIATED_WITH_SOMEONE_ELSE

        # 3. Conditional
        if self._matches_any(text, self.conditional_patterns):
            return AssertionStatus.CONDITIONAL

        # 4. Absent (Negation)
        # Note: We need to be careful not to trigger on "not ruled out" here, but we handled that edge case above.
        if self._matches_any(text, self.absent_patterns):
            return AssertionStatus.ABSENT

        # 5. Possible
        if self._matches_any(text, self.possible_patterns):
            return AssertionStatus.POSSIBLE

        # Default
        return AssertionStatus.PRESENT
