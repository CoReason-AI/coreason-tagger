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

    def _get_local_context(self, text: str, span_start: int, span_end: int) -> str:
        """
        Extract the local context (e.g., current clause) around the span.
        Splits by common clause delimiters: . ; ,
        """
        # Find clause boundaries
        # Look backwards for delimiter or start
        delimiters = r"[.;,]"

        # Search backwards from span_start
        pre_text = text[:span_start]
        match_pre = list(re.finditer(delimiters, pre_text))
        start_idx = match_pre[-1].end() if match_pre else 0

        # Search forwards from span_end
        post_text = text[span_end:]
        match_post = re.search(delimiters, post_text)
        end_idx = (span_end + match_post.start()) if match_post else len(text)

        return text[start_idx:end_idx].strip()

    def detect(self, text: str, span_text: str, span_start: int, span_end: int) -> AssertionStatus:
        """
        Detect assertion status by analyzing the window of text around the entity.
        Currently analyzes the local clause context.
        """
        # Use local context (clause) instead of full text to avoid cross-contamination
        context_text = self._get_local_context(text, span_start, span_end)

        # Check specific edge cases first (override general patterns)

        # Double Negation / Complex Logic
        # "not ruled out" -> POSSIBLE (contains "not" (absent) and "rule out" (possible))
        if re.search(r"\bnot ruled out\b", context_text, re.IGNORECASE):
            return AssertionStatus.POSSIBLE

        if re.search(r"\bcannot rule out\b", context_text, re.IGNORECASE):
            return AssertionStatus.POSSIBLE

        # Priority Check

        # 1. Family History (Strongest override usually)
        if self._matches_any(context_text, self.family_patterns):
            return AssertionStatus.FAMILY

        # 2. Associated with someone else
        if self._matches_any(context_text, self.associated_patterns):
            return AssertionStatus.ASSOCIATED_WITH_SOMEONE_ELSE

        # 3. Conditional
        if self._matches_any(context_text, self.conditional_patterns):
            return AssertionStatus.CONDITIONAL

        # 4. Absent (Negation)
        # Note: We need to be careful not to trigger on "not ruled out" here, but we handled that edge case above.
        if self._matches_any(context_text, self.absent_patterns):
            return AssertionStatus.ABSENT

        # 5. Possible
        if self._matches_any(context_text, self.possible_patterns):
            return AssertionStatus.POSSIBLE

        # Default
        return AssertionStatus.PRESENT
