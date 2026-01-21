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
    FAMILY > CONDITIONAL > ABSENT > POSSIBLE > HISTORY > PRESENT
    """

    def __init__(self) -> None:
        # Compile patterns for efficiency
        # We join patterns with | to create a single regex for each category
        # This avoids iterating through lists of strings

        self.family_regex = self._compile(
            [
                r"\b(mother|father|brother|sister|grandmother|grandfather|aunt|uncle|parent|sibling)s?\b",
                r"\bfamily history\b",
                r"\bmaternal\b",
                r"\bpaternal\b",
            ]
        )

        self.history_regex = self._compile(
            [
                r"\bhistory of\b",
                r"\bh/o\b",
                r"\bpast medical history\b",
                r"\bstatus post\b",
                r"\bprevious\b",
            ]
        )

        self.conditional_regex = self._compile(
            [
                r"\bif\b",
                r"\bunless\b",
                r"\bshould\b",
                r"\breturn if\b",
                r"\bmonitor for\b",
            ]
        )

        self.absent_regex = self._compile(
            [
                r"\bno\b",
                r"\bnot\b",
                r"\bdenies\b",
                r"\bdenied\b",
                r"\bwithout\b",
                r"\bfree of\b",
                r"\bnegative for\b",
                r"\bunlikely\b",
                r"\brules out\b",
            ]
        )

        self.possible_regex = self._compile(
            [
                r"\bpossible\b",
                r"\bprobable\b",
                r"\blikely\b",
                r"\brule out\b",
                r"\bsuspect\b",
                r"\bsuspected\b",
                r"\bquestion of\b",
                r"\bcannot rule out\b",
                r"\bnot ruled out\b",
            ]
        )

        # Specific edge case patterns (compiled separately)
        self.not_ruled_out_regex = re.compile(r"\bnot ruled out\b", re.IGNORECASE)
        self.cannot_rule_out_regex = re.compile(r"\bcannot rule out\b", re.IGNORECASE)

    def _compile(self, patterns: List[str]) -> re.Pattern[str]:
        """Compile a list of regex patterns into a single optimized pattern."""
        return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)

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
        if self.not_ruled_out_regex.search(context_text):
            return AssertionStatus.POSSIBLE

        if self.cannot_rule_out_regex.search(context_text):
            return AssertionStatus.POSSIBLE

        # Priority Check (Ordered)

        if self.family_regex.search(context_text):
            return AssertionStatus.FAMILY

        if self.conditional_regex.search(context_text):
            return AssertionStatus.CONDITIONAL

        if self.absent_regex.search(context_text):
            return AssertionStatus.ABSENT

        if self.possible_regex.search(context_text):
            return AssertionStatus.POSSIBLE

        if self.history_regex.search(context_text):
            return AssertionStatus.HISTORY

        # Default
        return AssertionStatus.PRESENT
