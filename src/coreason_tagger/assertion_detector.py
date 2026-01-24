# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import asyncio
import re
from typing import Any, List, Optional

from coreason_tagger.config import settings
from coreason_tagger.interfaces import BaseAssertionDetector
from coreason_tagger.registry import get_assertion_pipeline
from coreason_tagger.schema import AssertionStatus


class RegexBasedAssertionDetector(BaseAssertionDetector):
    """A rule-based assertion detector using regular expressions.

    Prioritizes statuses in a specific order:
    FAMILY > CONDITIONAL > ABSENT > POSSIBLE > HISTORY > PRESENT
    """

    def __init__(self) -> None:
        """Initialize the RegexBasedAssertionDetector and compile patterns."""
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
        """Compile a list of regex patterns into a single optimized pattern.

        Args:
            patterns: List of regex strings.

        Returns:
            re.Pattern: Compiled regex pattern.
        """
        return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)

    def _get_local_context(self, text: str, span_start: int, span_end: int) -> str:
        """Extract the local context (e.g., current clause) around the span.

        Splits by common clause delimiters: . ; ,

        Args:
            text: Full text.
            span_start: Start index of the entity.
            span_end: End index of the entity.

        Returns:
            str: The local clause text.
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

    async def detect(self, text: str, span_text: str, span_start: int, span_end: int) -> AssertionStatus:
        """Detect assertion status by analyzing the window of text around the entity.

        Currently analyzes the local clause context.

        Args:
            text: The full context text.
            span_text: The text of the entity.
            span_start: The start index of the entity.
            span_end: The end index of the entity.

        Returns:
            AssertionStatus: The detected assertion status.
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


class DistilBERTAssertionDetector(BaseAssertionDetector):
    """Assertion detector using a DistilBERT model."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize the DistilBERTAssertionDetector.

        Args:
            model_name: The name of the model to load.
        """
        self.model_name = model_name or settings.ASSERTION_MODEL_NAME
        self.model: Any = None
        # Default mapping for common assertion models
        self.label_map = {
            "present": AssertionStatus.PRESENT,
            "absent": AssertionStatus.ABSENT,
            "possible": AssertionStatus.POSSIBLE,
            "conditional": AssertionStatus.CONDITIONAL,
            "hypothetical": AssertionStatus.POSSIBLE,
            "associated_with_someone_else": AssertionStatus.FAMILY,
            "family": AssertionStatus.FAMILY,
            "history": AssertionStatus.HISTORY,
            # Fallbacks for generic models
            "label_0": AssertionStatus.ABSENT,
            "label_1": AssertionStatus.PRESENT,
        }

    async def load_model(self) -> None:
        """Lazy load the model pipeline."""
        if self.model is not None:
            return
        self.model = await get_assertion_pipeline(self.model_name)

    async def detect(self, text: str, span_text: str, span_start: int, span_end: int) -> AssertionStatus:
        """Detect assertion status using the loaded model.

        Args:
            text: The full context text.
            span_text: The text of the entity.
            span_start: The start index of the entity.
            span_end: The end index of the entity.

        Returns:
            AssertionStatus: The detected assertion status.
        """
        if self.model is None:
            await self.load_model()

        # Mark the entity with special tokens
        # Note: If the model uses specific tokens (like [entity]), they should be preserved.
        pre = text[:span_start]
        post = text[span_end:]
        formatted_text = f"{pre}[entity] {span_text} [/entity]{post}"

        # Run inference in executor to avoid blocking
        # pipeline returns a list of dicts: [{'label': '...', 'score': ...}]
        result = await asyncio.to_thread(self.model, formatted_text, truncation=True)

        if not result:
            return AssertionStatus.PRESENT

        # Handle output format (list of dicts or dict)
        top_result = result[0] if isinstance(result, list) else result
        label = top_result.get("label", "").lower()

        return self.label_map.get(label, AssertionStatus.PRESENT)
