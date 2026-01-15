# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import List

from coreason_tagger.interfaces import BaseAssertionDetector, BaseLinker, BaseNERExtractor
from coreason_tagger.schema import TaggedEntity


class CoreasonTagger:
    """
    The orchestrator for the coreason-tagger pipeline.
    Implements the Extract-Contextualize-Link loop.
    """

    def __init__(
        self,
        ner: BaseNERExtractor,
        assertion: BaseAssertionDetector,
        linker: BaseLinker,
    ) -> None:
        """
        Initialize the Tagger with its dependencies.

        Args:
            ner: The NER extractor (e.g., GLiNER).
            assertion: The assertion detector.
            linker: The entity linker.
        """
        self.ner = ner
        self.assertion = assertion
        self.linker = linker

    def tag(self, text: str, labels: List[str]) -> List[TaggedEntity]:
        """
        Process text to extract, contextualize, and link entities.

        Args:
            text: The input text.
            labels: The list of labels to extract (passed to NER).

        Returns:
            List[TaggedEntity]: The list of fully processed entities.
        """
        if not text:
            return []

        # 1. Extract (NER)
        spans = self.ner.extract(text, labels)

        tagged_entities: List[TaggedEntity] = []

        for span in spans:
            # 2. Contextualize (Assertion)
            assertion_status = self.assertion.detect(
                text=text,
                span_text=span.text,
                span_start=span.start,
                span_end=span.end,
            )

            # 3. Link (Vector Linking)
            link_result = self.linker.link(span)

            # If linking fails (returns empty dict), we skip this entity
            # per the "A string without an ID is useless" philosophy.
            if not link_result:
                continue

            concept_id = link_result.get("concept_id")
            if not concept_id:
                # Malformed result or missing ID -> Useless
                continue

            # Construct the final entity
            entity = TaggedEntity(
                span_text=span.text,
                label=span.label,
                concept_id=concept_id,
                concept_name=link_result.get("concept_name", ""),
                link_confidence=link_result.get("link_confidence", 0.0),
                assertion=assertion_status,
            )
            tagged_entities.append(entity)

        return tagged_entities
