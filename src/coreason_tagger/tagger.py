# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import List, Optional

from coreason_tagger.interfaces import BaseAssertionDetector, BaseLinker, BaseNERExtractor
from coreason_tagger.schema import ExtractedSpan, TaggedEntity


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

    def _process_span(self, text: str, span: ExtractedSpan) -> Optional[TaggedEntity]:
        """
        Process a single span: contextualize (assertion) and link.

        Args:
            text: The full context text.
            span: The extracted entity span.

        Returns:
            Optional[TaggedEntity]: The processed entity, or None if linking failed.
        """
        # Guard: If span text is empty, it's useless and will fail validation.
        if not span.text or not span.text.strip():
            return None

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
            return None

        concept_id = link_result.get("concept_id")
        if not concept_id:
            # Malformed result or missing ID -> Useless
            return None

        # Construct the final entity
        return TaggedEntity(
            span_text=span.text,
            label=span.label,
            concept_id=concept_id,
            concept_name=link_result.get("concept_name", ""),
            link_confidence=link_result.get("link_confidence", 0.0),
            assertion=assertion_status,
        )

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
            entity = self._process_span(text, span)
            if entity:
                tagged_entities.append(entity)

        return tagged_entities

    def tag_batch(self, texts: List[str], labels: List[str]) -> List[List[TaggedEntity]]:
        """
        Process a batch of texts to extract, contextualize, and link entities.
        Optimized for batch NER processing.

        Args:
            texts: The list of input texts.
            labels: The list of labels to extract.

        Returns:
            List[List[TaggedEntity]]: A list of lists of processed entities, corresponding to the input texts.
        """
        if not texts:
            return []

        # 1. Batch Extract (NER)
        batch_spans = self.ner.extract_batch(texts, labels)

        batch_results: List[List[TaggedEntity]] = []

        # Process each text's spans
        # zip(strict=True) ensures alignment, though extract_batch guarantees it too
        for text, spans in zip(texts, batch_spans, strict=True):
            tagged_entities: List[TaggedEntity] = []
            for span in spans:
                entity = self._process_span(text, span)
                if entity:
                    tagged_entities.append(entity)
            batch_results.append(tagged_entities)

        return batch_results
