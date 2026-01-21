# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Optional

from coreason_tagger.interfaces import BaseAssertionDetector, BaseLinker, BaseNERExtractor
from coreason_tagger.schema import EntityCandidate, ExtractionStrategy, LinkedEntity


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

    def _process_candidate(
        self, text: str, candidate: EntityCandidate, strategy: ExtractionStrategy
    ) -> Optional[LinkedEntity]:
        """
        Process a single candidate: contextualize (assertion) and link.

        Args:
            text: The full context text.
            candidate: The extracted entity candidate.
            strategy: The extraction strategy used.

        Returns:
            Optional[LinkedEntity]: The processed entity, or None if linking failed.
        """
        # Guard: If span text is empty, it's useless and will fail validation.
        if not candidate.text or not candidate.text.strip():
            return None

        # 2. Contextualize (Assertion)
        assertion_status = self.assertion.detect(
            text=text,
            span_text=candidate.text,
            span_start=candidate.start,
            span_end=candidate.end,
        )

        # 3. Link (Vector Linking)
        linked_entity = self.linker.resolve(candidate, text, strategy)

        # If linking fails (concept_id is None), we skip this entity
        if not linked_entity.concept_id:
            return None

        # Update assertion status (Linker returns default PRESENT)
        linked_entity.assertion = assertion_status

        return linked_entity

    def tag(
        self,
        text: str,
        labels: list[str],
        strategy: ExtractionStrategy = ExtractionStrategy.SPEED_GLINER,
    ) -> list[LinkedEntity]:
        """
        Process text to extract, contextualize, and link entities.

        Args:
            text: The input text.
            labels: The list of labels to extract (passed to NER).
            strategy: The strategy to attribute to the entities.

        Returns:
            list[LinkedEntity]: The list of fully processed entities.
        """
        if not text:
            return []

        # 1. Extract (NER)
        candidates = self.ner.extract(text, labels)

        linked_entities: list[LinkedEntity] = []

        for candidate in candidates:
            entity = self._process_candidate(text, candidate, strategy)
            if entity:
                linked_entities.append(entity)

        return linked_entities

    def tag_batch(
        self,
        texts: list[str],
        labels: list[str],
        strategy: ExtractionStrategy = ExtractionStrategy.SPEED_GLINER,
    ) -> list[list[LinkedEntity]]:
        """
        Process a batch of texts to extract, contextualize, and link entities.
        Optimized for batch NER processing.

        Args:
            texts: The list of input texts.
            labels: The list of labels to extract.
            strategy: The strategy to attribute to the entities.

        Returns:
            list[list[LinkedEntity]]: A list of lists of processed entities, corresponding to the input texts.
        """
        if not texts:
            return []

        # 1. Batch Extract (NER)
        batch_candidates = self.ner.extract_batch(texts, labels)

        batch_results: list[list[LinkedEntity]] = []

        # Process each text's candidates
        for text, candidates in zip(texts, batch_candidates, strict=True):
            linked_entities: list[LinkedEntity] = []
            for candidate in candidates:
                entity = self._process_candidate(text, candidate, strategy)
                if entity:
                    linked_entities.append(entity)
            batch_results.append(linked_entities)

        return batch_results
