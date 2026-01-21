# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Any, Dict, Optional

from gliner import GLiNER

from coreason_tagger.config import settings
from coreason_tagger.interfaces import BaseExtractor
from coreason_tagger.registry import get_gliner_model, get_nuner_pipeline
from coreason_tagger.schema import EntityCandidate, ExtractionStrategy


class GLiNERExtractor(BaseExtractor):
    """
    Zero-Shot NER Extractor using the GLiNER library.
    Wraps the underlying model to provide a clean interface for extracting entities.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the GLiNER extractor.

        Args:
            model_name (str, optional): The name of the GLiNER model to load.
                                        If None, uses the value from settings.NER_MODEL_NAME.
        """
        self.model_name = model_name or settings.NER_MODEL_NAME
        self.model: Optional[GLiNER] = None

    async def load_model(self) -> None:
        """
        Lazy loading of weights to VRAM via the registry (Singleton).
        """
        if self.model is not None:
            return

        self.model = await get_gliner_model(self.model_name)

    def _build_candidate(self, entity: dict[str, Any]) -> EntityCandidate:
        """
        Helper to convert a raw dictionary from GLiNER into an EntityCandidate.

        Args:
            entity (dict[str, Any]): Raw entity dictionary containing 'text', 'label', 'start', 'end', 'score'.

        Returns:
            EntityCandidate: The typed entity candidate.
        """
        return EntityCandidate(
            text=entity["text"],
            label=entity["label"],
            start=entity["start"],
            end=entity["end"],
            confidence=entity["score"],
            source_model=self.model_name,
        )

    async def extract(self, text: str, labels: list[str], threshold: float = 0.3) -> list[EntityCandidate]:
        """
        Extract entities from text using the provided labels.

        Args:
            text (str): The input text to process.
            labels (list[str]): A list of entity types to detect.
            threshold (float): The confidence threshold. Defaults to 0.3.

        Returns:
            list[EntityCandidate]: A list of detected entity candidates.
        """
        self.validate_threshold(threshold)

        if not text or not labels:
            return []

        if self.model is None:
            await self.load_model()

        # GLiNER returns a list of dicts:
        # [{'start': 0, 'end': 5, 'text': '...', 'label': '...', 'score': 0.95}, ...]
        assert self.model is not None
        raw_entities = await self.run_in_executor(self.model.predict_entities, text, labels, threshold=threshold)

        return [self._build_candidate(entity) for entity in raw_entities]

    async def extract_batch(
        self, texts: list[str], labels: list[str], threshold: float = 0.3
    ) -> list[list[EntityCandidate]]:
        """
        Extract entities from a batch of texts using the provided labels.

        Args:
            texts (list[str]): The list of input texts to process.
            labels (list[str]): A list of entity types to detect.
            threshold (float): The confidence threshold. Defaults to 0.3.

        Returns:
            list[list[EntityCandidate]]: A list of lists, where each inner list contains
                                       detected entity candidates for the corresponding text.
        """
        self.validate_threshold(threshold)

        if not texts or not labels:
            return [[] for _ in texts]

        if self.model is None:
            await self.load_model()

        # Use batch_predict_entities if available.
        # batch_predict_entities returns a list of lists of dicts.
        assert self.model is not None
        batch_raw_entities = await self.run_in_executor(
            self.model.batch_predict_entities, texts, labels, threshold=threshold
        )

        batch_extracted_candidates: list[list[EntityCandidate]] = []

        # Iterate over results
        # Use strict=True to ensure the model returns results for every input text
        # (Though we don't strictly need 'texts' for iteration anymore, zip is safer validation)
        for _, raw_entities in zip(texts, batch_raw_entities, strict=True):
            extracted_candidates = [self._build_candidate(entity) for entity in raw_entities]
            batch_extracted_candidates.append(extracted_candidates)

        return batch_extracted_candidates


class NuNERExtractor(BaseExtractor):
    """
    Precision NER Extractor using NuNER Zero (via transformers pipeline).
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the NuNER extractor.

        Args:
            model_name (str, optional): The name of the NuNER model to load.
                                        If None, uses the value from settings.NUNER_MODEL_NAME.
        """
        self.model_name = model_name or settings.NUNER_MODEL_NAME
        self.model: Any = None  # The pipeline object

    async def load_model(self) -> None:
        """
        Lazy loading of weights to VRAM via the registry (Singleton).
        """
        if self.model is not None:
            return

        self.model = await get_nuner_pipeline(self.model_name)

    def _build_candidate(self, entity: dict[str, Any]) -> EntityCandidate:
        """
        Helper to convert a raw dictionary from transformers pipeline into an EntityCandidate.

        Args:
            entity (dict[str, Any]): Raw entity dictionary containing 'word', 'entity_group', 'start', 'end', 'score'.
                                     Note: 'word' is the text span,
                                           'entity_group' is the label (when aggregation_strategy='simple').

        Returns:
            EntityCandidate: The typed entity candidate.
        """
        return EntityCandidate(
            text=entity.get("word", "").strip(),
            label=entity.get("entity_group", "UNKNOWN"),
            start=entity.get("start", 0),
            end=entity.get("end", 0),
            confidence=entity.get("score", 0.0),
            source_model=self.model_name,
        )

    async def extract(self, text: str, labels: list[str], threshold: float = 0.5) -> list[EntityCandidate]:
        """
        Extract entities from text.
        """
        self.validate_threshold(threshold)

        if not text:
            return []

        if self.model is None:
            await self.load_model()

        # Pipeline call
        raw_entities = await self.run_in_executor(self.model, text)

        candidates = []
        for entity in raw_entities:
            # Filter by confidence
            if entity.get("score", 0.0) < threshold:
                continue

            # Filter by label if labels are provided
            candidate = self._build_candidate(entity)
            if labels and candidate.label not in labels:
                continue

            candidates.append(candidate)

        return candidates

    async def extract_batch(
        self, texts: list[str], labels: list[str], threshold: float = 0.5
    ) -> list[list[EntityCandidate]]:
        """
        Extract entities from a batch of texts.
        """
        self.validate_threshold(threshold)

        if not texts:
            return []

        if self.model is None:
            await self.load_model()

        # Pipeline call with batch_size (defaulting to something reasonable or rely on auto)
        # transformers pipeline handles lists
        batch_raw_entities = await self.run_in_executor(self.model, texts)

        batch_extracted_candidates: list[list[EntityCandidate]] = []

        for raw_entities in batch_raw_entities:
            candidates = []
            for entity in raw_entities:
                if entity.get("score", 0.0) < threshold:
                    continue
                candidate = self._build_candidate(entity)
                if labels and candidate.label not in labels:
                    continue
                candidates.append(candidate)
            batch_extracted_candidates.append(candidates)

        return batch_extracted_candidates


class ExtractorFactory:
    """
    Factory for creating/retrieving extractor instances based on strategy.
    Implements the Strategy Pattern.
    """

    def __init__(self) -> None:
        self._cache: Dict[ExtractionStrategy, BaseExtractor] = {}

    def get_extractor(self, strategy: ExtractionStrategy) -> BaseExtractor:
        """
        Get the extractor instance for the given strategy.

        Args:
            strategy: The extraction strategy.

        Returns:
            BaseExtractor: The extractor instance.
        """
        if strategy in self._cache:
            return self._cache[strategy]

        extractor: BaseExtractor
        if strategy == ExtractionStrategy.SPEED_GLINER:
            extractor = GLiNERExtractor()
        elif strategy == ExtractionStrategy.PRECISION_NUNER:
            extractor = NuNERExtractor()
        else:
            # Fallback or Todo for REASONING_LLM
            extractor = GLiNERExtractor()  # Default

        self._cache[strategy] = extractor
        return extractor
