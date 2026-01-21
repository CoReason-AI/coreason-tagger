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
from typing import Any, Optional

from gliner import GLiNER

from coreason_tagger.config import settings
from coreason_tagger.interfaces import BaseExtractor
from coreason_tagger.registry import get_gliner_model, get_nuner_pipeline
from coreason_tagger.schema import EntityCandidate


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

    def _validate_threshold(self, threshold: float) -> None:
        """
        Validate that the threshold is within the valid range [0.0, 1.0].

        Args:
            threshold (float): The threshold value to check.

        Raises:
            ValueError: If the threshold is out of bounds.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

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
        self._validate_threshold(threshold)

        if not text or not labels:
            return []

        if self.model is None:
            await self.load_model()

        # GLiNER returns a list of dicts:
        # [{'start': 0, 'end': 5, 'text': '...', 'label': '...', 'score': 0.95}, ...]
        loop = asyncio.get_running_loop()
        # predict_entities is blocking, run in executor
        raw_entities = await loop.run_in_executor(
            None,
            lambda: self.model.predict_entities(text, labels, threshold=threshold),  # type: ignore
        )

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
        self._validate_threshold(threshold)

        if not texts or not labels:
            return [[] for _ in texts]

        if self.model is None:
            await self.load_model()

        # Use batch_predict_entities if available.
        # batch_predict_entities returns a list of lists of dicts.
        loop = asyncio.get_running_loop()
        # batch_predict_entities is blocking, run in executor
        batch_raw_entities = await loop.run_in_executor(
            None,
            lambda: self.model.batch_predict_entities(texts, labels, threshold=threshold),  # type: ignore
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

    def _validate_threshold(self, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

    async def extract(self, text: str, labels: list[str], threshold: float = 0.5) -> list[EntityCandidate]:
        """
        Extract entities from text.
        Note: NuNER Zero is a token classifier.
        The labels passed here might not be used directly if the model is fine-tuned on fixed classes.
        However, NuNER Zero is often used for specific schemas.
        The current NuNER Zero model behaves as a standard token classifier or GLiNER-like?
        NuNER Zero is actually a fine-tuned GLiNER model or similar in some contexts,
        but here we are using it via `transformers` pipeline.
        If it's a standard BERT-like token classifier, it ignores `labels` argument during inference
        (it predicts what it was trained on).
        However, if we are using it as a "Precision" variant as per requirements, we assume it extracts entities.
        We will filter the output by `labels` if provided and if they match the model's output labels.
        """
        self._validate_threshold(threshold)

        if not text:
            return []

        if self.model is None:
            await self.load_model()

        loop = asyncio.get_running_loop()
        # Pipeline call
        raw_entities = await loop.run_in_executor(
            None,
            lambda: self.model(text),
        )

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
        self._validate_threshold(threshold)

        if not texts:
            return []

        if self.model is None:
            await self.load_model()

        loop = asyncio.get_running_loop()
        # Pipeline call with batch_size (defaulting to something reasonable or rely on auto)
        # transformers pipeline handles lists
        batch_raw_entities = await loop.run_in_executor(
            None,
            lambda: self.model(texts),
        )

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
