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

from gliner import GLiNER

from coreason_tagger.interfaces import BaseNERExtractor
from coreason_tagger.schema import ExtractedSpan


class GLiNERExtractor(BaseNERExtractor):
    """
    Zero-Shot NER Extractor using the GLiNER library.
    Wraps the underlying model to provide a clean interface for extracting entities.
    """

    def __init__(self, model_name: str = "urchade/gliner_small-v2.1") -> None:
        """
        Initialize the GLiNER extractor.

        Args:
            model_name (str): The name of the GLiNER model to load.
                             Defaults to "urchade/gliner_small-v2.1" (lightweight).
        """
        self.model_name = model_name
        # Load the model. Note: This might download weights on first run.
        # In a real production setup, we might want to lazy-load or use a singleton pattern,
        # but for this atomic unit, strict encapsulation is preferred.
        self.model = GLiNER.from_pretrained(model_name)

    def extract(self, text: str, labels: List[str]) -> List[ExtractedSpan]:
        """
        Extract entities from text using the provided labels.

        Args:
            text (str): The input text to process.
            labels (List[str]): A list of entity types to detect.

        Returns:
            List[ExtractedSpan]: A list of detected entity spans.
        """
        if not text or not labels:
            return []

        # GLiNER returns a list of dicts:
        # [{'start': 0, 'end': 5, 'text': '...', 'label': '...', 'score': 0.95}, ...]
        raw_entities = self.model.predict_entities(text, labels)

        extracted_spans: List[ExtractedSpan] = []
        for entity in raw_entities:
            span = ExtractedSpan(
                text=entity["text"],
                label=entity["label"],
                start=entity["start"],
                end=entity["end"],
                score=entity["score"],
                context=text,
            )
            extracted_spans.append(span)

        return extracted_spans

    def extract_batch(self, texts: List[str], labels: List[str]) -> List[List[ExtractedSpan]]:
        """
        Extract entities from a batch of texts using the provided labels.

        Args:
            texts (List[str]): The list of input texts to process.
            labels (List[str]): A list of entity types to detect.

        Returns:
            List[List[ExtractedSpan]]: A list of lists, where each inner list contains
                                       detected entity spans for the corresponding text.
        """
        if not texts or not labels:
            return [[] for _ in texts]

        # Use batch_predict_entities if available, otherwise fallback to loop?
        # We verified batch_predict_entities exists on the model instance.
        # batch_predict_entities returns a list of lists of dicts.
        batch_raw_entities = self.model.batch_predict_entities(texts, labels)

        batch_extracted_spans: List[List[ExtractedSpan]] = []

        # Iterate over both the original texts (for context) and the results
        for text, raw_entities in zip(texts, batch_raw_entities, strict=False):
            extracted_spans: List[ExtractedSpan] = []
            for entity in raw_entities:
                span = ExtractedSpan(
                    text=entity["text"],
                    label=entity["label"],
                    start=entity["start"],
                    end=entity["end"],
                    score=entity["score"],
                    context=text,
                )
                extracted_spans.append(span)
            batch_extracted_spans.append(extracted_spans)

        return batch_extracted_spans
