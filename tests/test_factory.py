# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from unittest.mock import patch

from coreason_tagger.factory import get_tagger
from coreason_tagger.tagger import CoreasonTagger


def test_get_tagger_factory() -> None:
    """
    Test the factory function creates a valid CoreasonTagger instance.
    We mock the heavy dependencies to make this unit test fast.
    """
    with patch("coreason_tagger.factory.GLiNERExtractor"), patch("coreason_tagger.factory.VectorLinker"):
        tagger = get_tagger()
        assert isinstance(tagger, CoreasonTagger)
        assert tagger.ner is not None
        assert tagger.assertion is not None
        assert tagger.linker is not None
