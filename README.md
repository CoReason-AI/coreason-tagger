# coreason-tagger

**A High-Throughput Biomedical Semantic Extraction Engine**

[![License](https://img.shields.io/badge/License-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![CI](https://github.com/CoReason-AI/coreason_tagger/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_tagger/actions/workflows/ci.yml)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Coreason-Tagger** is a high-throughput, latency-aware NLP engine designed to normalize unstructured clinical and biomedical text into structured Knowledge Graph nodes. It utilizes a Strategy Pattern Architecture to dynamically switch extraction engines at runtime based on the complexity of the request.

## Features

*   **Dynamic Extraction Strategies:**
    *   **Speed (GLiNER):** Single-pass inference for high-volume ETL and real-time UI highlighting.
    *   **Precision (NuNER Zero):** Token-classification for fixed schemas where boundary precision is critical.
    *   **Reasoning (Ensemble):** LLM-verified candidates for complex, ambiguous text.
*   **Contextualization:** Lightweight assertion detection (Negation, Speculation, History) to prevent false positives.
*   **Normalization:** Maps ambiguous text spans to canonical IDs (e.g., SNOMED, RxNorm) using Vector Retrieval and Semantic Re-ranking.
*   **Resilience:** Built-in Circuit Breakers and fallback mechanisms for robust operation.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Here is a quick example of how to initialize and use the `CoreasonTagger`:

```python
import asyncio
from coreason_tagger.tagger import CoreasonTagger
from coreason_tagger.ner import ExtractorFactory
from coreason_tagger.assertion_detector import RegexBasedAssertionDetector
from coreason_tagger.codex_real import RealCoreasonCodex
from coreason_tagger.linker import VectorLinker
from coreason_tagger.schema import ExtractionStrategy

async def main():
    # Initialize components
    ner = ExtractorFactory()
    assertion = RegexBasedAssertionDetector()
    codex_client = RealCoreasonCodex(api_url="http://localhost:8000")
    linker = VectorLinker(codex_client=codex_client)

    # Initialize Tagger
    tagger = CoreasonTagger(ner=ner, assertion=assertion, linker=linker)

    # Tag text
    text = "Patient complains of severe migraine and nausea."
    results = await tagger.tag(
        text,
        labels=["Symptom", "Condition"],
        strategy=ExtractionStrategy.SPEED_GLINER
    )

    for entity in results:
        print(f"{entity.text}: {entity.label} (Assertion: {entity.assertion})")

if __name__ == "__main__":
    asyncio.run(main())
```
