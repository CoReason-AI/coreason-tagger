# The Architecture and Utility of coreason-tagger

### 1. The Philosophy (The Why)

**coreason-tagger** was born from a critical insight: clinical text is not merely a string of characters, but a dense web of assertions, negations, and temporal contexts. Standard Named Entity Recognition (NER) pipelines often fail because they treat extraction as a static task—either optimizing for speed at the cost of accuracy, or precision at the cost of latency.

The author designed this package to solve the specific pain point of **normalization-at-scale**. In biomedical domains, extracting "heart attack" is useless if the system cannot distinguish between "Patient has a heart attack" (Present), "Mother had a heart attack" (Family History), or "No signs of heart attack" (Absent). Furthermore, these extracted spans must be mapped to canonical ontologies (like SNOMED or RxNorm) to be computationally useful.

This architecture introduces a **Strategy Pattern** to the extraction layer. It acknowledges that not all text requires the same computational weight. A high-volume ETL job might demand the raw speed of a "Speed" strategy, while a complex diagnostic report requires the "Reasoning" capability of an LLM ensemble. By decoupling the *orchestration* from the *extraction logic*, `coreason-tagger` provides a flexible, latency-aware engine that turns unstructured text into structured Knowledge Graph nodes.

### 2. Under the Hood (The Dependencies & logic)

The codebase leverages a modern, async-native stack designed for high throughput and type safety:

*   **`asyncio` & `async-lru`**: The heart of the system is the `CoreasonTagger` orchestrator, which manages the "Extract-Contextualize-Link" loop. By building on `asyncio` and leveraging `async-lru` for aggressive caching, the system maximizes I/O concurrency—critical when fetching vector embeddings or querying external models.
*   **`gliner` & `transformers`**: These libraries power the swappable extraction backends. `gliner` provides the "Speed" strategy via efficient zero-shot inference, while `transformers` enables the "Precision" strategy (NuNER Zero) and the assertion detection logic (DistilBERT).
*   **`sentence-transformers` & `redis`**: The `VectorLinker` moves beyond brittle string matching. It uses Bi-Encoders from `sentence-transformers` to embed candidates and retrieve semantically similar concepts, backed by `redis` for persistent caching of expensive vector lookups.
*   **`pydantic`**: Data integrity is non-negotiable. Pydantic v2 enforces strict schemas (`EntityCandidate`, `LinkedEntity`), ensuring that the complex flow of data between extractors, assertion detectors, and linkers remains valid and predictable.

The internal logic follows a robust **Pipeline Architecture**. A request enters the `CoreasonTagger`, which delegates to an `ExtractorFactory` to instantiate the appropriate model (Singleton-wrapped to manage VRAM). Extracted candidates are then passed through an `AssertionEngine` for context (assigning attributes like `ABSENT` or `HISTORY`) before finally being resolved by the `VectorLinker`.

### 3. In Practice (The How)

The power of `coreason-tagger` lies in its ability to configure complex NLP pipelines with a clean, Pythonic API.

**Example 1: The Happy Path (Speed Strategy)**

Here, we initialize the tagger with standard components and process a clinical sentence. The system automatically handles the extraction, assertion detection ("denies"), and linking.

```python
import asyncio
from coreason_tagger.tagger import CoreasonTagger
from coreason_tagger.ner import ExtractorFactory
from coreason_tagger.assertion_detector import RegexBasedAssertionDetector
from coreason_tagger.linker import VectorLinker
from coreason_tagger.schema import ExtractionStrategy

async def main():
    # 1. Assemble the Pipeline Components
    # The Factory manages model loading (Singleton pattern)
    ner_factory = ExtractorFactory()

    # Lightweight rule-based assertion for speed
    assertion = RegexBasedAssertionDetector()

    # Linker requires a client (mocked here for demonstration)
    # In production, this connects to a Vector DB
    linker = VectorLinker(codex_client=...)

    # 2. Initialize the Orchestrator
    tagger = CoreasonTagger(ner=ner_factory, assertion=assertion, linker=linker)

    # 3. Process Text
    text = "Patient denies diabetes but reports a history of hypertension."
    entities = await tagger.tag(
        text,
        labels=["Disease"],
        strategy=ExtractionStrategy.SPEED_GLINER
    )

    for e in entities:
        print(f"Entity: {e.text} | Status: {e.assertion} | ID: {e.concept_id}")
        # Output:
        # Entity: diabetes     | Status: ABSENT  | ID: SNOMED:73211009
        # Entity: hypertension | Status: HISTORY | ID: SNOMED:38341003

if __name__ == "__main__":
    asyncio.run(main())
```

**Example 2: Batch Processing with Strategy Switching**

The system shines when processing multiple documents, allowing you to choose the "Reasoning" strategy for more complex tasks.

```python
async def batch_process():
    tagger = ... # (Initialized as above)

    texts = [
        "Mother had breast cancer.",
        "Rule out viral pneumonia."
    ]

    # Use the Reasoning strategy (LLM-verified) for ambiguous cases
    results = await tagger.tag_batch(
        texts,
        labels=["Disease"],
        strategy=ExtractionStrategy.REASONING_LLM
    )

    for text_entities in results:
        for e in text_entities:
            # The Reasoning strategy filters false positives and handles complex contexts
            print(f"[{e.strategy_used}] Found: {e.text} ({e.assertion})")

```
