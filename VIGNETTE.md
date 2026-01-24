# The Architecture and Utility of coreason-tagger

### 1. The Philosophy (The Why)

**coreason-tagger** was born from a critical insight: clinical text is not merely a string of characters, but a dense web of assertions, negations, and temporal contexts. Standard Named Entity Recognition (NER) pipelines often fail because they treat extraction as a static task—either optimizing for speed at the cost of accuracy, or precision at the cost of latency.

The author designed this package to solve the specific pain point of **normalization-at-scale**. In biomedical domains, extracting "heart attack" is useless if the system cannot distinguish between "Patient has a heart attack" (Present), "Mother had a heart attack" (Family History), or "No signs of heart attack" (Absent). Furthermore, these extracted spans must be mapped to canonical ontologies (like SNOMED or RxNorm) to be computationally useful.

This architecture introduces a **Strategy Pattern** to the extraction layer. It acknowledges that not all text requires the same computational weight. A high-volume ETL job might demand the raw speed of a "Speed" strategy, while a complex diagnostic report requires the "Reasoning" capability of an LLM ensemble. By decoupling the *orchestration* from the *extraction logic*, `coreason-tagger` provides a flexible, latency-aware engine that turns unstructured text into structured Knowledge Graph nodes.

### 2. Under the Hood (The Dependencies & logic)

The codebase leverages a modern, async-native stack designed for high throughput and resilience:

*   **`asyncio`, `async-lru` & `litellm`**: The heart of the system is the `CoreasonTagger` orchestrator, which manages the "Extract-Contextualize-Link" loop. The architecture utilizes `asyncio.gather` for concurrency and `async-lru` for aggressive in-memory caching. The "Reasoning" strategy implements a sophisticated **"Cluster & Verify"** pipeline: it uses `gliner` for high-recall candidate generation, clusters overlapping spans, and then uses `litellm` to verify candidates with a Small Language Model (SLM), filtering out false positives.
*   **`gliner` & `transformers`**: These libraries power the swappable extraction backends. `gliner` provides the efficient zero-shot inference for the "Speed" strategy, while `transformers` enables the "Precision" strategy via the NuNER Zero model.
*   **`sentence-transformers`, `redis` & `CircuitBreaker`**: The `VectorLinker` moves beyond brittle string matching. It uses Bi-Encoders from `sentence-transformers` to embed candidates and retrieve semantically similar concepts. Crucially, it implements a **Two-Tier Caching** strategy (Memory L1 + Redis L2) and protects external dependencies with a custom **Circuit Breaker**. If the vector database or API fails, the system "fails open" into an Offline Mode, preserving the entity extraction even if linking is temporarily unavailable.
*   **`pydantic`**: Data integrity is non-negotiable. Pydantic v2 enforces strict schemas (`EntityCandidate`, `LinkedEntity`), ensuring that the complex flow of data between extractors, assertion detectors, and linkers remains valid and predictable.

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
    # In production, this connects to a Vector DB or Codex API
    # The Linker includes automatic Circuit Breaker protection
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

The system shines when processing multiple documents, allowing you to choose the "Reasoning" strategy for more complex tasks where ambiguity is high.

```python
async def batch_process(tagger: CoreasonTagger):
    texts = [
        "Mother had breast cancer.",
        "Rule out viral pneumonia."
    ]

    # Use the Reasoning strategy (Recall + LLM Verification) for ambiguous cases
    # This strategy generates candidates with GLiNER and verifies them with an LLM
    results = await tagger.tag_batch(
        texts,
        labels=["Disease"],
        strategy=ExtractionStrategy.REASONING_LLM
    )

    for i, text_entities in enumerate(results):
        print(f"--- Text {i+1} ---")
        for e in text_entities:
            # The Reasoning strategy filters false positives and handles complex contexts
            print(f"[{e.strategy_used}] Found: {e.text} ({e.assertion})")

```
