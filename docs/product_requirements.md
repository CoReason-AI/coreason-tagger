# **Architectural Specification: Coreason-Tagger**

Project Name: coreason-tagger
Target Audience: Advanced AI Coding Agent
Domain: Biomedical Semantic Extraction, Entity Linking, & Assertion Detection
Tech Stack: Python 3.10+, PyTorch, HuggingFace, Pydantic v2, Qdrant (Vector DB), Redis (Cache)

## ---

**1. Executive Summary**

**Coreason-Tagger** is a high-throughput, latency-aware NLP engine designed to normalize unstructured clinical and biomedical text into structured Knowledge Graph nodes.

Unlike static NER pipelines, this system utilizes a **Strategy Pattern Architecture** to dynamically switch extraction engines at runtime based on the complexity of the request. It solves three critical problems:

1. **Extraction:** Identifying entities using the optimal tradeoff between speed (GLiNER) and reasoning (LLM Ensembles).
2. **Contextualization:** Detecting negation, speculation, and history (Assertion Status) to prevent false positives.
3. **Normalization:** Mapping ambiguous text spans to canonical IDs (SNOMED, RxNorm) using Vector Retrieval.

## ---

**2. System Architecture**

The system follows a linear pipeline with hot-swappable components.

Data Flow:
Raw Text Batch $\rightarrow$ Orchestrator $\rightarrow$ [NER Strategy] $\rightarrow$ Assertion Filter $\rightarrow$ Vector Linker $\rightarrow$ Structured Output

### **2.1 Component Definitions**

* **Orchestrator (TaggerPipeline):** The main controller. It manages async concurrency, error handling, and telemetry.
* **Strategy Factory (ExtractorFactory):** A factory pattern that instantiates the correct NER backend (Speed, Precision, or Reasoning) based on the request config.
* **Contextualizer (AssertionEngine):** A lightweight module (Dependency Parser or DistilBERT) that assigns attributes like ABSENT or HISTORY.
* **Semantic Linker (EntityResolver):** A Bi-Encoder retrieval system that maps spans to Vector DB embeddings.

## ---

**3. Functional Requirements (The Strategies)**

The agent must implement three distinct extraction strategies, selectable via configuration.

### **Strategy A: The "Speed" Variant (GLiNER)**

* **Engine:** GLiNER (Base or Large).
* **Logic:** Single-pass inference.
* **Use Case:** High-volume ETL, real-time UI highlighting.
* **Constraint:** Must maximize GPU batch utilization.

### **Strategy B: The "Precision" Variant (NuNER Zero)**

* **Engine:** NuNER Zero.
* **Logic:** Token-classification approach.
* **Use Case:** Fixed schemas where boundary precision is critical (e.g., extracting "dosage" separate from "drug").
* **Requirement:** Must handle overlapping entities better than GLiNER.

### **Strategy C: The "Reasoning" Variant (Ensemble)**

* **Engine:** Candidate Generation + LLM Verification.
* **Logic:**
  1. **Recall:** Run GLiNER with a low threshold (0.15) to capture all potential candidates.
  2. **Verify:** Send text + candidates to a Small Language Model (SLM) like GPT-4o-mini or Llama-3-8B.
  3. **Prompt:** *"Given the context, is 'X' a valid [Label]? Reject false positives."*
* **Use Case:** Complex, ambiguous text (e.g., "History of breast cancer" vs. current diagnosis).

## ---

**4. Data Model Specifications (Pydantic v2)**

The coding agent must strictly adhere to these data structures to ensure type safety.

```python
from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field

# --- Enums ---
class ExtractionStrategy(str, Enum):
    SPEED_GLINER = "SPEED_GLINER"
    PRECISION_NUNER = "PRECISION_NUNER"
    REASONING_LLM = "REASONING_LLM"

class AssertionStatus(str, Enum):
    PRESENT = "PRESENT"          # Default
    ABSENT = "ABSENT"            # Negated ("No signs of...")
    POSSIBLE = "POSSIBLE"        # Speculative ("Rule out...")
    CONDITIONAL = "CONDITIONAL"  # ("If symptoms persist...")
    HISTORY = "HISTORY"          # ("History of...")
    FAMILY = "FAMILY"            # ("Mother had...")

# --- Core Objects ---
class EntityCandidate(BaseModel):
    """Raw output from the NER layer."""
    text: str
    start: int
    end: int
    label: str
    confidence: float
    source_model: str  # e.g., "gliner_large_v2"

class LinkedEntity(EntityCandidate):
    """The final hydrated entity."""
    # Context
    assertion: AssertionStatus = Field(default=AssertionStatus.PRESENT)

    # Linking (NEN)
    concept_id: Optional[str] = None    # "SNOMED:12345"
    concept_name: Optional[str] = None  # "Viral Rhinitis"
    link_score: float = 0.0             # Cosine similarity

    # Traceability
    strategy_used: ExtractionStrategy

class BatchRequest(BaseModel):
    texts: List[str]
    labels: List[str]
    config: dict = Field(default_factory=dict) # Overrides
```

## ---

**5. Interface Definitions (Abstract Base Classes)**

The agent must implement these interfaces to allow for modular upgrades.

### **5.1 The Extractor Interface**

```python
from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    """
    Contract for all NER backends.
    """

    @abstractmethod
    async def load_model(self):
        """Lazy loading of weights to VRAM."""
        pass

    @abstractmethod
    async def extract_batch(self, texts: List[str], labels: List[str]) -> List[List[EntityCandidate]]:
        """
        Main entry point. Must handle batching logic.
        """
        pass
```

### **5.2 The Linker Interface**

```python
class BaseLinker(ABC):
    """
    Contract for Entity Normalization.
    """

    @abstractmethod
    async def resolve(self, entity: EntityCandidate, context: str) -> LinkedEntity:
        """
        1. Check Cache.
        2. Vector Search (BM25 + Dense).
        3. Threshold Check.
        """
        pass
```

## ---

**6. Algorithmic Directives**

### **6.1 The "Reasoning" Ensemble Logic**

The agent must implement the ReasoningExtractor class with the following specific logic:

1. **Candidate Generation:** Call GLiNER.predict_entities(threshold=0.15).
2. **Clustering:** If multiple spans overlap by >50%, group them.
3. **Verification Payload Construction:**
   * Construct a prompt: Check the following entities in this text: "{text}". Entities: {json_list}. Return only valid ones.
4. **Parsing:** Parse the JSON response from the LLM. If parsing fails, fallback to the raw GLiNER candidates (Fail-Open).

### **6.2 Vector Linking Logic**

1. **Embedding:** Use a domain-specific Bi-Encoder (e.g., BioLORD-2023).
2. **Hybrid Search:**
   * Query Qdrant/Milvus with vector AND keyword (BM25) to ensure "Tylenol" matches "Acetaminophen".
3. **Caching:** Implement a **Two-Tier Cache**:
   * L1: In-Memory LRU (for high-frequency terms like "Diabetes").
   * L2: Redis (persistent cache).

## ---

**7. Operational Requirements**

1. **Singleton Pattern:**
   * Heavy models (GLiNER, NuNER) must be wrapped in a Singleton. They should **never** be re-initialized per request.
   * Implement a ModelRegistry class to manage VRAM.
2. **Async/Await:**
   * The entire pipeline must be async.
   * Blocking GPU operations must run in a thread/process executor to avoid freezing the API event loop.
3. **Telemetry:**
   * Every step (extraction, assertion, linking) must emit a timing log.
   * Log the strategy_used for every request to analyze cost/performance later.
4. **Error Handling:**
   * **Circuit Breaker:** If the Vector DB fails 5 times in 10 seconds, switch to "Offline Mode" (skip linking, return raw entities).
   * **LLM Timeout:** If the Verification LLM takes >2s, return the raw candidates immediately.
