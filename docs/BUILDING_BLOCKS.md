# Building Blocks Design Documentation
## Tri-Lingual Turing Agent Pipeline

**Document Version:** 1.0
**Date:** November 29, 2025
**Author:** Tal Barda
**Compliance:** Software Submission Guidelines v2.0, Section 17

---

## Table of Contents

1. [Overview](#overview)
2. [Building Block Principles](#building-block-principles)
3. [Core Building Blocks](#core-building-blocks)
4. [Building Block Catalog](#building-block-catalog)
5. [Composition Patterns](#composition-patterns)
6. [Design Rationale](#design-rationale)
7. [Usage Examples](#usage-examples)

---

## Overview

This document describes the modular "building blocks" architecture of the Tri-Lingual Turing Agent Pipeline. Each building block is a self-contained component with clearly defined:

- **Input Data:** What the block receives
- **Output Data:** What the block produces
- **Setup Data:** Configuration parameters

This design enables:
- **Reusability:** Blocks can be used in different contexts
- **Testability:** Each block can be tested in isolation
- **Composability:** Blocks combine to create complex workflows
- **Maintainability:** Changes localized to specific blocks

---

## Building Block Principles

### Core Characteristics

Every building block in this system follows these principles:

#### 1. Single Responsibility
Each block does ONE thing well.

```python
# Good: Single responsibility
class SpellingErrorInjector:
    """Injects spelling errors into text."""
    def inject(self, text, error_rate):
        # Only handles error injection
        pass

# Bad: Multiple responsibilities
class TextProcessorAndTranslator:
    """Processes text AND translates it."""  # Violates SRP
    def inject_errors(self, text): pass
    def translate(self, text): pass
    def calculate_distance(self, text1, text2): pass
```

#### 2. Clear Interface Contract

**Input Data, Output Data, Setup Data** explicitly defined.

```python
class EmbeddingModel:
    """
    Building Block: Embedding Generator

    Setup Data:
        model_name: str - Sentence-transformer model identifier
        device: str - 'cpu' or 'cuda'

    Input Data:
        text: str or List[str] - Text to encode

    Output Data:
        embedding: np.ndarray - Dense vector representation
    """
```

#### 3. No Hidden Dependencies

All dependencies injected or explicitly configured.

```python
# Good: Dependencies explicit
class TranslationAgent:
    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key

# Bad: Hidden dependency
class TranslationAgent:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")  # Hidden!
```

#### 4. Composable

Blocks easily combine to create pipelines.

```python
# Composing blocks
error_injector = SpellingErrorInjector(error_rate=0.25)
agent = TranslationAgent(source="en", target="fr")
embedder = EmbeddingModel(model="all-MiniLM-L6-v2")

# Pipeline
corrupted_text = error_injector.inject(original_text)
french_text = agent.translate(corrupted_text)
embedding = embedder.encode(french_text)
```

---

## Core Building Blocks

### Block Hierarchy

```
Core Building Blocks
├── Input Processing Blocks
│   ├── ErrorInjectionBlock
│   └── TextValidationBlock
├── Agent Execution Blocks
│   ├── TranslationAgentBlock
│   └── ParallelAgentOrchestratorBlock
├── Embedding Blocks
│   ├── EmbeddingModelBlock
│   └── ParallelEmbeddingProcessorBlock
├── Distance Calculation Blocks
│   ├── DistanceCalculatorBlock
│   └── SimilarityComparatorBlock
├── Orchestration Blocks
│   ├── ExperimentRunnerBlock
│   └── ErrorRateSweepBlock
└── Output Blocks
    ├── VisualizationBlock
    └── ResultsStorageBlock
```

---

## Building Block Catalog

### 1. Error Injection Block

**Purpose:** Inject controlled spelling errors into text

**Module:** `tri_lingual_agents.error_injection.injector`
**Class:** `SpellingErrorInjector`

#### Interface Definition

```python
class SpellingErrorInjector:
    """
    Building Block: Spelling Error Injection

    Introduces realistic spelling errors into English text at a
    specified rate for robustness testing.
    """

    # ===== SETUP DATA =====
    def __init__(
        self,
        error_rate: float = 0.0,
        seed: Optional[int] = None,
        min_word_length: int = 4
    ):
        """
        Setup Data:
            error_rate: Fraction of words to corrupt (0.0 to 1.0)
            seed: Random seed for reproducibility
            min_word_length: Minimum word length to consider for corruption
        """
        pass

    # ===== MAIN FUNCTION =====
    def inject_spelling_errors(
        self,
        text: str  # INPUT DATA
    ) -> Tuple[str, List[str]]:  # OUTPUT DATA
        """
        Input Data:
            text: str - Clean English text to corrupt

        Output Data:
            corrupted_text: str - Text with spelling errors
            corrupted_words: List[str] - List of word pairs
                                          "original → corrupted"
        """
        pass
```

#### Internal Operations

1. **Tokenization:** Split text into words
2. **Selection:** Randomly select words based on error rate
3. **Corruption:** Apply one of four strategies:
   - Substitution (keyboard-proximity)
   - Transposition (adjacent character swap)
   - Omission (character deletion)
   - Duplication (character repetition)
4. **Reconstruction:** Reassemble text preserving formatting

#### Dependencies

- **External:** `random` (Python standard library)
- **Internal:** None

#### Example Usage

```python
# Setup
injector = SpellingErrorInjector(
    error_rate=0.25,
    seed=42
)

# Process
original = "The remarkable transformation of systems"
corrupted, changes = injector.inject_spelling_errors(original)

# Output
print(corrupted)
# "The remarkabel trqnsformation of syatems"
print(changes)
# ['remarkable → remarkabel', 'transformation → trqnsformation', 'systems → syatems']
```

---

### 2. Translation Agent Block

**Purpose:** Translate text between language pairs

**Module:** `tri_lingual_agents.agents.translators`
**Class:** `TranslationAgent`

#### Interface Definition

```python
class TranslationAgent:
    """
    Building Block: Language Translation Agent

    Wrapper for LLM translation APIs with retry logic and
    error handling.
    """

    # ===== SETUP DATA =====
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        provider: str = "anthropic",
        model: Optional[str] = None,
        timeout: int = 300
    ):
        """
        Setup Data:
            source_lang: ISO 639-1 code (e.g., "en", "fr", "he")
            target_lang: ISO 639-1 code
            provider: "anthropic" or "openai" or "mock"
            model: Specific model name (optional)
            timeout: Maximum seconds per request
        """
        pass

    # ===== MAIN FUNCTION =====
    def translate(
        self,
        text: str,  # INPUT DATA
        handle_errors: bool = False
    ) -> str:  # OUTPUT DATA
        """
        Input Data:
            text: str - Text in source language
            handle_errors: bool - Whether to prompt LLM about errors

        Output Data:
            translated_text: str - Text in target language
        """
        pass
```

#### Internal Operations

1. **Prompt Construction:** Build translation request
2. **API Invocation:** Call LLM provider
3. **Retry Logic:** Exponential backoff (3 attempts)
4. **Response Extraction:** Parse translation from response
5. **Error Handling:** Timeout protection, fallback options

#### Dependencies

- **External:** `anthropic` or `openai` SDK
- **Internal:** None

#### Example Usage

```python
# Setup
agent = TranslationAgent(
    source_lang="en",
    target_lang="fr",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

# Process
english = "The remarkable transformation"
french = agent.translate(english, handle_errors=True)

# Output
print(french)
# "La transformation remarquable"
```

---

### 3. Embedding Model Block

**Purpose:** Generate semantic embeddings for text

**Module:** `tri_lingual_agents.embeddings.distance`
**Class:** `EmbeddingModel`

#### Interface Definition

```python
class EmbeddingModel:
    """
    Building Block: Sentence Embedding Generator

    Converts text into dense vector representations for
    semantic similarity measurement.
    """

    # ===== SETUP DATA =====
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Setup Data:
            model_name: Sentence-transformer model identifier
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        pass

    # ===== MAIN FUNCTION =====
    def encode(
        self,
        text: Union[str, List[str]]  # INPUT DATA
    ) -> np.ndarray:  # OUTPUT DATA
        """
        Input Data:
            text: str or List[str] - Text to encode

        Output Data:
            embedding: np.ndarray - Shape (embedding_dim,) or
                                     (num_texts, embedding_dim)
        """
        pass
```

#### Internal Operations

1. **Model Loading:** Load pre-trained sentence-transformer
2. **Tokenization:** Convert text to token IDs
3. **Encoding:** Pass through transformer network
4. **Pooling:** Mean pooling over token representations
5. **Normalization:** L2 normalization for cosine similarity

#### Dependencies

- **External:** `sentence-transformers`, `torch`
- **Internal:** None

#### Example Usage

```python
# Setup
model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

# Process single text
text = "The remarkable transformation"
embedding = model.encode(text)

# Output
print(embedding.shape)
# (384,)

# Process batch
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = model.encode(texts)
print(embeddings.shape)
# (3, 384)
```

---

### 4. Distance Calculator Block

**Purpose:** Compute semantic distance between embeddings

**Module:** `tri_lingual_agents.embeddings.distance`
**Functions:** `calculate_distance`, `calculate_similarity`

#### Interface Definition

```python
# ===== SETUP DATA: None (stateless function) =====

def calculate_distance(
    embedding1: np.ndarray,  # INPUT DATA
    embedding2: np.ndarray,  # INPUT DATA
    metric: str = "cosine"
) -> float:  # OUTPUT DATA
    """
    Building Block: Semantic Distance Calculator

    Input Data:
        embedding1: np.ndarray - First embedding vector
        embedding2: np.ndarray - Second embedding vector
        metric: str - "cosine" or "euclidean"

    Output Data:
        distance: float - Semantic distance (lower = more similar)
                         Range: [0, 2] for cosine, [0, ∞] for euclidean
    """
    pass
```

#### Internal Operations

**Cosine Distance:**
```python
cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
distance = 1 - cosine_sim
```

**Euclidean Distance:**
```python
distance = np.linalg.norm(emb1 - emb2)
```

#### Dependencies

- **External:** `numpy`
- **Internal:** None

#### Example Usage

```python
# Setup (get embeddings first)
model = EmbeddingModel()
emb1 = model.encode("The remarkable transformation")
emb2 = model.encode("La transformation remarquable")

# Process
distance = calculate_distance(emb1, emb2, metric="cosine")

# Output
print(f"Distance: {distance:.6f}")
# Distance: 0.030116

# Interpretation
if distance < 0.1:
    print("Very similar")
elif distance < 0.3:
    print("Similar")
else:
    print("Different")
# "Very similar"
```

---

### 5. Experiment Runner Block

**Purpose:** Execute single translation experiment

**Module:** `tri_lingual_agents.pipeline.orchestrator`
**Function:** `run_experiment`

#### Interface Definition

```python
def run_experiment(
    # ===== INPUT DATA =====
    sentence: str,
    error_rate: float,
    agents: Tuple[TranslationAgent, TranslationAgent, TranslationAgent],
    embedding_model: EmbeddingModel,

    # ===== SETUP DATA =====
    seed: Optional[int] = None,
    save_results: bool = True,
    output_dir: str = "results"
) -> Dict[str, Any]:  # OUTPUT DATA
    """
    Building Block: Single Experiment Orchestrator

    Coordinates error injection, translation pipeline, and
    semantic distance measurement.

    Input Data:
        sentence: str - Original English text
        error_rate: float - Error rate (0.0 to 1.0)
        agents: Tuple[3] - Three translation agents (EN→FR, FR→HE, HE→EN)
        embedding_model: EmbeddingModel - Embedding generator

    Setup Data:
        seed: int - Random seed
        save_results: bool - Whether to persist results
        output_dir: str - Directory for outputs

    Output Data:
        result: Dict containing:
            - error_rate: float
            - original_sentence: str
            - corrupted_sentence: str
            - french_translation: str
            - hebrew_translation: str
            - final_english: str
            - cosine_distance: float
            - timestamp: str
    """
    pass
```

#### Internal Operations

1. **Error Injection:** Call `SpellingErrorInjector`
2. **Translation Chain:**
   - Agent 1: EN → FR (with error handling)
   - Agent 2: FR → HE
   - Agent 3: HE → EN
3. **Embedding Generation:**
   - Encode original English
   - Encode final English
4. **Distance Calculation:** Compute cosine distance
5. **Result Compilation:** Assemble output dictionary
6. **Persistence:** Save to JSON (if enabled)

#### Example Usage

```python
# Setup
from tri_lingual_agents import create_agent_pipeline, get_embedding_model

agents = create_agent_pipeline(provider="anthropic")
embedder = get_embedding_model()

# Process
result = run_experiment(
    sentence="The remarkable transformation of artificial intelligence systems",
    error_rate=0.25,
    agents=agents,
    embedding_model=embedder,
    seed=42
)

# Output
print(f"Distance: {result['cosine_distance']}")
# Distance: 0.043375
print(f"Final: {result['final_english']}")
# Final: "The impressive transformation of AI systems..."
```

---

### 6. Error Rate Sweep Block

**Purpose:** Execute experiments across multiple error rates

**Module:** `tri_lingual_agents.pipeline.orchestrator`
**Function:** `run_error_rate_sweep`

#### Interface Definition

```python
def run_error_rate_sweep(
    # ===== INPUT DATA =====
    sentence: str,
    error_rates: List[float],

    # ===== SETUP DATA =====
    provider: str = "anthropic",
    model: Optional[str] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    output_dir: str = "results",
    save_results: bool = True,
    generate_plots: bool = True
) -> Dict[str, Any]:  # OUTPUT DATA
    """
    Building Block: Multi-Experiment Batch Orchestrator

    Runs multiple experiments across error rate range.

    Input Data:
        sentence: str - Base sentence
        error_rates: List[float] - Error rates to test

    Setup Data:
        provider: str - LLM provider
        model: str - Model name
        embedding_model_name: str - Embedding model
        output_dir: str - Output directory
        save_results: bool - Persist results
        generate_plots: bool - Create visualizations

    Output Data:
        results: Dict containing:
            - experiment_metadata: {...}
            - results: List[experiment_result]
            - plots_generated: List[str]
    """
    pass
```

#### Internal Operations

1. **Initialization:** Create agents and embedding model
2. **Loop Execution:**
   - For each error rate:
     - Call `run_experiment`
     - Collect result
3. **Aggregation:** Compile all results
4. **Visualization:** Generate graphs (if enabled)
5. **Persistence:** Save JSON and plots

#### Example Usage

```python
# Setup and process in one call
results = run_error_rate_sweep(
    sentence="The remarkable transformation...",
    error_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    provider="anthropic",
    generate_plots=True
)

# Output
print(f"Completed {len(results['results'])} experiments")
# Completed 6 experiments
print(f"Plots: {results['plots_generated']}")
# Plots: ['results/error_rate_vs_distance.png', 'results/experiment_summary.png']
```

---

### 7. Visualization Block

**Purpose:** Generate graphs from experimental results

**Module:** `tri_lingual_agents.visualization.plots`
**Function:** `plot_error_vs_distance`

#### Interface Definition

```python
def plot_error_vs_distance(
    # ===== INPUT DATA =====
    error_rates: List[float],
    distances: List[float],

    # ===== SETUP DATA =====
    output_path: str = "results/error_rate_vs_distance.png",
    title: str = "Semantic Drift vs. Error Rate",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300
) -> str:  # OUTPUT DATA
    """
    Building Block: Visualization Generator

    Creates publication-quality line plot.

    Input Data:
        error_rates: List[float] - X-axis values
        distances: List[float] - Y-axis values

    Setup Data:
        output_path: str - File path for PNG
        title: str - Graph title
        figsize: Tuple - Figure dimensions
        dpi: int - Resolution

    Output Data:
        saved_path: str - Path to saved PNG file
    """
    pass
```

#### Example Usage

```python
# Setup data
error_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
distances = [0.030, 0.030, 0.030, 0.043, 0.037, 0.047]

# Process
plot_path = plot_error_vs_distance(
    error_rates=error_rates,
    distances=distances,
    output_path="results/my_plot.png",
    dpi=300
)

# Output
print(f"Saved to: {plot_path}")
# Saved to: results/my_plot.png
```

---

## Composition Patterns

### Pattern 1: Sequential Pipeline

```python
# Compose blocks sequentially
def full_translation_pipeline(sentence: str, error_rate: float):
    # Block 1: Error injection
    injector = SpellingErrorInjector(error_rate=error_rate)
    corrupted, _ = injector.inject_spelling_errors(sentence)

    # Block 2-4: Translation chain
    agent_en_fr = TranslationAgent("en", "fr")
    agent_fr_he = TranslationAgent("fr", "he")
    agent_he_en = TranslationAgent("he", "en")

    french = agent_en_fr.translate(corrupted)
    hebrew = agent_fr_he.translate(french)
    final = agent_he_en.translate(hebrew)

    # Block 5: Embedding
    embedder = EmbeddingModel()
    original_emb = embedder.encode(sentence)
    final_emb = embedder.encode(final)

    # Block 6: Distance
    distance = calculate_distance(original_emb, final_emb)

    return {
        "original": sentence,
        "final": final,
        "distance": distance
    }
```

### Pattern 2: Parallel Composition

```python
# Compose blocks in parallel
def parallel_experiment_batch(sentences: List[str], error_rate: float):
    # Parallel Block: Embeddings
    embedder_parallel = ParallelEmbeddingProcessor(num_processes=4)
    embeddings = embedder_parallel.encode_batch(sentences)

    # Parallel Block: Agents
    orchestrator = ParallelAgentOrchestrator(max_concurrent=3)
    results = orchestrator.run_experiments_parallel(sentences)

    return results
```

### Pattern 3: Nested Composition

```python
# Higher-order block using sub-blocks
class FullPipelineBlock:
    """Composite block combining multiple sub-blocks."""

    def __init__(self):
        # Sub-blocks
        self.error_injector = SpellingErrorInjector()
        self.agents = create_agent_pipeline()
        self.embedder = EmbeddingModel()

    def run(self, sentence: str, error_rate: float):
        # Orchestrate sub-blocks
        result = run_experiment(
            sentence=sentence,
            error_rate=error_rate,
            agents=self.agents,
            embedding_model=self.embedder
        )
        return result
```

---

## Design Rationale

### Why Building Blocks?

#### 1. Testability

Each block tested independently:

```python
def test_error_injector():
    """Test error injection block in isolation."""
    injector = SpellingErrorInjector(error_rate=0.5, seed=42)
    text = "Hello world"
    corrupted, _ = injector.inject_spelling_errors(text)

    assert corrupted != text  # Corruption occurred
    assert len(corrupted.split()) == len(text.split())  # Word count preserved
```

#### 2. Reusability

Blocks used in multiple contexts:

```python
# Context 1: Research experiment
embedder = EmbeddingModel()
result = run_experiment(..., embedding_model=embedder)

# Context 2: Production similarity search
embedder = EmbeddingModel()  # Same block!
query_emb = embedder.encode(user_query)
doc_embs = embedder.encode(documents)
```

#### 3. Maintainability

Changes localized:

```
If embedding model changes:
  → Only modify EmbeddingModel block
  → No changes to TranslationAgent block
  → No changes to ErrorInjector block
```

#### 4. Clarity

Interface makes data flow explicit:

```python
# Clear: What goes in, what comes out
def translate(text: str) -> str:
    """Input: text | Output: translated_text"""
    pass

# Unclear: Hidden state, unclear outputs
def process(self):
    """Does something with self.data..."""
    pass
```

---

## Compliance Checklist

**Software Submission Guidelines v2.0, Section 17 Requirements:**

- [x] Each major component documented as building block
- [x] Input Data, Output Data, Setup Data clearly specified
- [x] Building blocks demonstrated to be reusable
- [x] Composition patterns documented
- [x] Examples showing block usage
- [x] Design rationale explained
- [x] Interface contracts explicit

---

**Document Version:** 1.0
**Last Updated:** November 29, 2025
**Status:** Complete
**Compliance:** Software Submission Guidelines v2.0, Section 17 ✓
