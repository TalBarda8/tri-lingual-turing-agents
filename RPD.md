# RPD Document: Tri-Lingual Turing Agent Pipeline with Spelling Error Robustness Analysis

**Project:** Multi-Agent Translation Loop with Embedding Distance Evaluation
**Course:** AI Agent Systems
**Date:** November 2025

---

## 1. Problem Definition and Goals

### 1.1 Purpose
This project implements a "Turing machine"-style pipeline composed of three sequential AI translation agents to explore the robustness of multi-agent systems to noisy input. The system transforms an English sentence through a round-trip translation path (English → French → Hebrew → English) and measures semantic drift using vector embeddings.

### 1.2 Core Research Questions
- How much semantic information is preserved when text passes through multiple translation agents?
- How do spelling errors in the input affect the final output after multi-agent processing?
- What is the relationship between input noise level and semantic drift in a sequential agent pipeline?

### 1.3 Primary Goals
1. **Agent Pipeline Construction**: Build three translation agents forming a sequential processing chain
2. **Semantic Drift Measurement**: Quantify the difference between original and final English text using embedding-based distance metrics
3. **Noise Robustness Analysis**: Systematically evaluate how spelling errors in the input impact semantic preservation through the pipeline
4. **Visualization**: Generate empirical data showing the relationship between input error rate and semantic drift

---

## 2. Scope and Non-goals

### 2.1 In Scope
- ✅ Three sequential translation agents (EN→FR, FR→HE, HE→EN)
- ✅ Python-based implementation using LLM APIs for translation
- ✅ Embedding model integration for semantic similarity measurement
- ✅ Automated spelling error injection at controlled rates
- ✅ Distance metric computation between original and final sentences
- ✅ Data collection across multiple error rates (0%–50%)
- ✅ Graph generation showing error rate vs. semantic distance
- ✅ Basic validation and testing of the pipeline

### 2.2 Out of Scope (Non-goals)
- ❌ Production-ready system with error handling and monitoring
- ❌ Graphical user interface (CLI/script execution is sufficient)
- ❌ Real-time or streaming translation
- ❌ Multi-threading or performance optimization
- ❌ Support for additional languages beyond EN, FR, HE
- ❌ Fine-tuning custom translation or embedding models
- ❌ Statistical significance testing or multiple trial averaging
- ❌ Comparison with human translation quality

---

## 3. System Overview

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Processing                         │
│  • Original English sentence (≥15 words)                     │
│  • Spelling error injection (configurable rate)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Agent Pipeline                              │
│  ┌───────────┐     ┌───────────┐     ┌───────────┐         │
│  │  Agent 1  │ ──► │  Agent 2  │ ──► │  Agent 3  │         │
│  │  EN → FR  │     │  FR → HE  │     │  HE → EN  │         │
│  └───────────┘     └───────────┘     └───────────┘         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Embedding & Distance Computation                │
│  • Embed original sentence                                   │
│  • Embed final output sentence                               │
│  • Calculate distance (e.g., cosine distance)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Analysis & Visualization                          │
│  • Collect distance measurements                             │
│  • Generate error rate vs. distance graph                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Component Roles

**Agent 1 (EN→FR)**: Receives potentially noisy English text and translates to French. Must handle spelling errors gracefully.

**Agent 2 (FR→HE)**: Translates French text to Hebrew. Operates on Agent 1's output.

**Agent 3 (HE→EN)**: Translates Hebrew back to English, completing the round-trip transformation.

**Embedding Module**: Converts text to high-dimensional vectors using a pre-trained model (e.g., sentence-transformers).

**Distance Calculator**: Computes similarity metrics between embedding vectors.

**Error Injector**: Systematically corrupts words in the input sentence at specified rates.

**Experiment Controller**: Orchestrates multiple runs across different error rates and collects results.

---

## 4. Data and Input Specification

### 4.1 Input Requirements

**Base Experiment:**
- Minimum sentence length: **15 words**
- Minimum spelling error rate: **25%** of words
- At least one test sentence, optionally more for validation

**Extended Experiment:**
- Same base sentence(s) tested at multiple error rates: 0%, 10%, 20%, 30%, 40%, 50%
- Each error rate tested independently through the pipeline

### 4.2 Spelling Error Definition

**Error Rate Calculation:**
```
Error Rate = (Number of words with spelling errors) / (Total number of words) × 100%
```

**Error Injection Strategy:**
1. Tokenize sentence into words
2. Calculate target number of words to corrupt: `⌈Error Rate × Word Count⌉`
3. Randomly select words to corrupt (excluding very short words like articles)
4. Apply corruption techniques:
   - Letter substitution (e.g., "hello" → "helo", "wrold")
   - Letter transposition (e.g., "receive" → "recieve")
   - Letter omission (e.g., "spelling" → "speling")
   - Letter duplication (e.g., "error" → "errror")

**Consistency:**
- For fair comparison across error rates, use the same base sentence
- Apply different random seeds or systematic corruption patterns
- Document which words were corrupted for reproducibility

### 4.3 Example Input

**Clean sentence (20 words):**
> "The quick brown fox jumps over the lazy dog while the cat watches from a comfortable position nearby."

**25% errors (5 words corrupted):**
> "The quik brown fox jmps over the lazy dog while the cat waches from a comfortible position nearby."

---

## 5. AI Agents Design

### 5.1 Translation Agent Architecture

Each agent is implemented as a Python class or function that:
1. Receives text in the source language
2. Constructs a prompt for an LLM API (e.g., OpenAI, Anthropic Claude, local model)
3. Sends the translation request
4. Returns the translated text

### 5.2 Agent Specifications

**Agent 1: English → French**
- **Input**: English text (potentially with spelling errors)
- **Prompt strategy**:
  - Instruct the model to translate despite potential typos
  - Example: "Translate the following English text to French. The text may contain spelling errors; please interpret the intended meaning and translate accurately."
- **Output**: French text

**Agent 2: French → Hebrew**
- **Input**: French text from Agent 1
- **Prompt strategy**:
  - Standard translation prompt
  - Example: "Translate the following French text to Hebrew accurately, preserving the meaning."
- **Output**: Hebrew text

**Agent 3: Hebrew → English**
- **Input**: Hebrew text from Agent 2
- **Prompt strategy**:
  - Standard translation prompt
  - Example: "Translate the following Hebrew text to English accurately, preserving the meaning."
- **Output**: English text (final output)

### 5.3 LLM Selection Considerations

**Options:**
- **OpenAI GPT-4/GPT-3.5**: Strong multilingual capabilities, API-based
- **Anthropic Claude**: Excellent instruction following, API-based
- **Local models**: Llama, Mistral (free but may have lower quality)

**Recommendation**: Use a consistent, capable model (e.g., GPT-4 or Claude) across all three agents for reliability.

### 5.4 Handling Noisy Input

**Strategy for Agent 1:**
- Explicitly instruct the model to be robust to spelling errors
- The model should infer intended meaning from context
- Do not ask the model to "correct" spelling first—translate as-is

**Downstream agents:**
- Should receive relatively clean input (assuming Agent 1 produces valid French)
- Standard translation prompts sufficient

---

## 6. Embeddings and Distance Metric

### 6.1 Embedding Model Selection

**Recommended: Sentence Transformers**
- Model: `all-MiniLM-L6-v2` or `all-mpnet-base-v2`
- Rationale:
  - Pre-trained for semantic similarity tasks
  - Efficient and fast
  - Available via Python `sentence-transformers` library
  - Produces 384 or 768-dimensional dense vectors

**Alternative: OpenAI Embeddings**
- Model: `text-embedding-3-small` or `text-embedding-ada-002`
- Rationale: High quality, but API cost considerations

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(sentence)
```

### 6.2 Distance Metric

**Primary Metric: Cosine Distance**
```
Cosine Distance = 1 - Cosine Similarity
Cosine Similarity = (A · B) / (||A|| × ||B||)
```

**Rationale:**
- Cosine distance measures angular difference between vectors
- Invariant to vector magnitude (focuses on direction/meaning)
- Range: [0, 2], where 0 = identical, 2 = opposite
- Standard for semantic similarity tasks

**Alternative: Euclidean Distance**
- Measures absolute distance in embedding space
- May be sensitive to vector magnitude
- Can be used for comparison but less interpretable

**Implementation:**
```python
from scipy.spatial.distance import cosine
distance = cosine(embedding_original, embedding_final)
```

### 6.3 Interpretation

- **Low distance (< 0.1)**: Sentences are semantically very similar
- **Medium distance (0.1–0.5)**: Moderate semantic drift
- **High distance (> 0.5)**: Significant semantic change
- **Threshold values are approximate and model-dependent**

---

## 7. Experiment Design and Evaluation

### 7.1 Base Experiment

**Objective**: Validate that the pipeline works and produces measurable semantic drift with noisy input.

**Procedure:**
1. Select or create an English sentence ≥15 words
2. Inject spelling errors at ≥25% rate
3. Record the corrupted sentence
4. Pass through Agent 1 (EN→FR), capture output
5. Pass through Agent 2 (FR→HE), capture output
6. Pass through Agent 3 (HE→EN), capture output
7. Compute embeddings for original (clean) and final sentences
8. Calculate distance metric
9. Document all intermediate outputs and distance

**Expected Outcome:**
- Distance > 0, showing some semantic drift
- Final English sentence should be somewhat coherent but may differ from original

### 7.2 Extended Experiment (Error Rate Sweep)

**Objective**: Quantify the relationship between spelling error rate and semantic drift.

**Procedure:**
1. Select a clean English sentence ≥15 words
2. For each error rate in [0%, 10%, 20%, 30%, 40%, 50%]:
   a. Create a corrupted version with that error rate
   b. Run through the three-agent pipeline
   c. Compute embedding distance between clean original and final output
   d. Record (error_rate, distance) pair
3. Compile results into a dataset

**Variables:**
- **Independent variable**: Spelling error rate (%)
- **Dependent variable**: Embedding distance
- **Control**: Same base sentence, same agents, same embedding model

**Expected Hypothesis:**
- Distance should generally increase with error rate
- Relationship may be non-linear (agents might handle low errors well, degrade at higher rates)

### 7.3 Validation

**Quality Checks:**
- Manually inspect intermediate translations for reasonableness
- Verify embeddings are computed correctly (sanity check: distance from sentence to itself ≈ 0)
- Test with a known sentence pair with expected similarity

---

## 8. Graph and Visualization

### 8.1 Graph Design

**Graph Type**: Line plot or scatter plot with connecting line

**Axes:**
- **X-axis**: Spelling Error Rate (%) — range [0, 50]
- **Y-axis**: Cosine Distance — range will depend on data, typically [0, 1]

**Elements:**
- Data points at each tested error rate
- Connecting line to show trend
- Grid for readability
- Title: "Semantic Drift vs. Spelling Error Rate in Tri-Lingual Agent Pipeline"
- Axis labels with units

### 8.2 Implementation

**Library**: Matplotlib or Seaborn

**Sample Code:**
```python
import matplotlib.pyplot as plt

error_rates = [0, 10, 20, 30, 40, 50]
distances = [...]  # Computed from experiments

plt.figure(figsize=(10, 6))
plt.plot(error_rates, distances, marker='o', linewidth=2, markersize=8)
plt.xlabel('Spelling Error Rate (%)', fontsize=12)
plt.ylabel('Cosine Distance', fontsize=12)
plt.title('Semantic Drift vs. Spelling Error Rate in Tri-Lingual Agent Pipeline', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(error_rates)
plt.savefig('error_rate_vs_distance.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 8.3 Expected Trends

**Potential Patterns:**
1. **Linear increase**: Distance grows proportionally with error rate
2. **Threshold effect**: Low errors handled well, sharp increase after ~20–30%
3. **Saturation**: Distance plateaus at high error rates (agents maximally confused)

**Insights:**
- Steep slope → agents are sensitive to spelling errors
- Flat slope → agents are robust to noisy input
- Non-monotonic behavior would be interesting (e.g., distance decreases at very high error rates due to random corrections)

---

## 9. Implementation Plan

### 9.1 Development Phases

**Phase 1: Setup and Infrastructure**
1. Set up Python environment (Python 3.8+)
2. Install dependencies:
   - `openai` or `anthropic` for LLM API
   - `sentence-transformers` for embeddings
   - `matplotlib` for visualization
   - `numpy`, `scipy` for numerical operations
3. Configure API keys and test connectivity
4. Create project directory structure:
   ```
   project/
   ├── agents.py           # Translation agent implementations
   ├── embeddings.py       # Embedding and distance functions
   ├── error_injection.py  # Spelling error generator
   ├── pipeline.py         # Main experiment orchestration
   ├── visualization.py    # Graphing functions
   ├── main.py            # Entry point
   └── results/           # Output directory
   ```

**Phase 2: Agent Implementation**
1. Implement `TranslationAgent` class or functions
2. Create three agent instances (EN→FR, FR→HE, HE→EN)
3. Test each agent individually with sample sentences
4. Verify pipeline connectivity (output of Agent 1 → input of Agent 2, etc.)

**Phase 3: Error Injection**
1. Implement spelling error injection function
2. Test error rate calculation (verify 25% corruption produces correct count)
3. Create multiple corrupted versions of test sentences
4. Manually verify error quality (errors should be realistic typos)

**Phase 4: Embedding and Distance**
1. Load sentence transformer model
2. Implement embedding function
3. Implement distance calculation (cosine distance)
4. Test with known sentence pairs
5. Validate: distance(sentence, sentence) ≈ 0

**Phase 5: Base Experiment**
1. Run one complete experiment with ≥25% errors
2. Log all intermediate outputs (EN → FR → HE → EN)
3. Compute and record distance
4. Verify results are reasonable

**Phase 6: Extended Experiment**
1. Implement loop over error rates [0, 10, 20, 30, 40, 50]
2. For each rate:
   - Generate corrupted sentence
   - Run pipeline
   - Compute distance
   - Store result
3. Save results to file (JSON or CSV)

**Phase 7: Visualization**
1. Load experiment results
2. Generate graph with Matplotlib
3. Save graph as high-resolution image
4. Review and refine visualization aesthetics

**Phase 8: Documentation and Deliverables**
1. Prepare final report with:
   - Original and corrupted sentences
   - Word counts and error rates
   - All intermediate translations
   - Distance measurements
   - Graph
2. Clean up code and add comments
3. Create README with usage instructions

### 9.2 Modular Code Structure

**agents.py:**
```python
class TranslationAgent:
    def __init__(self, source_lang, target_lang, model="gpt-4"):
        ...

    def translate(self, text):
        ...
```

**embeddings.py:**
```python
def get_embedding(text, model):
    ...

def calculate_distance(emb1, emb2, metric='cosine'):
    ...
```

**error_injection.py:**
```python
def inject_spelling_errors(sentence, error_rate):
    ...

def corrupt_word(word):
    ...
```

**pipeline.py:**
```python
def run_experiment(sentence, error_rate, agents, embedding_model):
    ...

def run_error_rate_sweep(base_sentence, error_rates, agents, embedding_model):
    ...
```

### 9.3 Testing Strategy

1. **Unit tests**: Test each agent, embedding function, error injection independently
2. **Integration test**: Run full pipeline with known input
3. **Validation**: Manual review of sample outputs
4. **Edge cases**: Empty strings, single words, all errors

---

## 10. Risks, Limitations, and Future Improvements

### 10.1 Risks and Challenges

**LLM API Reliability:**
- Risk: API rate limits, timeouts, or errors during batch processing
- Mitigation: Implement retry logic, handle exceptions, use sleep delays

**Translation Quality Variability:**
- Risk: LLMs may produce inconsistent translations across runs
- Impact: Distance measurements may have variance
- Mitigation: Fix random seeds if possible, or run multiple trials and average

**Embedding Model Limitations:**
- Risk: Embeddings may not perfectly capture semantic similarity
- Impact: Distance may not fully reflect human perception of similarity
- Mitigation: Acknowledge limitation, consider alternative metrics if needed

**Cost:**
- Risk: API costs for LLM calls (especially with GPT-4)
- Mitigation: Use cheaper models (GPT-3.5), limit number of experiments, or use local models

**Language-Specific Issues:**
- Risk: Hebrew uses different script (right-to-left), French has accents
- Impact: Encoding issues or API handling problems
- Mitigation: Ensure UTF-8 encoding, test with simple examples first

### 10.2 Limitations

1. **Single sentence experiments**: Results based on one or few sentences, not statistically robust
2. **No human evaluation**: No validation against human judgment of semantic similarity
3. **Specific language path**: Results specific to EN→FR→HE→EN, may differ with other languages
4. **Synthetic errors**: Spelling errors are artificially injected, may not match real-world error patterns
5. **No error type analysis**: All errors treated equally (typos vs. phonetic errors vs. omissions)
6. **No semantic categorization**: Distance is aggregate, doesn't distinguish what aspects changed (topic, sentiment, details)

### 10.3 Future Improvements

**Methodological Enhancements:**
- Run multiple trials per error rate and compute mean ± standard deviation
- Test with multiple diverse sentences (different lengths, topics, complexity)
- Compare multiple distance metrics (cosine, Euclidean, BERTScore)
- Add human evaluation of semantic similarity for validation

**Expanded Experiments:**
- Test different language paths (e.g., EN→ES→ZH→EN)
- Compare different LLM models (GPT-4 vs. Claude vs. local models)
- Test different error types separately (typos vs. phonetic vs. grammatical)
- Vary position of errors (beginning vs. end of sentence)
- Test with different embedding models

**Advanced Analysis:**
- Track semantic drift at each stage (EN vs. FR, FR vs. HE reconstructed, etc.)
- Analyze which words/concepts are most affected by errors
- Use attention-based methods to identify critical words
- Cluster final outputs to identify common translation patterns

**System Improvements:**
- Add caching to avoid redundant API calls
- Implement parallel processing for multiple experiments
- Create interactive visualization dashboard
- Build automated testing framework
- Add logging and detailed error tracking

**Production Considerations:**
- Add comprehensive error handling
- Implement configuration management (YAML/JSON config files)
- Create Docker container for reproducibility
- Add command-line interface with argparse
- Generate structured experiment reports (PDF, HTML)

---

## Appendix: Deliverables Checklist

For the assignment submission, ensure the following are included:

- [ ] **Original English sentence(s)**: Clean versions before error injection
- [ ] **Word counts**: Number of words in each test sentence
- [ ] **Error rates**: Percentage and exact count of corrupted words
- [ ] **Corrupted sentences**: The actual input sent to Agent 1
- [ ] **Intermediate outputs**:
  - [ ] French output from Agent 1
  - [ ] Hebrew output from Agent 2
  - [ ] Final English output from Agent 3
- [ ] **Distance measurements**: Numerical values for each experiment
- [ ] **Graph**: Error rate (0%–50%) vs. distance plot as PNG/PDF
- [ ] **Code**: All Python scripts used for the experiment
- [ ] **Documentation**: README explaining how to run the code
- [ ] **Analysis**: Brief interpretation of results and graph trends

---

**Document Version:** 1.0
**Last Updated:** November 2025
