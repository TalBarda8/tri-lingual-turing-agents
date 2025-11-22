# Implementation Plan: Tri-Lingual Turing Agent Pipeline

**Status:** In Progress
**Started:** November 2025
**Reference:** See [RPD.md](RPD.md) for detailed specifications

---

## Overview

This document tracks the step-by-step implementation of the tri-lingual agent pipeline project. Each phase builds upon the previous one, following the plan outlined in Section 9 of the RPD.

---

## Phase 1: Setup and Infrastructure ⏳

**Goal:** Create the project structure and set up the development environment.

### Tasks

- [ ] **1.1 Create project directory structure**
  ```
  tri-lingual-turing-agents/
  ├── src/
  │   ├── __init__.py
  │   ├── agents.py           # Translation agent implementations
  │   ├── embeddings.py       # Embedding and distance functions
  │   ├── error_injection.py  # Spelling error generator
  │   ├── pipeline.py         # Main experiment orchestration
  │   ├── visualization.py    # Graphing functions
  │   └── main.py            # Entry point
  ├── results/               # Output directory (created at runtime)
  ├── tests/                 # Unit tests (optional)
  ├── .env                   # API keys (not committed)
  ├── .gitignore            # Git ignore file
  ├── requirements.txt       # Python dependencies
  ├── README.md             # Project overview
  └── RPD.md                # Research/Product/Design document
  ```

- [ ] **1.2 Set up Python virtual environment**
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

- [ ] **1.3 Create requirements.txt**
  - Dependencies needed:
    - `anthropic` or `openai` (for LLM API)
    - `sentence-transformers` (for embeddings)
    - `matplotlib` (for visualization)
    - `numpy` (numerical operations)
    - `scipy` (distance calculations)
    - `python-dotenv` (environment variables)

- [ ] **1.4 Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **1.5 Configure API keys**
  - Create `.env` file with API credentials
  - Test API connectivity

- [ ] **1.6 Create .gitignore**
  - Exclude: `venv/`, `.env`, `__pycache__/`, `results/`, `*.pyc`

**Completion Criteria:**
- ✅ All directories and files created
- ✅ Virtual environment activated
- ✅ All packages installed successfully
- ✅ API key configured and tested

---

## Phase 2: Agent Implementation 📝

**Goal:** Build the three translation agents (EN→FR, FR→HE, HE→EN).

### Tasks

- [ ] **2.1 Implement TranslationAgent class in agents.py**
  - Constructor: `__init__(source_lang, target_lang, model, api_key)`
  - Method: `translate(text) -> str`
  - Proper error handling for API calls
  - Logging for debugging

- [ ] **2.2 Create three agent instances**
  - Agent 1: English → French
  - Agent 2: French → Hebrew
  - Agent 3: Hebrew → English

- [ ] **2.3 Design prompts for each agent**
  - Agent 1: Robust to spelling errors
  - Agent 2 & 3: Standard translation prompts

- [ ] **2.4 Test each agent individually**
  - Test Agent 1 with clean English text
  - Test Agent 1 with noisy English text (spelling errors)
  - Test Agent 2 with French text
  - Test Agent 3 with Hebrew text
  - Verify output quality

- [ ] **2.5 Test full pipeline connectivity**
  - Run: English → Agent 1 → Agent 2 → Agent 3 → English
  - Verify data flows correctly
  - Document sample outputs

**Completion Criteria:**
- ✅ All three agents implemented and functional
- ✅ Each agent tested independently
- ✅ Full pipeline tested end-to-end
- ✅ Sample outputs documented

---

## Phase 3: Error Injection 🔧

**Goal:** Implement spelling error injection with configurable error rates.

### Tasks

- [ ] **3.1 Implement error injection functions in error_injection.py**
  - `inject_spelling_errors(sentence, error_rate) -> str`
  - `corrupt_word(word) -> str`
  - `calculate_error_count(sentence, error_rate) -> int`

- [ ] **3.2 Implement corruption techniques**
  - Letter substitution
  - Letter transposition
  - Letter omission
  - Letter duplication
  - Random selection of technique

- [ ] **3.3 Add word filtering logic**
  - Exclude very short words (≤2 characters)
  - Exclude common articles/prepositions (optional)

- [ ] **3.4 Test error rate calculation**
  - Verify 25% corruption produces correct count
  - Test with various sentence lengths

- [ ] **3.5 Create test sentences with multiple error rates**
  - 0% (clean baseline)
  - 10%, 20%, 30%, 40%, 50%
  - Document which words were corrupted

- [ ] **3.6 Manual verification**
  - Review corrupted sentences for realism
  - Ensure errors look like natural typos

**Completion Criteria:**
- ✅ Error injection functions implemented
- ✅ Multiple corruption techniques working
- ✅ Error rate calculation verified accurate
- ✅ Test sentences created and validated

---

## Phase 4: Embedding and Distance 📊

**Goal:** Implement semantic similarity measurement using embeddings.

### Tasks

- [ ] **4.1 Implement embedding functions in embeddings.py**
  - `get_embedding(text, model) -> np.ndarray`
  - `calculate_distance(emb1, emb2, metric='cosine') -> float`
  - `load_embedding_model(model_name) -> SentenceTransformer`

- [ ] **4.2 Select and load embedding model**
  - Choose: `all-MiniLM-L6-v2` or `all-mpnet-base-v2`
  - Test model loading
  - Verify output dimensions

- [ ] **4.3 Implement distance metrics**
  - Primary: Cosine distance
  - Alternative: Euclidean distance (optional)

- [ ] **4.4 Test with known sentence pairs**
  - Identical sentences (distance ≈ 0)
  - Similar sentences (distance < 0.3)
  - Unrelated sentences (distance > 0.5)

- [ ] **4.5 Validate embedding computation**
  - Test: `distance(sentence, sentence) ≈ 0`
  - Test: `distance(sentence1, sentence2)` is symmetric
  - Document typical distance ranges

**Completion Criteria:**
- ✅ Embedding model loaded successfully
- ✅ Distance calculation working correctly
- ✅ Validation tests passing
- ✅ Sanity checks completed

---

## Phase 5: Base Experiment 🧪

**Goal:** Run one complete experiment with ≥25% spelling errors.

### Tasks

- [ ] **5.1 Create base test sentence**
  - Minimum 15 words
  - Meaningful content
  - Document the clean version

- [ ] **5.2 Implement pipeline.py orchestration**
  - `run_experiment(sentence, error_rate, agents, embedding_model) -> dict`
  - Return: original, corrupted, intermediate outputs, final, distance

- [ ] **5.3 Run base experiment**
  - Inject 25% spelling errors
  - Pass through all three agents
  - Capture all intermediate outputs:
    - Original English (clean)
    - Corrupted English (input to Agent 1)
    - French output (Agent 1)
    - Hebrew output (Agent 2)
    - Final English (Agent 3)

- [ ] **5.4 Compute embeddings and distance**
  - Embed original (clean) sentence
  - Embed final English output
  - Calculate cosine distance

- [ ] **5.5 Document results**
  - Save all outputs to file
  - Record distance measurement
  - Manual review: Does the output make sense?

- [ ] **5.6 Verify results are reasonable**
  - Distance > 0 (some drift expected)
  - Final sentence is coherent
  - Meaning somewhat preserved

**Completion Criteria:**
- ✅ Pipeline orchestration implemented
- ✅ Base experiment executed successfully
- ✅ All outputs captured and logged
- ✅ Distance measurement recorded
- ✅ Results reviewed and validated

---

## Phase 6: Extended Experiment 📈

**Goal:** Systematically vary error rates from 0% to 50% and collect data.

### Tasks

- [ ] **6.1 Implement error rate sweep in pipeline.py**
  - `run_error_rate_sweep(base_sentence, error_rates, agents, embedding_model) -> list`
  - Loop over: [0%, 10%, 20%, 30%, 40%, 50%]
  - For each rate: run experiment, collect distance

- [ ] **6.2 Run sweep with base sentence**
  - Same sentence for all error rates
  - Record (error_rate, distance) pairs
  - Save intermediate outputs for each run

- [ ] **6.3 Store results to file**
  - Format: JSON or CSV
  - Include metadata: sentence, word count, model used, timestamp

- [ ] **6.4 Verify data consistency**
  - Check all 6 error rates completed
  - Verify distance values are numeric
  - Look for trends in data

**Completion Criteria:**
- ✅ Error rate sweep implemented
- ✅ All 6 error rates tested
- ✅ Results saved to structured file
- ✅ Data verified for consistency

---

## Phase 7: Visualization 📊

**Goal:** Generate graph showing error rate vs. semantic distance.

### Tasks

- [ ] **7.1 Implement visualization.py**
  - `plot_error_vs_distance(error_rates, distances, output_path) -> None`
  - Use Matplotlib
  - Proper labels, title, grid

- [ ] **7.2 Design graph aesthetics**
  - X-axis: Spelling Error Rate (%)
  - Y-axis: Cosine Distance
  - Title: "Semantic Drift vs. Spelling Error Rate in Tri-Lingual Agent Pipeline"
  - Markers and line for clarity

- [ ] **7.3 Generate graph from results**
  - Load data from Phase 6
  - Create plot
  - Save as high-resolution PNG (300 DPI)

- [ ] **7.4 Review and refine visualization**
  - Check readability
  - Adjust colors/fonts if needed
  - Ensure professional appearance

**Completion Criteria:**
- ✅ Visualization module implemented
- ✅ Graph generated successfully
- ✅ Output saved as high-quality image
- ✅ Graph is clear and informative

---

## Phase 8: Documentation and Deliverables 📝

**Goal:** Prepare final report and organize deliverables for submission.

### Tasks

- [ ] **8.1 Create comprehensive project README**
  - Installation instructions
  - Usage examples
  - Configuration guide
  - Dependencies list

- [ ] **8.2 Update RPD.md if needed**
  - Document any deviations from plan
  - Add lessons learned

- [ ] **8.3 Prepare final report**
  - Original sentence(s) used
  - Word counts and error rates
  - All intermediate translations
  - Distance measurements table
  - Graph (embedded or attached)
  - Analysis and interpretation

- [ ] **8.4 Clean up code**
  - Add docstrings to all functions
  - Add inline comments for clarity
  - Remove debug print statements
  - Format code consistently (PEP 8)

- [ ] **8.5 Organize deliverables**
  - Code: All Python scripts
  - Data: Results files (JSON/CSV)
  - Visualizations: Graph PNG
  - Documentation: README, RPD, final report
  - Optional: Presentation slides

- [ ] **8.6 Create results summary**
  - Table of error rates vs. distances
  - Interpretation of trends
  - Discussion of findings

**Completion Criteria:**
- ✅ All documentation complete
- ✅ Code cleaned and commented
- ✅ Final report written
- ✅ All deliverables organized
- ✅ Ready for submission

---

## Testing and Validation Checklist ✅

Throughout development, ensure:

- [ ] Unit tests for each module (optional but recommended)
- [ ] Integration test: Full pipeline with known input
- [ ] Edge case testing:
  - [ ] Empty strings
  - [ ] Single word sentences
  - [ ] 100% error rate
  - [ ] Special characters
- [ ] Manual review of sample outputs at each phase
- [ ] Distance sanity check: `distance(x, x) ≈ 0`
- [ ] API error handling tested

---

## Progress Tracking

| Phase | Status | Completion Date | Notes |
|-------|--------|-----------------|-------|
| Phase 1: Setup | ⏳ Not Started | - | - |
| Phase 2: Agents | ⏳ Not Started | - | - |
| Phase 3: Error Injection | ⏳ Not Started | - | - |
| Phase 4: Embeddings | ⏳ Not Started | - | - |
| Phase 5: Base Experiment | ⏳ Not Started | - | - |
| Phase 6: Extended Experiment | ⏳ Not Started | - | - |
| Phase 7: Visualization | ⏳ Not Started | - | - |
| Phase 8: Documentation | ⏳ Not Started | - | - |

**Legend:**
- ⏳ Not Started
- 🔄 In Progress
- ✅ Completed
- ⚠️ Blocked/Issues

---

## Notes and Decisions

### Design Decisions
- **LLM Choice:** TBD (OpenAI GPT-4, Anthropic Claude, or local model)
- **Embedding Model:** TBD (all-MiniLM-L6-v2 vs. all-mpnet-base-v2)
- **Base Sentence:** TBD (will be selected in Phase 5)

### Issues and Blockers
- None yet

### Lessons Learned
- Will be updated as we progress

---

**Last Updated:** November 2025
**Next Review:** After Phase 1 completion
