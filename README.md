# Tri-Lingual Turing Agent Pipeline

A multi-agent translation system that explores semantic drift through a round-trip translation pipeline (English → French → Hebrew → English) with spelling error robustness analysis.

## Project Overview

This project implements a "Turing machine"-style pipeline composed of three sequential AI translation agents to measure how spelling errors in input text affect semantic preservation through multiple translation steps.

## Features

- **🎨 Interactive UI** - Beautiful step-by-step visualization showing agents working in real-time
- **🤖 Real Agent Support** - Use actual Claude Code translation agents OR mock agents
- Three-agent translation pipeline (EN→FR→HE→EN)
- Automated spelling error injection at configurable rates
- Semantic similarity measurement using embeddings
- Visualization of error rate vs. semantic drift
- Comprehensive experiment orchestration and result tracking
- **No API key required** - Pre-built demo with Claude-generated translations

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/TalBarda8/tri-lingual-turing-agents.git
cd tri-lingual-turing-agents

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode (required for tests)
pip install -e .
```

### 2. Run the Interactive Demo (Recommended!)

**🎨 The easiest way to see the system in action - NO API KEY REQUIRED!**

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the beautiful interactive UI
python run_interactive.py
```

**What you'll see:**
- 🤖 Beautiful colored panels and headers
- 🇬🇧 → 🇫🇷 → 🇮🇱 → 🇬🇧 Translation chain with country flags
- ⏳ Progress bars showing each agent "thinking"
- 📊 Tables displaying input/output for every translation step
- 📈 Real-time semantic distance calculation with color-coded interpretation
- ✅ Summary table showing results across all error rates

**Interactive Mode Options:**
1. **Quick Demo** - Tests 0%, 25%, 50% error rates (fast preview)
2. **Full Analysis** - Tests 0%, 10%, 20%, 30%, 40%, 50% (comprehensive)
3. **Single Test** - Choose your own custom error rate

**Example Output:**
```
╭──────────────────────────────────────────────────────────────╮
│  🤖 AGENT 1: English → French 🇬🇧 → 🇫🇷                      │
╰──────────────────────────────────────────────────────────────╯
  Agent 1 is translating...

  Input (English)                    Output (French)
 ──────────────────────────────────────────────────────────────
  The remarkable transformation...   La transformation...

╭──────────────────────────────────────────────────────────────╮
│  Cosine Distance: 0.009197                                   │
│  Interpretation: ✅ Very similar - Minimal semantic drift!   │
╰──────────────────────────────────────────────────────────────╯
```

### 3. Alternative: Run Automated Experiment

For batch processing without the interactive UI:

```bash
# No API key needed - uses pre-generated translations
python run_experiment_mock.py
```

Results saved in `results/`:
- `experiment_results_mock_*.json` - Full data
- `error_rate_vs_distance.png` - Main graph
- `experiment_summary.png` - Summary figure

### 4. Advanced: Run with Your Own API Key

If you want to test custom sentences with real-time API calls:

```bash
# Configure API keys (only needed for custom experiments)
cp .env.example .env
# Edit .env and add your API key (Anthropic or OpenAI)

# Run full error rate sweep (0% to 50%)
python -m src.main

# Run with custom sentence
python -m src.main --sentence "Your custom sentence with at least fifteen words for testing robustness"

# Run single experiment with 25% error rate
python -m src.main --single 0.25

# Use OpenAI instead of Anthropic
python -m src.main --provider openai --model gpt-4
```

## Project Structure

```
tri-lingual-turing-agents/
├── src/
│   ├── __init__.py
│   ├── agents.py           # Translation agent implementations
│   ├── embeddings.py       # Embedding and distance functions
│   ├── error_injection.py  # Spelling error generator
│   ├── pipeline.py         # Experiment orchestration
│   ├── visualization.py    # Graph generation
│   └── main.py            # CLI entry point
├── run_interactive.py     # 🎨 Interactive UI (RECOMMENDED!)
├── run_experiment_mock.py # Automated demo (no API key)
├── run_experiment_with_real_agents.py  # 🤖 Real agent orchestration
├── run_full_experiment_suite.py        # Batch preparation for real agents
├── compile_real_agent_results.py       # Results compiler for real agents
├── results/               # Output directory (generated)
├── tests/                 # Unit tests
├── .env.example          # Environment template
├── .gitignore
├── requirements.txt
├── README.md
├── RPD.md                # Research/Product/Design document
├── USAGE_GUIDE.md        # Detailed usage instructions
├── REAL_AGENTS_GUIDE.md  # Real vs. mock agents guide
├── FINAL_REPORT.md       # Experimental results and analysis
└── IMPLEMENTATION_PLAN.md  # Development roadmap
```

## Documentation

- **[Usage Guide](USAGE_GUIDE.md)** - Complete usage instructions with examples
- **[Real Agents Guide](REAL_AGENTS_GUIDE.md)** - How to use actual Claude Code agents vs. mock agents
- **[RPD Document](RPD.md)** - Research/Product/Design specifications
- **[Implementation Plan](IMPLEMENTATION_PLAN.md)** - Detailed development phases
- **[Final Report](FINAL_REPORT.md)** - Experimental results and analysis
- **[Test Coverage Report](TEST_COVERAGE_FINAL_REPORT.md)** - Comprehensive test coverage documentation

## Running Tests

The project has **94% test coverage** with 197 comprehensive tests.

**⚠️ Prerequisites**: Make sure you've installed the package in editable mode:
```bash
pip install -e .
```

### Quick Test Run
```bash
# Make sure you're in the project directory with venv activated
python3 -m pytest tests/ -v
```

### Run Tests with Coverage Report
```bash
python3 -m pytest tests/ --cov=src/tri_lingual_agents --cov-report=term
```

### Generate Interactive HTML Coverage Report
```bash
python3 -m pytest tests/ --cov=src/tri_lingual_agents --cov-report=html
open htmlcov/index.html  # Opens interactive coverage report in browser
```

### Run Specific Test Modules
```bash
python3 -m pytest tests/test_pipeline/ -v          # Pipeline orchestration tests
python3 -m pytest tests/test_visualization/ -v     # Visualization tests
python3 -m pytest tests/test_parallel/ -v          # Parallel processing tests
python3 -m pytest tests/test_embeddings/ -v        # Embeddings tests
python3 -m pytest tests/test_agents/ -v            # Agent tests
python3 -m pytest tests/test_error_injection/ -v   # Error injection tests
```

**Expected Results:**
- ✅ 195/197 tests passing (99%)
- ✅ 94% overall coverage
- ⚠️ 2 expected failures (OpenAI API key tests - validates error handling)

See **[TEST_COVERAGE_FINAL_REPORT.md](TEST_COVERAGE_FINAL_REPORT.md)** for detailed coverage analysis.

## Requirements

- Python 3.8+
- ~2GB disk space (for embedding models)
- **Optional**: Anthropic API key OR OpenAI API key (only for custom experiments)

## Configuration

Edit `.env` file to configure:

```bash
# Choose your LLM provider
ANTHROPIC_API_KEY=your_key_here
# OR
OPENAI_API_KEY=your_key_here

# Optional: Override defaults
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Command Line Options

```
python -m src.main [OPTIONS]

Options:
  --sentence TEXT         Base sentence to test (≥15 words required)
  --error-rates TEXT      Comma-separated rates (e.g., "0,10,20,30,40,50")
  --provider {anthropic,openai}
                         LLM provider to use
  --model TEXT           Specific model name
  --embedding-model TEXT  Sentence transformer model
  --output-dir TEXT      Output directory for results
  --single FLOAT         Run single experiment with this error rate (0.0-1.0)
```

## Understanding the Interactive UI

The interactive demo (`run_interactive.py`) shows you exactly what's happening at each step:

### Step-by-Step Process

1. **System Introduction**
   - Explains the 3-agent pipeline
   - Shows what the experiment measures

2. **Original Sentence**
   - Displays the clean English sentence (20 words)
   - Shows word count

3. **Spelling Error Injection** (if error rate > 0%)
   - Shows before/after comparison
   - Lists corrupted words
   - Visual table highlighting changes

4. **Agent 1: English → French** 🇬🇧 → 🇫🇷
   - Progress bar showing translation in progress
   - Input: English (clean or corrupted)
   - Output: French translation
   - Table comparing both

5. **Agent 2: French → Hebrew** 🇫🇷 → 🇮🇱
   - Progress bar showing translation in progress
   - Input: French text
   - Output: Hebrew translation
   - Table comparing both

6. **Agent 3: Hebrew → English** 🇮🇱 → 🇬🇧
   - Progress bar showing translation in progress
   - Input: Hebrew text
   - Output: Final English translation
   - Table comparing both

7. **Semantic Similarity Analysis** 📊
   - Computes embeddings for original and final English
   - Calculates cosine distance
   - Shows distance value (e.g., 0.009197)
   - Color-coded interpretation:
     - ✅ **Green** (< 0.1): Very similar - Minimal semantic drift
     - ✓ **Yellow** (0.1-0.3): Similar - Low semantic drift
     - ~ **Orange** (0.3-0.5): Moderate drift
     - ⚠ **Red** (> 0.5): High semantic drift

8. **Final Summary**
   - Table showing all error rates tested
   - Distance measurements for each
   - Drift percentage compared to baseline
   - Status indicators for each result

### Key Findings

The experiment demonstrates that:
- **0-20% error rate**: Distance ≈ 0.030 (no measurable impact)
- **30% error rate**: Distance ≈ 0.043 (threshold effect begins)
- **50% error rate**: Distance ≈ 0.047 (still "very similar")
- **Conclusion**: LLMs are exceptionally robust to spelling errors
- **Semantic preservation**: >95% across all error rates

See [FINAL_REPORT.md](FINAL_REPORT.md) for detailed analysis.

## Course

AI Agent Systems - University Assignment

## License

Educational use only
