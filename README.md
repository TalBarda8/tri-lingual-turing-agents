# Tri-Lingual Turing Agent Pipeline

A multi-agent translation system that explores semantic drift through a round-trip translation pipeline (English → French → Hebrew → English) with spelling error robustness analysis.

## Project Overview

This project implements a "Turing machine"-style pipeline composed of three sequential AI translation agents to measure how spelling errors in input text affect semantic preservation through multiple translation steps.

## Features

- **🎨 Interactive UI** - Beautiful step-by-step visualization showing agents working in real-time
- **🤖 Multi-Agent Translation** - Three specialized Claude Code translation agents working sequentially
- Three-agent translation pipeline (EN→FR→HE→EN)
- Automated spelling error injection at configurable rates
- Semantic similarity measurement using embeddings
- Visualization of error rate vs. semantic drift
- Comprehensive experiment orchestration and result tracking
- Pre-built demo with Claude-generated translations

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

**🎨 The easiest way to see the system in action!**

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the beautiful interactive UI
python scripts/run_interactive.py
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

### 3. Advanced: Run with Your Own API Key

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
├── docs/
│   ├── PRD.md             # Product Requirements Document
│   └── ARCHITECTURE.md    # System architecture with C4 diagrams
├── run_interactive.py     # 🎨 Interactive UI (RECOMMENDED!)
├── run_experiment_with_real_agents.py  # 🤖 Real agent orchestration
├── run_full_experiment_suite.py        # Batch preparation for real agents
├── compile_real_agent_results.py       # Results compiler for real agents
├── results/               # Output directory (generated)
├── tests/                 # Unit tests
├── .env.example          # Environment template
├── .gitignore
├── requirements.txt
├── README.md             # Project overview
├── FINAL_REPORT.md       # Experimental results and analysis
└── TEST_COVERAGE_FINAL_REPORT.md  # Test coverage documentation
```

## Documentation

- **[Product Requirements](docs/PRD.md)** - Complete product specifications and requirements
- **[Architecture](docs/ARCHITECTURE.md)** - System architecture with C4 diagrams
- **[Final Report](FINAL_REPORT.md)** - Experimental results and analysis
- **[Test Coverage](TEST_COVERAGE_FINAL_REPORT.md)** - Comprehensive test coverage documentation (94% coverage)

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

The interactive demo (`scripts/run_interactive.py`) shows you exactly what's happening at each step:

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

## Troubleshooting

This section covers common issues and their solutions.

### 1. Installation Errors

#### Problem: `pip install -e .` fails with "No module named setuptools"
**Solution:**
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

#### Problem: Package installation fails on M1/M2 Mac
**Solution:**
```bash
# Use conda for ARM64 compatibility
conda create -n tri-lingual python=3.9
conda activate tri-lingual
pip install -r requirements.txt
pip install -e .
```

### 2. Embedding Model Download Issues

#### Problem: "ConnectionError: Can't reach HuggingFace" or slow download
**Solution:**
```bash
# Pre-download the model manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Or use alternative mirror (China/restricted regions)
export HF_ENDPOINT=https://hf-mirror.com
python scripts/run_interactive.py
```

#### Problem: "OSError: Disk quota exceeded" during model download
**Solution:**
```bash
# Check available space (need ~2GB)
df -h

# Or specify custom cache directory with more space
export TRANSFORMERS_CACHE=/path/to/larger/disk
export SENTENCE_TRANSFORMERS_HOME=/path/to/larger/disk
```

### 3. API Key Issues

#### Problem: "AuthenticationError: Invalid API key"
**Solution:**
```bash
# Verify .env file exists and has correct format
cat .env
# Should show: ANTHROPIC_API_KEY=sk-ant-...

# Ensure no extra spaces or quotes
# Correct:   ANTHROPIC_API_KEY=sk-ant-api123456
# Incorrect: ANTHROPIC_API_KEY = "sk-ant-api123456"

# Verify key is loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('ANTHROPIC_API_KEY'))"
```

#### Problem: "RateLimitError: Rate limit exceeded"
**Solution:**
- **Free tier limit**: Wait 60 seconds between experiments
- **Use mock mode**: Run `python scripts/run_experiment_mock.py` instead
- **Upgrade API tier**: Contact Anthropic for higher limits

### 4. Test Failures

#### Problem: Tests fail with "ModuleNotFoundError: No module named 'tri_lingual_agents'"
**Solution:**
```bash
# Install package in editable mode first
pip install -e .

# Verify installation
python -c "import tri_lingual_agents; print(tri_lingual_agents.__version__)"
# Should print: 1.0.0
```

#### Problem: Specific test fails: `test_openai_translator`
**Solution:**
This is expected if you don't have an OpenAI API key. The test validates error handling:
```bash
# Run tests excluding OpenAI tests
pytest tests/ -v -k "not openai"

# Or set a dummy key to pass validation tests
export OPENAI_API_KEY=sk-dummy-key-for-testing
```

### 5. Runtime Errors

#### Problem: "RuntimeError: CUDA out of memory" on GPU systems
**Solution:**
```bash
# Force CPU usage for embeddings
export CUDA_VISIBLE_DEVICES=""
python scripts/run_interactive.py

# Or modify code to use CPU explicitly
# In src/tri_lingual_agents/embeddings/distance.py, set device='cpu'
```

#### Problem: Hebrew text displays as "???" or boxes
**Solution:**
```bash
# Ensure UTF-8 encoding
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# On Windows, use UTF-8 code page
chcp 65001
```

#### Problem: "Timeout waiting for agent response"
**Solution:**
- **Increase timeout**: Edit `src/tri_lingual_agents/agents/translators.py`, change `timeout=300` to `timeout=600`
- **Check internet**: Agents need stable connection to API
- **Use mock mode**: For offline testing, use `python scripts/run_experiment_mock.py`

### 6. Import Errors After Updates

#### Problem: "ImportError: cannot import name 'ParallelAgentOrchestrator'"
**Solution:**
```bash
# Reinstall package after code changes
pip install -e . --force-reinstall --no-deps

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

### 7. Notebook Issues

#### Problem: Jupyter notebook kernel crashes when running analysis
**Solution:**
```bash
# Install ipykernel in virtual environment
pip install ipykernel
python -m ipykernel install --user --name=tri-lingual

# Select "tri-lingual" kernel in Jupyter
# Kernel > Change Kernel > tri-lingual
```

#### Problem: "FileNotFoundError: results/experiment_results_*.json"
**Solution:**
Run an experiment first to generate results:
```bash
python scripts/run_experiment_mock.py
# Then run the notebook
```

### 8. Performance Issues

#### Problem: Experiments take too long (>5 minutes per error rate)
**Solution:**
- **Use parallel processing**: Ensure `use_parallel=True` in code
- **Reduce error rates**: Test fewer points: `[0, 0.25, 0.5]` instead of 6 rates
- **Use mock mode**: `python scripts/run_experiment_mock.py` (instant results)

### 9. Windows-Specific Issues

#### Problem: "FileNotFoundError" or path errors on Windows
**Solution:**
```bash
# Use raw strings for paths or forward slashes
# Instead of: path = "C:\Users\..."
# Use: path = r"C:\Users\..." or path = "C:/Users/..."

# Or run from WSL (Windows Subsystem for Linux)
wsl
cd /mnt/c/path/to/project
source venv/bin/activate
```

### 10. Getting More Help

If issues persist:

1. **Check logs**: Look for error messages in console output
2. **Verify Python version**: Must be 3.8+ (`python --version`)
3. **Check dependencies**: `pip list | grep -E "anthropic|sentence-transformers|torch"`
4. **Report issue**: [GitHub Issues](https://github.com/TalBarda8/tri-lingual-turing-agents/issues)
5. **Enable debug mode**: Set `export DEBUG=1` before running scripts

**Common Debug Commands:**
```bash
# Verify environment
python -c "import sys; print(f'Python {sys.version}')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import anthropic; print(f'Anthropic {anthropic.__version__}')"

# Test imports
python -c "from tri_lingual_agents import run_experiment; print('✓ Import successful')"

# Check file structure
ls -la src/tri_lingual_agents/
```

---

## Course

AI Agent Systems - University Assignment

## License

Educational use only
