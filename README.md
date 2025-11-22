# Tri-Lingual Turing Agent Pipeline

A multi-agent translation system that explores semantic drift through a round-trip translation pipeline (English → French → Hebrew → English) with spelling error robustness analysis.

## Project Overview

This project implements a "Turing machine"-style pipeline composed of three sequential AI translation agents to measure how spelling errors in input text affect semantic preservation through multiple translation steps.

## Features

- Three-agent translation pipeline (EN→FR→HE→EN)
- Automated spelling error injection at configurable rates
- Semantic similarity measurement using embeddings
- Visualization of error rate vs. semantic drift
- Comprehensive experiment orchestration and result tracking

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

# Configure API keys
cp .env.example .env
# Edit .env and add your API key (Anthropic or OpenAI)
```

### 2. Run Experiment

```bash
# Run full error rate sweep (0% to 50%)
python -m src.main

# Run with custom sentence
python -m src.main --sentence "Your custom sentence with at least fifteen words for testing robustness"

# Run single experiment with 25% error rate
python -m src.main --single 0.25

# Use OpenAI instead of Anthropic
python -m src.main --provider openai --model gpt-4
```

### 3. View Results

Results will be saved in the `results/` directory:
- `experiment_results_*.json` - Detailed experiment data
- `error_rate_vs_distance.png` - Main visualization
- `experiment_summary.png` - Comprehensive summary figure

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
│   └── main.py            # Entry point
├── results/               # Output directory (generated)
├── tests/                 # Unit tests
├── .env.example          # Environment template
├── .gitignore
├── requirements.txt
├── README.md
├── RPD.md                # Research/Product/Design document
└── IMPLEMENTATION_PLAN.md  # Development roadmap
```

## Documentation

- **[RPD Document](RPD.md)** - Research/Product/Design specifications
- **[Implementation Plan](IMPLEMENTATION_PLAN.md)** - Detailed development phases

## Requirements

- Python 3.8+
- Anthropic API key OR OpenAI API key
- ~2GB disk space (for embedding models)

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

## Course

AI Agent Systems - University Assignment

## License

Educational use only
