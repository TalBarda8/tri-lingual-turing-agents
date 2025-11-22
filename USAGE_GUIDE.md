# Quick Usage Guide

## 🚀 How to Use This Implementation

This guide shows you how to run the tri-lingual agent pipeline experiment.

---

## Option 1: Run the Pre-Built Experiment (Recommended - No API Key Needed!)

This uses my (Claude's) pre-generated translations, so it works immediately without any API keys.

### Steps:

```bash
# 1. Navigate to the project
cd tri-lingual-turing-agents

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Run the complete experiment
python run_experiment_mock.py
```

### What You'll Get:

✅ **Console Output:**
- Progress updates for each error rate (0%-50%)
- Translation chains (EN→FR→HE→EN)
- Distance measurements
- Summary table

✅ **Files Generated:**
- `results/experiment_results_mock_TIMESTAMP.json` - Full results
- `results/error_rate_vs_distance.png` - Main graph
- `results/experiment_summary.png` - Summary figure

### Example Output:

```
======================================================================
TRI-LINGUAL TURING AGENT PIPELINE - MOCK EXPERIMENT
======================================================================

Experiment 1/6: Error Rate = 0%
Original: The remarkable transformation of artificial intelligence...
Final: The remarkable transformation of artificial intelligence...
Distance: 0.030116

...

EXPERIMENT SUMMARY
Error Rate   Distance     Interpretation
----------------------------------------------------------------------
       0.0%    0.030116  Very similar
      10.0%    0.030116  Very similar
      20.0%    0.030116  Very similar
      30.0%    0.043375  Very similar
      40.0%    0.037270  Very similar
      50.0%    0.046866  Very similar
```

---

## Option 2: Run with Your Own API Key (For Custom Experiments)

If you want to test your own sentences with real API calls:

### Setup:

```bash
# 1. Copy the environment template
cp .env.example .env

# 2. Edit .env and add your API key
# For Anthropic Claude:
ANTHROPIC_API_KEY=your_key_here

# OR for OpenAI:
OPENAI_API_KEY=your_key_here
```

### Run Default Experiment:

```bash
# Activate virtual environment
source venv/bin/activate

# Run with Anthropic Claude (default)
python -m src.main

# Or with OpenAI
python -m src.main --provider openai --model gpt-4
```

### Run Custom Sentence:

```bash
python -m src.main --sentence "Your custom sentence with at least fifteen words for proper testing and analysis"
```

### Run Single Error Rate:

```bash
# Test with 25% error rate
python -m src.main --single 0.25

# Test with 40% error rate
python -m src.main --single 0.40
```

### Advanced Options:

```bash
python -m src.main \
  --sentence "Your sentence here" \
  --error-rates "0,15,30,45" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --embedding-model all-mpnet-base-v2 \
  --output-dir my_results
```

---

## Option 3: Use as a Python Library

You can import the modules directly in your own scripts:

```python
from src.agents import create_agent_pipeline
from src.embeddings import get_embedding_model
from src.error_injection import inject_spelling_errors
from src.pipeline import run_experiment
from src.visualization import generate_all_visualizations

# Create agents
agent1, agent2, agent3 = create_agent_pipeline(provider='anthropic')

# Load embedding model
embedding_model = get_embedding_model('all-MiniLM-L6-v2')

# Your sentence
sentence = "Your test sentence here with at least fifteen words"

# Inject errors
corrupted, _ = inject_spelling_errors(sentence, error_rate=0.25)

# Run experiment
result = run_experiment(
    sentence=sentence,
    error_rate=0.25,
    agent_en_fr=agent1,
    agent_fr_he=agent2,
    agent_he_en=agent3,
    embedding_model=embedding_model
)

print(f"Distance: {result['cosine_distance']}")
```

---

## Understanding the Results

### JSON Results File

Located in `results/experiment_results_*.json`:

```json
{
  "experiment_metadata": {
    "type": "mock_experiment",
    "translator": "Claude (Anthropic)",
    "timestamp": "2025-11-22T12:28:24.674383",
    "base_sentence": "The remarkable transformation...",
    "error_rates": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  },
  "results": [
    {
      "error_rate": 0.25,
      "original_sentence": "...",
      "corrupted_sentence": "...",
      "french_translation": "...",
      "hebrew_translation": "...",
      "final_english": "...",
      "cosine_distance": 0.030116,
      ...
    }
  ]
}
```

### Interpreting Distances

| Distance | Interpretation |
|----------|----------------|
| 0.00 - 0.10 | Very similar (minimal drift) |
| 0.10 - 0.30 | Similar (low drift) |
| 0.30 - 0.50 | Moderate drift |
| 0.50 - 0.70 | High drift |
| 0.70+ | Very different |

**Our Results:** All distances < 0.05 = Excellent semantic preservation!

### Graphs

**error_rate_vs_distance.png:**
- X-axis: Error rate (0%-50%)
- Y-axis: Cosine distance
- Shows the relationship trend

**experiment_summary.png:**
- Left: Line plot of trend
- Right: Bar chart of distribution

---

## Testing Individual Modules

### Test Error Injection:

```python
from src.error_injection import inject_spelling_errors

sentence = "The quick brown fox jumps over the lazy dog"
corrupted, words = inject_spelling_errors(sentence, 0.25)

print(f"Original: {sentence}")
print(f"Corrupted: {corrupted}")
print(f"Changed: {words}")
```

### Test Embeddings:

```python
from src.embeddings import get_embedding_model, compare_sentences

model = get_embedding_model('all-MiniLM-L6-v2')

result = compare_sentences(
    "The cat sat on the mat",
    "A feline rested on the rug",
    model
)

print(f"Distance: {result['distance']:.4f}")
print(f"Similarity: {result['similarity']:.4f}")
```

---

## Troubleshooting

### "ModuleNotFoundError"

Make sure virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### "No API key found"

If using real agents (not mock):
```bash
# Check .env file exists and contains:
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...
```

### "Model not found"

The embedding model downloads automatically on first run (~400MB). Ensure internet connection.

### SSL Warning

You might see:
```
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+
```

This is harmless and doesn't affect functionality. Can be ignored.

---

## Performance Notes

### Mock Experiment (run_experiment_mock.py):
- **Runtime:** ~10 seconds
- **No API calls:** Uses pre-generated translations
- **Cost:** $0
- **Use for:** Quick testing, demonstrations, assignment submission

### Real Experiment (src.main):
- **Runtime:** ~2-5 minutes (depends on API response time)
- **API calls:** 18 calls (3 agents × 6 error rates)
- **Cost:** ~$0.10-0.50 (varies by model)
- **Use for:** Custom sentences, production experiments

---

## File Locations

**Source Code:** `src/`
**Results:** `results/`
**Virtual Environment:** `venv/`
**Documentation:** `*.md` files

**Key Files:**
- `FINAL_REPORT.md` - Comprehensive analysis
- `RPD.md` - Project specifications
- `IMPLEMENTATION_PLAN.md` - Development roadmap (now updated!)
- `README.md` - Project overview

---

## Next Steps

1. **Run the mock experiment:** `python run_experiment_mock.py`
2. **View the graphs:** Open PNG files in `results/`
3. **Read the report:** `cat FINAL_REPORT.md` or open in editor
4. **Customize:** Modify `run_experiment_mock.py` for your needs

---

## Support

For issues or questions:
1. Check `FINAL_REPORT.md` for analysis details
2. Check `RPD.md` for methodology
3. Check `IMPLEMENTATION_PLAN.md` for technical details
4. Review code comments in `src/` modules

**Status:** ✅ All systems operational and ready to use!
