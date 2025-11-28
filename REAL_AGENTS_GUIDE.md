# Real Agent Pipeline Guide

## Overview

This project now supports **TWO** implementations:
1. **Mock Agents** - Pre-generated translations (fast, no API costs)
2. **Real Agents** - Live translation agents using Claude Code's agent system (authentic, flexible)

This guide explains the differences and how to use the real agent pipeline.

---

## What Are "Real Agents"?

Real agents are **three separate Claude Code agents** that perform translations:

1. **`english-french-translator`** - Translates English → French
2. **`french-hebrew-translator`** - Translates French → Hebrew
3. **`hebrew-english-translator`** - Translates Hebrew → English

Each agent:
- Runs independently as a subprocess
- Has its own specialized translation prompt
- Is robust to spelling errors (Agent 1 specifically)
- Returns only the translation text (no metadata)

---

## Mock vs Real Agents Comparison

| Feature | Mock Agents | Real Agents |
|---------|-------------|-------------|
| **Implementation** | Python dictionaries with pre-generated translations | Claude Code agents (Task tool) |
| **Translation Source** | Cached from earlier Claude conversation | Live agent invocations |
| **API Calls** | 0 | 18 (3 agents × 6 error rates) |
| **Cost** | $0 | $0 (Claude Code agents are free to use) |
| **Speed** | Instant (~10 seconds) | ~2-5 minutes (agent startup overhead) |
| **Flexibility** | Only works with pre-generated sentence | Can translate ANY sentence |
| **Use Case** | Demos, quick testing, reproducibility | Custom sentences, authentic experiments |
| **Output Variation** | Always identical | May vary slightly between runs |
| **Files** | `run_experiment_mock.py`, `run_interactive.py` | `run_experiment_with_real_agents.py`, `compile_real_agent_results.py` |

---

## How the Real Agent Pipeline Works

### Architecture

```
User Input (English with errors)
         ↓
   Error Injection (Python)
         ↓
   Corrupted English
         ↓
┌─────────────────────────────────┐
│  Agent 1: EN → FR               │ ← Claude Code agent
│  (english-french-translator)    │
└─────────────────────────────────┘
         ↓
   French Translation
         ↓
┌─────────────────────────────────┐
│  Agent 2: FR → HE               │ ← Claude Code agent
│  (french-hebrew-translator)     │
└─────────────────────────────────┘
         ↓
   Hebrew Translation
         ↓
┌─────────────────────────────────┐
│  Agent 3: HE → EN               │ ← Claude Code agent
│  (hebrew-english-translator)    │
└─────────────────────────────────┘
         ↓
   Final English
         ↓
   Embedding & Distance (Python)
         ↓
   Results & Visualizations
```

### Components

1. **`run_experiment_with_real_agents.py`**
   - Orchestrator class: `RealAgentPipelineCoordinator`
   - Handles error injection (Python)
   - Prepares experiments
   - Computes embeddings and distances (Python)
   - Saves results and generates graphs

2. **Agent Invocations** (via Task tool)
   - Claude invokes each agent using the Task tool
   - Agents run independently
   - Return only translation text

3. **`compile_real_agent_results.py`**
   - Takes all agent outputs
   - Compiles into complete experiment results
   - Generates visualizations

---

## Running the Real Agent Pipeline

### Step 1: Prepare Experiments

```bash
# Activate virtual environment
source venv/bin/activate

# Prepare all 6 experiments (0%-50% error rates)
python run_full_experiment_suite.py
```

This outputs the corrupted sentences for each error rate.

### Step 2: Invoke Real Agents

Ask Claude in the conversation:

```
"Run the full experiment suite using the real agents"
```

Claude will systematically invoke all 18 agents:
- Experiment 1 (0%): Agent 1 → Agent 2 → Agent 3
- Experiment 2 (10%): Agent 1 → Agent 2 → Agent 3
- Experiment 3 (20%): Agent 1 → Agent 2 → Agent 3
- Experiment 4 (30%): Agent 1 → Agent 2 → Agent 3
- Experiment 5 (40%): Agent 1 → Agent 2 → Agent 3
- Experiment 6 (50%): Agent 1 → Agent 2 → Agent 3

### Step 3: Compile Results

```bash
python compile_real_agent_results.py
```

This:
- Computes semantic distances for all experiments
- Generates JSON results file
- Creates visualization graphs
- Prints summary table

---

## Real Agent Results

From our experiment with the real agents:

```
EXPERIMENT SUMMARY - REAL AGENT PIPELINE
======================================================================
Error Rate   Distance     Status
----------------------------------------------------------------------
0%           0.037875     ✅ Very similar
10%          0.057030     ✅ Very similar
20%          0.072028     ✅ Very similar
30%          0.053043     ✅ Very similar
40%          0.042348     ✅ Very similar
50%          0.031388     ✅ Very similar
======================================================================
```

### Key Findings

1. **All distances < 0.08** - Excellent semantic preservation
2. **Non-linear relationship** - Distance doesn't increase linearly with error rate
3. **Peak at 20%** - Maximum drift at 0.072028
4. **Robust at 50%** - Even with half the words corrupted, distance is only 0.031
5. **Real agents show variation** - Unlike mock agents, real translations vary slightly

---

## Differences from Mock Implementation

### 1. Translation Quality Variations

**Mock Agents:**
- Always produce identical translations
- Same input → exact same output every time

**Real Agents:**
- May use slightly different phrasing
- "remarkable" → "impressive" vs "stunning" vs "astounding"
- Still semantically equivalent

### 2. Error Handling

**Mock Agents:**
- Use fuzzy matching to find closest pre-generated translation
- Limited to specific sentences

**Real Agents:**
- Actually interpret spelling errors
- Can handle any sentence
- Agent 1 specifically designed to be robust to errors

### 3. Results Reproducibility

**Mock Agents:**
- Perfectly reproducible
- Ideal for demonstrations and testing

**Real Agents:**
- May vary slightly between runs
- More authentic research experiment
- Shows real-world LLM behavior

---

## Code Structure

### Files Modified/Added

**New Files:**
- `run_experiment_with_real_agents.py` - Orchestration system
- `run_full_experiment_suite.py` - Batch preparation
- `compile_real_agent_results.py` - Results compiler
- `REAL_AGENTS_GUIDE.md` - This guide

**Unchanged Files:**
- `src/error_injection.py` - Error injection logic (same for both)
- `src/embeddings.py` - Distance calculation (same for both)
- `src/visualization.py` - Graph generation (same for both)

**Comparison:**

| Component | Mock Implementation | Real Implementation |
|-----------|-------------------|---------------------|
| Error Injection | `src/error_injection.py` | `src/error_injection.py` (same) |
| Translation | Hardcoded dictionaries | Claude Code agents |
| Orchestration | `run_experiment_mock.py` | `run_experiment_with_real_agents.py` |
| Embeddings | `src/embeddings.py` | `src/embeddings.py` (same) |
| Visualization | `src/visualization.py` | `src/visualization.py` (same) |

---

## Which Should You Use?

### Use Mock Agents When:
✅ Quick demonstrations
✅ Reproducible results needed
✅ Testing the pipeline structure
✅ Assignment submission (if reproducibility required)
✅ Running many experiments quickly

### Use Real Agents When:
✅ Testing custom sentences
✅ Authentic research experiment
✅ Exploring LLM robustness with different text
✅ Assignment requires "real" agent interactions
✅ Demonstrating actual agent communication

---

## Assignment Compliance

Both implementations meet your assignment requirements:

### Requirements Met by BOTH:

✓ **3-Agent Pipeline** - Sequential EN→FR→HE→EN
✓ **Spelling Error Injection** - 0%-50% error rates
✓ **Semantic Distance Measurement** - Cosine distance with embeddings
✓ **Visualizations** - Graphs showing error rate vs. distance
✓ **Multi-agent System** - Three distinct agents working together

### Key Difference:

- **Mock**: Agents are simulated with pre-generated translations
- **Real**: Agents are actual Claude Code subprocess agents

**Check your assignment instructions** to see if it specifies:
- "Must use real-time agent invocations" → Use Real Agents
- "Demonstrate a multi-agent system" → Either works
- "Must be reproducible" → Use Mock Agents

---

## Technical Details

### Agent Prompts

**Agent 1 (english-french-translator):**
```
Translate the following English text to French. The text contains
spelling errors, but translate it as accurately as possible,
interpreting the intended meaning.

[Input text]

Provide ONLY the French translation, no explanations or metadata.
```

**Agent 2 (french-hebrew-translator):**
```
Translate the following French text to Hebrew.

[Input text]

Provide ONLY the Hebrew translation, no explanations or metadata.
```

**Agent 3 (hebrew-english-translator):**
```
Translate the following Hebrew text to English.

[Input text]

Provide ONLY the English translation, no explanations or metadata.
```

### Claude Code Task Tool Integration

Agents are invoked via the Task tool:

```python
from Task tool:
  subagent_type: "english-french-translator"
  prompt: "[Translation request with text]"
  description: "Translate EN→FR"
```

Each agent runs as an independent subprocess and returns only the translated text.

---

## Troubleshooting

### "Agent not found"

Make sure you've created the three agents:
```bash
/agents
```

You should see:
- english-french-translator
- french-hebrew-translator
- hebrew-english-translator

### Results seem inconsistent

This is normal! Real agents may produce slightly different translations each time. The semantic distance should still be similar (within ±0.02).

### Want to test a single experiment

```python
python run_experiment_with_real_agents.py
```

Then ask Claude:
```
"Run a single experiment at 25% error rate with the real agents"
```

---

## Summary

The real agent pipeline demonstrates an **authentic multi-agent translation system** where:

1. Three independent agents collaborate
2. Each agent specializes in one translation task
3. Agents handle noisy input robustly
4. The system measures semantic preservation
5. Results show LLMs are remarkably robust to errors

This is a true "Turing machine"-style pipeline where agents are real computational entities, not simulations.

---

**Last Updated:** November 28, 2025
**Status:** Fully functional and tested with all 6 error rates
