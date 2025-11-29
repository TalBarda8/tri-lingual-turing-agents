#!/usr/bin/env python3
"""
Full Experiment Suite Orchestrator for Real Agent Pipeline

This script prepares ALL experiments (0%-50% error rates) and outputs
instructions for Claude to invoke the real translation agents.

Usage:
  python run_full_experiment_suite.py

This will:
  1. Prepare 6 experiments (one for each error rate)
  2. Save them to /tmp/experiments_queue.json
  3. Display instructions for Claude to process each one
"""

import sys
import os
# Add parent directory to path to allow imports from src and scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from scripts.run_experiment_with_real_agents import RealAgentPipelineCoordinator


def prepare_all_experiments():
    """Prepare all experiments for the full suite."""

    base_sentence = "The remarkable transformation of artificial intelligence systems has fundamentally changed how researchers approach complex computational problems in modern scientific investigations"

    error_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

    coordinator = RealAgentPipelineCoordinator(base_sentence, error_rates)

    prepared_experiments = []

    print("PREPARING FULL EXPERIMENT SUITE")
    print("="*70)

    for error_rate in error_rates:
        experiment = coordinator.prepare_experiment(error_rate, seed=42)
        prepared_experiments.append(experiment)

        print(f"\n✓ Prepared experiment at {error_rate*100:.0f}% error rate")
        print(f"  Corrupted: {experiment['corrupted'][:60]}...")

    # Save all experiments
    with open('/tmp/experiments_queue.json', 'w', encoding='utf-8') as f:
        json.dump(prepared_experiments, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print(f"✅ All {len(prepared_experiments)} experiments prepared")
    print("✅ Saved to /tmp/experiments_queue.json")
    print("="*70)

    return prepared_experiments


def print_claude_instructions(experiments):
    """Print instructions for Claude to process the experiments."""

    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          INSTRUCTIONS FOR CLAUDE TO RUN FULL SUITE                   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("Claude, please process these experiments using the real agents:")
    print()

    for i, exp in enumerate(experiments, 1):
        error_rate = exp['error_rate']
        corrupted_text = exp['ready_for_translation']

        print(f"─────────────────────────────────────────────────────────────────────")
        print(f"EXPERIMENT {i}/6 - Error Rate: {error_rate*100:.0f}%")
        print(f"─────────────────────────────────────────────────────────────────────")
        print(f"Input Text: {corrupted_text}")
        print()
        print(f"Step 1: Invoke english-french-translator agent with the above text")
        print(f"Step 2: Invoke french-hebrew-translator agent with the French output")
        print(f"Step 3: Invoke hebrew-english-translator agent with the Hebrew output")
        print()

    print("═══════════════════════════════════════════════════════════════════════")
    print()
    print("After all agents complete, run:")
    print("  python compile_real_agent_results.py")
    print()


if __name__ == '__main__':
    experiments = prepare_all_experiments()
    print_claude_instructions(experiments)

    print("\n📋 Next Step:")
    print("   Ask Claude: 'Run the full experiment suite using the real agents'")
    print("   Claude will invoke all 18 agent calls (3 agents × 6 error rates)")
    print()
