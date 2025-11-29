#!/usr/bin/env python3
"""Complete the experiment with real agent outputs."""

import sys
import os
# Add parent directory to path to allow imports from src and scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from scripts.run_experiment_with_real_agents import RealAgentPipelineCoordinator

# Load the prepared experiment
with open('/tmp/current_experiment.json', 'r') as f:
    experiment = json.load(f)

# Initialize coordinator
base_sentence = experiment['original']
coordinator = RealAgentPipelineCoordinator(base_sentence)

# Add Agent 1 output (English → French)
french_translation = "La transformation remarquable des systèmes d'intelligence artificielle a fondamentalement changé la façon dont les chercheurs abordent les problèmes informatiques complexes dans les recherches scientifiques modernes"
experiment = coordinator.add_french_translation(experiment, french_translation)

# Add Agent 2 output (French → Hebrew)
hebrew_translation = "השינוי המדהים של מערכות הבינה המלאכותית שינה באופן יסודי את הדרך שבה חוקרים ניגשים לבעיות חישוביות מורכבות במחקרים המדעיים המודרניים"
experiment = coordinator.add_hebrew_translation(experiment, hebrew_translation)

# Add Agent 3 output (Hebrew → English)
final_english = "The astounding change of artificial intelligence systems changed in a fundamental way the way in which researchers approach complex computational problems in modern scientific research"
experiment = coordinator.add_final_translation(experiment, final_english)

# Add to coordinator's experiments list
coordinator.experiments.append(experiment)

print('REAL AGENT PIPELINE - EXPERIMENT COMPLETE')
print('='*70)
print(f'Error Rate: {experiment["error_rate"]*100:.0f}%')
print()
print('ORIGINAL ENGLISH:')
print(f'  {experiment["original"]}')
print()
print('CORRUPTED ENGLISH (Input to Agent 1):')
print(f'  {experiment["corrupted"]}')
print()
print('FRENCH TRANSLATION (Agent 1 Output):')
print(f'  {experiment["french"]}')
print()
print('HEBREW TRANSLATION (Agent 2 Output):')
print(f'  {experiment["hebrew"]}')
print()
print('FINAL ENGLISH (Agent 3 Output):')
print(f'  {experiment["final_english"]}')
print()
print(f'COSINE DISTANCE: {experiment["cosine_distance"]:.6f}')
print()

# Interpret distance
if experiment['cosine_distance'] < 0.1:
    interpretation = '✅ Very similar - Minimal semantic drift'
elif experiment['cosine_distance'] < 0.3:
    interpretation = '✓ Similar - Low semantic drift'
else:
    interpretation = '⚠ Moderate drift'

print(f'INTERPRETATION: {interpretation}')
print('='*70)

# Save to file
with open('/tmp/completed_experiment.json', 'w', encoding='utf-8') as f:
    json.dump(experiment, f, indent=2, ensure_ascii=False)

print()
print('✅ Experiment saved to /tmp/completed_experiment.json')
