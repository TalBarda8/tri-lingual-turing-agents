#!/usr/bin/env python3
"""
Compile Results from Real Agent Pipeline

This script takes all the agent outputs and compiles them into a complete
experiment results file with semantic distance calculations.
"""

import sys
import os
# Add parent directory to path to allow imports from src and scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from scripts.run_experiment_with_real_agents import RealAgentPipelineCoordinator

# Define all the real agent outputs from the 6 experiments
experiments_data = [
    {
        "error_rate": 0.0,
        "corrupted": "The remarkable transformation of artificial intelligence systems has fundamentally changed how researchers approach complex computational problems in modern scientific investigations",
        "french": "La transformation remarquable des systèmes d'intelligence artificielle a fondamentalement changé la façon dont les chercheurs abordent les problèmes informatiques complexes dans les investigations scientifiques modernes",
        "hebrew": "השינוי המרשים של מערכות הבינה המלאכותית שינה באופן יסודי את הדרך שבה חוקרים ניגשים לבעיות מחשוביות מורכבות בחקירות מדעיות מודרניות",
        "final_english": "The impressive change of artificial intelligence systems changed in a fundamental way the way in which researchers approach complex computational problems in modern scientific investigations"
    },
    {
        "error_rate": 0.10,
        "corrupted": "The reamrkable transformation of artificial intlligence systems has fundamentally changed how researchers approach complex computational problems in modern scientific investigations",
        "french": "La transformation remarquable des systèmes d'intelligence artificielle a fondamentalement changé la façon dont les chercheurs abordent les problèmes informatiques complexes dans les enquêtes scientifiques modernes",
        "hebrew": "השינוי המדהים של מערכות הבינה המלאכותית שינה באופן יסודי את הדרך שבה חוקרים ניגשים לבעיות חישוביות מורכבות בחקירות המדעיות המודרניות",
        "final_english": "The astounding change of artificial intelligence systems changed in a fundamental manner the way in which researchers approach complex computational problems in modern scientific investigations"
    },
    {
        "error_rate": 0.20,
        "corrupted": "The rmearkable transformation of artificial intleligence aystems has fundamentally changed how researchers approach complex computatipnal problems in modern scientific investigations",
        "french": "La transformation remarquable des systèmes d'intelligence artificielle a fondamentalement changé la façon dont les chercheurs abordent les problèmes informatiques complexes dans les recherches scientifiques modernes",
        "hebrew": "השינוי המדהים של מערכות הבינה המלאכותית שינה באופן מהותי את הדרך שבה חוקרים ניגשים לבעיות חישוביות מורכבות במחקרים מדעיים מודרניים",
        "final_english": "The astounding change of artificial intelligence systems changed in a fundamental way the way in which researchers approach complex computational problems in modern scientific research"
    },
    {
        "error_rate": 0.30,
        "corrupted": "The remarkabke transformation of artificial inetlligence aystems has fundamentally changed how researchers approach complex ccomputational problems in modern scientific investigtaions",
        "french": "La transformation remarquable des systèmes d'intelligence artificielle a fondamentalement changé la façon dont les chercheurs abordent les problèmes informatiques complexes dans les investigations scientifiques modernes",
        "hebrew": "השינוי המדהים של מערכות הבינה המלאכותית שינה באופן יסודי את הדרך שבה חוקרים ניגשים לבעיות חישוביות מורכבות בחקירות מדעיות מודרניות",
        "final_english": "The astounding change of artificial intelligence systems changed in a fundamental way the way in which researchers approach complex computational problems in modern scientific investigations"
    },
    {
        "error_rate": 0.40,
        "corrupted": "The rremarkable transformation of artificiall intelligenxe systmes has fundamentally changed how researchers approach complex cimputational problems in moodern scientific investigstions",
        "french": "La transformation remarquable des systèmes d'intelligence artificielle a fondamentalement changé la façon dont les chercheurs abordent les problèmes informatiques complexes dans les investigations scientifiques modernes",
        "hebrew": "השינוי המדהים של מערכות הבינה המלאכותית שינה באופן יסודי את הדרך שבה חוקרים ניגשים לבעיות חישוביות מורכבות בחקירות מדעיות מודרניות",
        "final_english": "The amazing change of artificial intelligence systems changed in a fundamental way the way in which researchers approach complex computational problems in modern scientific investigations"
    },
    {
        "error_rate": 0.50,
        "corrupted": "The eemarkable transformatoin of rtificial intelligemce systwms has fundamentally changed how researchers approach complex comuptational problems in moderrn scientific inveestigations",
        "french": "La transformation remarquable des systèmes d'intelligence artificielle a fondamentalement changé la façon dont les chercheurs abordent les problèmes informatiques complexes dans les investigations scientifiques modernes",
        "hebrew": "השינוי המדהים של מערכות הבינה המלאכותית שינה באופן יסודי את הדרך שבה חוקרים ניגשים לבעיות חישוביות מורכבות בחקירות מדעיות מודרניות",
        "final_english": "The stunning change of artificial intelligence systems changed fundamentally the way in which researchers approach complex computational problems in modern scientific investigations"
    }
]

base_sentence = "The remarkable transformation of artificial intelligence systems has fundamentally changed how researchers approach complex computational problems in modern scientific investigations"

# Initialize coordinator
coordinator = RealAgentPipelineCoordinator(base_sentence)

print("COMPILING REAL AGENT RESULTS")
print("="*70)

# Process each experiment
for exp_data in experiments_data:
    # Create experiment with corrupted text
    experiment = {
        'error_rate': exp_data['error_rate'],
        'original': base_sentence,
        'corrupted': exp_data['corrupted'],
        'corrupted_words': [],  # Not tracking individual words here
        'ready_for_translation': exp_data['corrupted'],
        'status': 'prepared'
    }

    # Add translations
    experiment = coordinator.add_french_translation(experiment, exp_data['french'])
    experiment = coordinator.add_hebrew_translation(experiment, exp_data['hebrew'])
    experiment = coordinator.add_final_translation(experiment, exp_data['final_english'])

    # Add error_rate_percent for visualization compatibility
    experiment['error_rate_percent'] = experiment['error_rate'] * 100

    # Add to coordinator
    coordinator.experiments.append(experiment)

    print(f"✓ Processed experiment at {exp_data['error_rate']*100:.0f}% error rate")
    print(f"  Distance: {experiment['cosine_distance']:.6f}")

print("="*70)

# Print summary
coordinator.print_summary()

# Save results
results_file = coordinator.save_results()

# Generate visualizations
print("\nGenerating visualizations...")
coordinator.generate_visualizations()

print("\n" + "="*70)
print("✅ REAL AGENT PIPELINE COMPLETE!")
print("="*70)
print(f"📁 Results saved to: {results_file}")
print(f"📊 Graphs saved to: results/")
print()
print("Key Files:")
print("  - results/experiment_results_real_agents_*.json")
print("  - results/error_rate_vs_distance.png")
print("  - results/experiment_summary.png")
