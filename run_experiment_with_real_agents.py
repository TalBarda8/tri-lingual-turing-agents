#!/usr/bin/env python3
"""
Real Agent Pipeline Orchestrator

This script coordinates experiments using the three real translation agents:
1. english-french-translator
2. french-hebrew-translator
3. hebrew-english-translator

These agents are Claude Code agents invoked via the Task tool during conversation.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

from src.error_injection import inject_spelling_errors
from src.embeddings import get_embedding_model, calculate_distance
from src.visualization import generate_all_visualizations


class RealAgentPipelineCoordinator:
    """
    Coordinates pipeline execution using real translation agents.

    Unlike mock versions, this coordinator:
    - Does NOT use hardcoded translation dictionaries
    - Expects translations to come from real agents (called externally)
    - Focuses on data preparation, analysis, and visualization
    """

    def __init__(self, base_sentence: str, error_rates: List[float] = None):
        self.base_sentence = base_sentence
        self.error_rates = error_rates or [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
        self.embedding_model = get_embedding_model('all-MiniLM-L6-v2')

        # Storage for experiment results
        self.experiments = []

    def prepare_experiment(self, error_rate: float, seed: int = 42) -> Dict[str, Any]:
        """
        Prepare an experiment by injecting errors.

        Returns a dictionary with:
        - error_rate: The error rate used
        - original: Clean English sentence
        - corrupted: Sentence with spelling errors
        - corrupted_words: List of words that were corrupted
        - ready_for_translation: Text ready for Agent 1 (EN→FR)
        """
        corrupted_sentence, corrupted_words = inject_spelling_errors(
            self.base_sentence,
            error_rate=error_rate,
            seed=seed
        )

        return {
            'error_rate': error_rate,
            'original': self.base_sentence,
            'corrupted': corrupted_sentence,
            'corrupted_words': corrupted_words,
            'ready_for_translation': corrupted_sentence,
            'status': 'prepared'
        }

    def add_french_translation(self, experiment: Dict[str, Any], french_text: str) -> Dict[str, Any]:
        """
        Add French translation from Agent 1 to the experiment.

        Args:
            experiment: The experiment dictionary
            french_text: Translation from english-french-translator agent
        """
        experiment['french'] = french_text.strip()
        experiment['ready_for_agent2'] = french_text.strip()
        experiment['agent1_complete'] = True
        return experiment

    def add_hebrew_translation(self, experiment: Dict[str, Any], hebrew_text: str) -> Dict[str, Any]:
        """
        Add Hebrew translation from Agent 2 to the experiment.

        Args:
            experiment: The experiment dictionary
            hebrew_text: Translation from french-hebrew-translator agent
        """
        experiment['hebrew'] = hebrew_text.strip()
        experiment['ready_for_agent3'] = hebrew_text.strip()
        experiment['agent2_complete'] = True
        return experiment

    def add_final_translation(self, experiment: Dict[str, Any], final_english: str) -> Dict[str, Any]:
        """
        Add final English translation from Agent 3 and compute distance.

        Args:
            experiment: The experiment dictionary
            final_english: Translation from hebrew-english-translator agent
        """
        experiment['final_english'] = final_english.strip()
        experiment['agent3_complete'] = True

        # Compute semantic distance
        orig_embedding = self.embedding_model.encode(self.base_sentence)
        final_embedding = self.embedding_model.encode(final_english.strip())
        distance = calculate_distance(orig_embedding, final_embedding, metric='cosine')

        experiment['cosine_distance'] = float(distance)
        experiment['status'] = 'complete'

        return experiment

    def save_results(self, output_dir: str = 'results'):
        """Save all experiment results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'experiment_results_real_agents_{timestamp}.json'
        filepath = os.path.join(output_dir, filename)

        results = {
            'experiment_metadata': {
                'type': 'real_agent_experiment',
                'translator': 'Claude Code Agents (english-french-translator, french-hebrew-translator, hebrew-english-translator)',
                'timestamp': datetime.now().isoformat(),
                'base_sentence': self.base_sentence,
                'error_rates': self.error_rates,
                'embedding_model': 'all-MiniLM-L6-v2'
            },
            'results': self.experiments
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Results saved to: {filepath}")
        return filepath

    def generate_visualizations(self, output_dir: str = 'results'):
        """Generate graphs from completed experiments."""
        if not self.experiments:
            print("⚠️  No experiments to visualize")
            return

        # Extract data for visualization
        error_rates = [exp['error_rate'] for exp in self.experiments]
        distances = [exp['cosine_distance'] for exp in self.experiments]

        generate_all_visualizations(self.experiments, output_dir=output_dir)
        print(f"✅ Visualizations generated in {output_dir}/")

    def print_summary(self):
        """Print a summary of all experiments."""
        if not self.experiments:
            print("⚠️  No experiments completed yet")
            return

        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY - REAL AGENT PIPELINE")
        print("="*70)
        print(f"Base Sentence: {self.base_sentence}")
        print(f"Total Experiments: {len(self.experiments)}")
        print()
        print(f"{'Error Rate':<12} {'Distance':<12} {'Status':<20}")
        print("-"*70)

        for exp in self.experiments:
            error_pct = f"{exp['error_rate']*100:.0f}%"
            distance = f"{exp['cosine_distance']:.6f}"

            if exp['cosine_distance'] < 0.1:
                status = "✅ Very similar"
            elif exp['cosine_distance'] < 0.3:
                status = "✓ Similar"
            else:
                status = "⚠ Moderate drift"

            print(f"{error_pct:<12} {distance:<12} {status:<20}")

        print("="*70)


def print_instructions():
    """Print instructions for running the real agent pipeline."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         REAL AGENT PIPELINE - ORCHESTRATION INSTRUCTIONS             ║
╚══════════════════════════════════════════════════════════════════════╝

This pipeline uses THREE REAL TRANSLATION AGENTS:
  1. english-french-translator  (EN → FR)
  2. french-hebrew-translator   (FR → HE)
  3. hebrew-english-translator  (HE → EN)

HOW IT WORKS:
  1. Python code: Prepares data (error injection)
  2. Claude invokes: Agent 1 for English → French
  3. Claude invokes: Agent 2 for French → Hebrew
  4. Claude invokes: Agent 3 for Hebrew → English
  5. Python code: Computes embeddings, distances, visualizations

TO RUN AN EXPERIMENT:
  Ask Claude: "Run an experiment with the real agents at 25% error rate"

  Claude will:
  - Prepare the corrupted sentence
  - Invoke english-french-translator agent
  - Invoke french-hebrew-translator agent
  - Invoke hebrew-english-translator agent
  - Compute semantic distance
  - Save results

EXAMPLE WORKFLOW:
  User: "Run the full experiment suite (0%-50%) with real agents"
  Claude will orchestrate all 6 experiments automatically.

═══════════════════════════════════════════════════════════════════════
""")


if __name__ == '__main__':
    print_instructions()

    # Example: Create coordinator
    base_sentence = "The remarkable transformation of artificial intelligence systems has fundamentally changed how researchers approach complex computational problems in modern scientific investigations"

    coordinator = RealAgentPipelineCoordinator(base_sentence)

    print(f"\n📋 Coordinator initialized with base sentence:")
    print(f"   \"{base_sentence[:80]}...\"")
    print(f"\n📊 Ready to run experiments at error rates: {coordinator.error_rates}")
    print(f"\n💡 To run experiments, ask Claude to orchestrate the pipeline using the real agents.")
