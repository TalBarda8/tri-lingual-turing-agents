"""
Experiment Pipeline Module

This module orchestrates the tri-lingual translation pipeline experiments
and collects data on semantic drift vs. spelling error rates.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path

from .agents import TranslationAgent
from .embeddings import EmbeddingModel, calculate_distance
from .error_injection import inject_spelling_errors, calculate_error_statistics


def run_experiment(
    sentence: str,
    error_rate: float,
    agent_en_fr: TranslationAgent,
    agent_fr_he: TranslationAgent,
    agent_he_en: TranslationAgent,
    embedding_model: EmbeddingModel,
    seed: int = None
) -> Dict:
    """
    Run a single experiment with a given error rate.

    Args:
        sentence: The original (clean) English sentence
        error_rate: Error rate to apply (0.0 to 1.0)
        agent_en_fr: English to French translation agent
        agent_fr_he: French to Hebrew translation agent
        agent_he_en: Hebrew to English translation agent
        embedding_model: Embedding model for distance calculation
        seed: Random seed for error injection

    Returns:
        Dictionary containing all experiment data
    """
    print(f"\n{'='*70}")
    print(f"Running experiment with error rate: {error_rate*100:.1f}%")
    print(f"{'='*70}")

    # Step 1: Inject spelling errors (if error_rate > 0)
    if error_rate > 0:
        corrupted_sentence, corrupted_words = inject_spelling_errors(sentence, error_rate, seed)
        error_stats = calculate_error_statistics(sentence, corrupted_sentence, corrupted_words)
    else:
        corrupted_sentence = sentence
        corrupted_words = []
        error_stats = {
            'total_words': len(sentence.split()),
            'corrupted_words': 0,
            'error_rate_percent': 0.0,
            'corrupted_list': [],
            'original_sentence': sentence,
            'corrupted_sentence': sentence
        }

    print(f"\nOriginal sentence: {sentence}")
    print(f"Corrupted sentence: {corrupted_sentence}")
    if corrupted_words:
        print(f"Corrupted words ({len(corrupted_words)}): {', '.join(corrupted_words)}")

    # Step 2: Pass through Agent 1 (EN → FR)
    print(f"\n[Agent 1] Translating EN → FR...")
    french_output = agent_en_fr.translate(corrupted_sentence, handle_errors=True)
    print(f"French output: {french_output}")

    # Step 3: Pass through Agent 2 (FR → HE)
    print(f"\n[Agent 2] Translating FR → HE...")
    hebrew_output = agent_fr_he.translate(french_output)
    print(f"Hebrew output: {hebrew_output}")

    # Step 4: Pass through Agent 3 (HE → EN)
    print(f"\n[Agent 3] Translating HE → EN...")
    final_english = agent_he_en.translate(hebrew_output)
    print(f"Final English: {final_english}")

    # Step 5: Compute embeddings and distance
    print(f"\n[Embeddings] Computing semantic distance...")
    original_embedding = embedding_model.encode(sentence)
    final_embedding = embedding_model.encode(final_english)

    distance = calculate_distance(original_embedding, final_embedding, metric='cosine')
    print(f"Cosine distance: {distance:.4f}")

    # Compile results
    result = {
        'error_rate': error_rate,
        'error_rate_percent': error_rate * 100,
        'original_sentence': sentence,
        'corrupted_sentence': corrupted_sentence,
        'corrupted_words': corrupted_words,
        'error_statistics': error_stats,
        'french_translation': french_output,
        'hebrew_translation': hebrew_output,
        'final_english': final_english,
        'cosine_distance': distance,
        'timestamp': datetime.now().isoformat()
    }

    return result


def run_error_rate_sweep(
    base_sentence: str,
    error_rates: List[float],
    agent_en_fr: TranslationAgent,
    agent_fr_he: TranslationAgent,
    agent_he_en: TranslationAgent,
    embedding_model: EmbeddingModel,
    output_dir: str = 'results',
    save_results: bool = True
) -> List[Dict]:
    """
    Run experiments across multiple error rates.

    Args:
        base_sentence: The base (clean) sentence to test
        error_rates: List of error rates to test (e.g., [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        agent_en_fr: English to French translation agent
        agent_fr_he: French to Hebrew translation agent
        agent_he_en: Hebrew to English translation agent
        embedding_model: Embedding model
        output_dir: Directory to save results
        save_results: Whether to save results to file

    Returns:
        List of result dictionaries, one per error rate
    """
    print(f"\n{'#'*70}")
    print(f"# ERROR RATE SWEEP")
    print(f"# Base sentence: {base_sentence}")
    print(f"# Error rates: {[f'{r*100:.0f}%' for r in error_rates]}")
    print(f"{'#'*70}\n")

    results = []

    for i, error_rate in enumerate(error_rates):
        # Use index as seed for reproducibility
        result = run_experiment(
            sentence=base_sentence,
            error_rate=error_rate,
            agent_en_fr=agent_en_fr,
            agent_fr_he=agent_fr_he,
            agent_he_en=agent_he_en,
            embedding_model=embedding_model,
            seed=42 + i  # Different seed for each error rate
        )
        results.append(result)

    # Save results if requested
    if save_results:
        save_experiment_results(results, base_sentence, output_dir)

    # Print summary
    print_summary(results)

    return results


def save_experiment_results(results: List[Dict], base_sentence: str, output_dir: str = 'results'):
    """
    Save experiment results to JSON file.

    Args:
        results: List of experiment result dictionaries
        base_sentence: The base sentence used
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"experiment_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Prepare data for JSON serialization
    output_data = {
        'experiment_metadata': {
            'timestamp': datetime.now().isoformat(),
            'base_sentence': base_sentence,
            'num_error_rates': len(results),
            'error_rates': [r['error_rate'] for r in results]
        },
        'results': results
    }

    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {filepath}")


def print_summary(results: List[Dict]):
    """
    Print a summary table of the experiment results.

    Args:
        results: List of experiment result dictionaries
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Error Rate':<12} {'Distance':<12} {'Interpretation'}")
    print(f"{'-'*70}")

    for result in results:
        error_rate = result['error_rate_percent']
        distance = result['cosine_distance']

        # Interpretation
        if distance < 0.1:
            interp = "Very similar"
        elif distance < 0.3:
            interp = "Similar"
        elif distance < 0.5:
            interp = "Moderate drift"
        else:
            interp = "High drift"

        print(f"{error_rate:>10.1f}% {distance:>11.4f}  {interp}")

    print(f"{'='*70}\n")


def load_experiment_results(filepath: str) -> Tuple[Dict, List[Dict]]:
    """
    Load experiment results from JSON file.

    Args:
        filepath: Path to the results JSON file

    Returns:
        Tuple of (metadata, results_list)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data['experiment_metadata'], data['results']
