"""
Main Entry Point

This script runs the complete tri-lingual agent pipeline experiment.
"""

import os
import argparse
from dotenv import load_dotenv

from .agents import create_agent_pipeline
from .embeddings import get_embedding_model
from .pipeline import run_experiment, run_error_rate_sweep
from .visualization import generate_all_visualizations


# Load environment variables
load_dotenv()


def main():
    """Main function to run the tri-lingual agent pipeline experiment."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run tri-lingual agent pipeline experiment'
    )
    parser.add_argument(
        '--sentence',
        type=str,
        help='Base sentence to test (must have ≥15 words)',
        default=None
    )
    parser.add_argument(
        '--error-rates',
        type=str,
        help='Comma-separated error rates (e.g., "0,10,20,30,40,50")',
        default='0,10,20,30,40,50'
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['anthropic', 'openai'],
        help='LLM provider to use',
        default='anthropic'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Specific model name (optional)',
        default=None
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        help='Sentence transformer model',
        default='all-MiniLM-L6-v2'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results',
        default='results'
    )
    parser.add_argument(
        '--single',
        type=float,
        help='Run single experiment with this error rate (0.0-1.0)',
        default=None
    )

    args = parser.parse_args()

    # Default sentence if not provided
    if args.sentence is None:
        args.sentence = (
            "The remarkable transformation of artificial intelligence systems has "
            "fundamentally changed how researchers approach complex computational "
            "problems in modern scientific investigations."
        )

    # Validate sentence length
    word_count = len(args.sentence.split())
    if word_count < 15:
        print(f"Warning: Sentence has only {word_count} words (minimum 15 required)")
        print("Consider using a longer sentence for better results.")

    print("\n" + "="*70)
    print("TRI-LINGUAL TURING AGENT PIPELINE EXPERIMENT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - LLM Provider: {args.provider}")
    print(f"  - LLM Model: {args.model or 'default'}")
    print(f"  - Embedding Model: {args.embedding_model}")
    print(f"  - Output Directory: {args.output_dir}")
    print(f"  - Base Sentence: {args.sentence}")
    print(f"  - Sentence Length: {word_count} words")

    # Initialize components
    print(f"\n{'='*70}")
    print("INITIALIZING COMPONENTS")
    print(f"{'='*70}")

    print("\n[1/2] Creating translation agents...")
    agent1, agent2, agent3 = create_agent_pipeline(
        provider=args.provider,
        model=args.model
    )
    print(f"  ✓ Agent 1 (EN→FR): {agent1}")
    print(f"  ✓ Agent 2 (FR→HE): {agent2}")
    print(f"  ✓ Agent 3 (HE→EN): {agent3}")

    print("\n[2/2] Loading embedding model...")
    embedding_model = get_embedding_model(args.embedding_model)
    print(f"  ✓ Embedding model loaded")

    # Run experiment
    if args.single is not None:
        # Single experiment mode
        print(f"\n{'='*70}")
        print(f"RUNNING SINGLE EXPERIMENT (Error Rate: {args.single*100:.1f}%)")
        print(f"{'='*70}")

        result = run_experiment(
            sentence=args.sentence,
            error_rate=args.single,
            agent_en_fr=agent1,
            agent_fr_he=agent2,
            agent_he_en=agent3,
            embedding_model=embedding_model
        )

        print(f"\n{'='*70}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*70}")
        print(f"Cosine Distance: {result['cosine_distance']:.4f}")

    else:
        # Error rate sweep mode
        error_rates = [float(x) / 100 for x in args.error_rates.split(',')]

        print(f"\n{'='*70}")
        print(f"RUNNING ERROR RATE SWEEP")
        print(f"Error rates: {[f'{r*100:.0f}%' for r in error_rates]}")
        print(f"{'='*70}")

        results = run_error_rate_sweep(
            base_sentence=args.sentence,
            error_rates=error_rates,
            agent_en_fr=agent1,
            agent_fr_he=agent2,
            agent_he_en=agent3,
            embedding_model=embedding_model,
            output_dir=args.output_dir,
            save_results=True
        )

        # Generate visualizations
        generate_all_visualizations(results, output_dir=args.output_dir)

        print(f"\n{'='*70}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*70}")
        print(f"✓ All results saved to: {args.output_dir}/")
        print(f"✓ Visualizations generated")
        print(f"\nNext steps:")
        print(f"  1. Review the results in {args.output_dir}/")
        print(f"  2. Check the graphs: error_rate_vs_distance.png")
        print(f"  3. Analyze the experiment_results_*.json file")


if __name__ == '__main__':
    main()
