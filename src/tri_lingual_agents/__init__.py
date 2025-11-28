"""
Tri-Lingual Turing Agents Package

A multi-agent translation system that explores semantic drift through
round-trip translation pipeline (English → French → Hebrew → English).

This package demonstrates LLM robustness to spelling errors in multi-step
translation workflows.
"""

__version__ = "1.0.0"
__author__ = "Tal Barda"
__all__ = [
    # Submodules
    "agents",
    "embeddings",
    "error_injection",
    "pipeline",
    "visualization",
    # Key classes
    "TranslationAgent",
    "create_agent_pipeline",
    "EmbeddingModel",
    "get_embedding_model",
    "calculate_distance",
    "inject_spelling_errors",
    "run_experiment",
    "run_error_rate_sweep",
    "plot_from_results",
    "generate_all_visualizations",
    # Parallel processing
    "ParallelEmbeddingProcessor",
    "ParallelAgentOrchestrator",
]

# Import key components for easy access
from .agents import (
    TranslationAgent,
    create_agent_pipeline,
    ParallelAgentOrchestrator,
)
from .embeddings import (
    EmbeddingModel,
    get_embedding_model,
    calculate_distance,
    calculate_similarity,
    compare_sentences,
    ParallelEmbeddingProcessor,
)
from .error_injection import inject_spelling_errors, calculate_error_statistics
from .pipeline import (
    run_experiment,
    run_error_rate_sweep,
    save_experiment_results,
    load_experiment_results,
)
from .visualization import (
    plot_error_vs_distance,
    plot_from_results,
    create_summary_figure,
    generate_all_visualizations,
)
