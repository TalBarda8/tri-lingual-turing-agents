"""
Embeddings and Semantic Distance Module

This module provides functions to compute sentence embeddings and calculate
semantic distances between sentences.

Includes parallel processing capabilities for performance optimization.
"""

from .distance import (
    EmbeddingModel,
    calculate_distance,
    calculate_similarity,
    compare_sentences,
    interpret_distance,
    get_embedding_model,
)
from .parallel import (
    ParallelEmbeddingProcessor,
    benchmark_parallel_vs_sequential,
)

__all__ = [
    # Core embedding functions
    "EmbeddingModel",
    "calculate_distance",
    "calculate_similarity",
    "compare_sentences",
    "interpret_distance",
    "get_embedding_model",
    # Parallel processing
    "ParallelEmbeddingProcessor",
    "benchmark_parallel_vs_sequential",
]
