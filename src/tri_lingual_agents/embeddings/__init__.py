"""
Embeddings and Semantic Distance Module

This module provides functions to compute sentence embeddings and calculate
semantic distances between sentences.
"""

from .distance import (
    EmbeddingModel,
    calculate_distance,
    calculate_similarity,
    compare_sentences,
    interpret_distance,
    get_embedding_model,
)

__all__ = [
    "EmbeddingModel",
    "calculate_distance",
    "calculate_similarity",
    "compare_sentences",
    "interpret_distance",
    "get_embedding_model",
]
