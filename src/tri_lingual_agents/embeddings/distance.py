"""
Embeddings and Distance Module

This module provides functions to compute sentence embeddings and calculate
semantic distances between sentences.
"""

import numpy as np
from typing import Union, Literal
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine, euclidean


class EmbeddingModel:
    """
    Wrapper class for sentence embedding models.

    Attributes:
        model_name (str): Name of the sentence-transformers model
        model: The loaded SentenceTransformer model
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use
                       Options: 'all-MiniLM-L6-v2' (default), 'all-mpnet-base-v2'
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def encode(self, text: Union[str, list[str]], show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode text into embedding vectors.

        Args:
            text: A single sentence (str) or list of sentences
            show_progress_bar: Whether to show progress bar for batch encoding

        Returns:
            Numpy array of embeddings (1D for single sentence, 2D for multiple)
        """
        return self.model.encode(text, show_progress_bar=show_progress_bar, convert_to_numpy=True)

    def get_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()


def calculate_distance(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    metric: Literal['cosine', 'euclidean'] = 'cosine'
) -> float:
    """
    Calculate distance between two embedding vectors.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        metric: Distance metric to use ('cosine' or 'euclidean')

    Returns:
        Distance value (float)

    Raises:
        ValueError: If embeddings have different dimensions or invalid metric
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError(
            f"Embeddings must have the same shape. "
            f"Got {embedding1.shape} and {embedding2.shape}"
        )

    if metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        # Range: [0, 2] where 0 = identical, 2 = opposite
        return float(cosine(embedding1, embedding2))

    elif metric == 'euclidean':
        # Euclidean distance (L2 norm)
        return float(euclidean(embedding1, embedding2))

    else:
        raise ValueError(f"Invalid metric: {metric}. Use 'cosine' or 'euclidean'")


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embedding vectors.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Similarity value in range [-1, 1] where 1 = identical, -1 = opposite
    """
    # Cosine similarity = 1 - cosine distance
    distance = calculate_distance(embedding1, embedding2, metric='cosine')
    return 1.0 - distance


def compare_sentences(
    sentence1: str,
    sentence2: str,
    model: EmbeddingModel,
    metric: Literal['cosine', 'euclidean'] = 'cosine'
) -> dict:
    """
    Compare two sentences and return detailed similarity metrics.

    Args:
        sentence1: First sentence
        sentence2: Second sentence
        model: EmbeddingModel instance
        metric: Distance metric to use

    Returns:
        Dictionary containing:
            - 'sentence1': First sentence
            - 'sentence2': Second sentence
            - 'embedding1': First embedding vector
            - 'embedding2': Second embedding vector
            - 'distance': Distance value
            - 'similarity': Cosine similarity (always computed)
            - 'metric': Metric used
    """
    # Encode both sentences
    embeddings = model.encode([sentence1, sentence2])
    emb1, emb2 = embeddings[0], embeddings[1]

    # Calculate distance
    distance = calculate_distance(emb1, emb2, metric=metric)

    # Always calculate cosine similarity for reference
    similarity = calculate_similarity(emb1, emb2)

    return {
        'sentence1': sentence1,
        'sentence2': sentence2,
        'embedding1': emb1,
        'embedding2': emb2,
        'distance': distance,
        'similarity': similarity,
        'metric': metric
    }


def interpret_distance(distance: float, metric: str = 'cosine') -> str:
    """
    Provide a human-readable interpretation of a distance value.

    Args:
        distance: The distance value
        metric: The metric used ('cosine' or 'euclidean')

    Returns:
        String interpretation of the distance
    """
    if metric == 'cosine':
        if distance < 0.1:
            return "Very similar (minimal semantic drift)"
        elif distance < 0.3:
            return "Similar (low semantic drift)"
        elif distance < 0.5:
            return "Moderately similar (moderate semantic drift)"
        elif distance < 0.7:
            return "Somewhat different (high semantic drift)"
        else:
            return "Very different (very high semantic drift)"
    else:
        # Euclidean distance interpretation depends on embedding dimension
        return f"Euclidean distance: {distance:.4f}"


# Convenience function to create and cache model
_model_cache = {}


def get_embedding_model(model_name: str = 'all-MiniLM-L6-v2') -> EmbeddingModel:
    """
    Get or create an embedding model (with caching).

    Args:
        model_name: Name of the model to load

    Returns:
        EmbeddingModel instance
    """
    if model_name not in _model_cache:
        _model_cache[model_name] = EmbeddingModel(model_name)
    return _model_cache[model_name]
