"""
Parallel Embedding Processing Module

Building Block for parallel embedding calculations using multiprocessing.
This module optimizes CPU-bound embedding operations by distributing work
across multiple processes.

Input Data:
- texts: List[str] - List of texts to embed

Output Data:
- embeddings: List[np.ndarray] - List of embedding vectors

Setup Data:
- model_name: str - Embedding model name (default: 'all-MiniLM-L6-v2')
- n_processes: int - Number of parallel processes (default: CPU count)
"""

import multiprocessing as mp
from typing import List, Tuple
from functools import partial
import numpy as np

from .distance import EmbeddingModel, calculate_distance


class ParallelEmbeddingProcessor:
    """
    Building block for parallel embedding processing.

    This class provides multiprocessing capabilities for CPU-intensive
    embedding calculations, significantly improving performance when
    processing large batches of text.

    Input Data:
    - texts: List[str] - List of texts to embed

    Output Data:
    - embeddings: List[np.ndarray] - List of embedding vectors
    - distances: List[float] - List of cosine distances (when comparing pairs)

    Setup Data:
    - model_name: str - Embedding model name
    - n_processes: int - Number of parallel processes
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_processes: int = None):
        """
        Initialize parallel embedding processor.

        Args:
            model_name: Name of the sentence-transformers model
            n_processes: Number of parallel processes (None = use CPU count)

        Raises:
            ValueError: If n_processes is invalid
        """
        # Setup Data
        self.model_name = model_name
        self.n_processes = n_processes or mp.cpu_count()

        # Validate configuration
        self._validate_config()

        # Initialize embedding model (will be used in main process)
        print(f"Initializing ParallelEmbeddingProcessor with {self.n_processes} processes...")
        self.model = EmbeddingModel(model_name)

    def _validate_config(self):
        """
        Validate setup configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.n_processes, int) or self.n_processes <= 0:
            raise ValueError(f"n_processes must be a positive integer, got {self.n_processes}")

        max_processes = mp.cpu_count()
        if self.n_processes > max_processes:
            raise ValueError(
                f"n_processes ({self.n_processes}) exceeds available "
                f"CPU count ({max_processes})"
            )

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")

    def process_batch(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        """
        Process batch of texts in parallel using multiprocessing.

        This method is optimized for CPU-bound embedding calculations.
        It distributes the work across multiple processes for better
        performance on multi-core systems.

        Input Data:
        - texts: List of texts to embed

        Output Data:
        - embeddings: List of embedding vectors (numpy arrays)

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors (numpy arrays)

        Raises:
            TypeError: If texts is not a list or contains non-strings
            ValueError: If texts is empty
        """
        # Input validation
        self._validate_texts_input(texts)

        # For small batches, sequential processing may be faster due to overhead
        if len(texts) < self.n_processes * 2:
            print(f"Small batch ({len(texts)} texts), using sequential processing...")
            return [self.model.encode(text) for text in texts]

        # Use multiprocessing for larger batches
        print(f"Processing {len(texts)} texts in parallel using {self.n_processes} processes...")

        # Create worker function
        def encode_text(text: str, model_name: str) -> np.ndarray:
            """Worker function to encode a single text."""
            # Each process creates its own model instance
            model = EmbeddingModel(model_name)
            return model.encode(text)

        # Use multiprocessing pool with context manager for automatic cleanup
        with mp.Pool(processes=self.n_processes) as pool:
            # Partial function with model_name bound
            worker = partial(encode_text, model_name=self.model_name)

            # Map work to processes
            if show_progress:
                from tqdm import tqdm
                embeddings = list(tqdm(
                    pool.imap(worker, texts),
                    total=len(texts),
                    desc="Embedding"
                ))
            else:
                embeddings = pool.map(worker, texts)

        return embeddings

    def calculate_distances_parallel(
        self,
        original_texts: List[str],
        final_texts: List[str],
        show_progress: bool = True
    ) -> List[float]:
        """
        Calculate semantic distances between text pairs in parallel.

        This method optimizes the computation of cosine distances by:
        1. Computing all embeddings in parallel (CPU-bound)
        2. Calculating distances efficiently in batch

        Input Data:
        - original_texts: List of original texts
        - final_texts: List of final/translated texts

        Output Data:
        - distances: List of cosine distances

        Args:
            original_texts: List of original texts
            final_texts: List of final texts
            show_progress: Whether to show progress bar

        Returns:
            List of cosine distances

        Raises:
            ValueError: If text lists have different lengths
            TypeError: If inputs are not lists of strings
        """
        # Input validation
        self._validate_texts_input(original_texts, "original_texts")
        self._validate_texts_input(final_texts, "final_texts")

        if len(original_texts) != len(final_texts):
            raise ValueError(
                f"original_texts ({len(original_texts)}) and final_texts "
                f"({len(final_texts)}) must have the same length"
            )

        # Get embeddings for all texts in parallel
        all_texts = original_texts + final_texts
        all_embeddings = self.process_batch(all_texts, show_progress=show_progress)

        # Split embeddings back into original and final
        n = len(original_texts)
        orig_embeddings = all_embeddings[:n]
        final_embeddings = all_embeddings[n:]

        # Calculate distances (this is fast, no need for parallelization)
        distances = [
            calculate_distance(orig, final, metric="cosine")
            for orig, final in zip(orig_embeddings, final_embeddings)
        ]

        return distances

    def _validate_texts_input(self, texts: List[str], param_name: str = "texts"):
        """
        Validate text input comprehensively.

        Args:
            texts: List of texts to validate
            param_name: Parameter name for error messages

        Raises:
            TypeError: If input type is invalid
            ValueError: If input value is invalid
        """
        # Type checking
        if not isinstance(texts, list):
            raise TypeError(f"{param_name} must be a list, got {type(texts).__name__}")

        # Value checking
        if not texts:
            raise ValueError(f"{param_name} cannot be empty")

        # Element type checking
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"{param_name}[{i}] must be a string, got {type(text).__name__}"
                )

            # Precondition checking
            if not text.strip():
                raise ValueError(f"{param_name}[{i}] cannot be empty or whitespace only")


def benchmark_parallel_vs_sequential(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    n_processes: int = None
) -> dict:
    """
    Benchmark parallel vs sequential embedding processing.

    Args:
        texts: List of texts to process
        model_name: Embedding model name
        n_processes: Number of processes for parallel (None = CPU count)

    Returns:
        Dictionary with benchmark results:
            - sequential_time: Time for sequential processing (seconds)
            - parallel_time: Time for parallel processing (seconds)
            - speedup: Speedup factor (sequential/parallel)
            - num_texts: Number of texts processed
            - num_processes: Number of processes used
    """
    import time

    # Sequential processing
    print("\n=== SEQUENTIAL PROCESSING ===")
    model = EmbeddingModel(model_name)
    start = time.time()
    for text in texts:
        model.encode(text)
    sequential_time = time.time() - start
    print(f"Time: {sequential_time:.2f}s")

    # Parallel processing
    print("\n=== PARALLEL PROCESSING ===")
    processor = ParallelEmbeddingProcessor(model_name, n_processes)
    start = time.time()
    processor.process_batch(texts, show_progress=False)
    parallel_time = time.time() - start
    print(f"Time: {parallel_time:.2f}s")

    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0

    print(f"\n=== RESULTS ===")
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Parallel:   {parallel_time:.2f}s")
    print(f"Speedup:    {speedup:.2f}x")
    print(f"Processes:  {processor.n_processes}")

    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
        "num_texts": len(texts),
        "num_processes": processor.n_processes,
    }
