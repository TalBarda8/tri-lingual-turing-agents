"""
Parallel Agent Orchestration Module

Building Block for parallel agent execution using multithreading.
This module optimizes I/O-bound agent operations (API calls) by running
multiple experiments concurrently.

Input Data:
- experiments: List[Dict] - List of experiment configurations

Output Data:
- results: List[Dict] - List of completed experiment results

Setup Data:
- max_threads: int - Maximum concurrent threads (default: 3)
- timeout: int - Timeout per agent call in seconds (default: 300)
"""

import threading
from typing import List, Dict, Callable, Any
from queue import Queue
import time

from ..agents import TranslationAgent


class ParallelAgentOrchestrator:
    """
    Building block for parallel agent orchestration using multithreading.

    This class provides thread-based parallelization for I/O-bound operations
    like agent API calls. It includes thread safety mechanisms and proper
    resource management.

    Input Data:
    - experiments: List[Dict] - Experiment configurations with 'text' and 'error_rate'

    Output Data:
    - results: List[Dict] - Completed experiments with translations

    Setup Data:
    - max_threads: int - Maximum concurrent threads
    - timeout: int - Timeout per operation in seconds
    """

    def __init__(self, max_threads: int = 3, timeout: int = 300):
        """
        Initialize parallel agent orchestrator.

        Args:
            max_threads: Maximum number of concurrent threads
            timeout: Timeout for agent calls in seconds

        Raises:
            ValueError: If configuration is invalid
        """
        # Setup Data
        self.max_threads = max_threads
        self.timeout = timeout

        # Thread safety mechanisms
        self.results_queue = Queue()  # Thread-safe result storage
        self.lock = threading.Lock()  # For critical sections
        self.semaphore = threading.Semaphore(max_threads)  # Limit concurrent threads

        # Validate configuration
        self._validate_config()

        print(f"Initialized ParallelAgentOrchestrator with max_threads={max_threads}")

    def _validate_config(self):
        """
        Validate setup configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.max_threads, int) or self.max_threads <= 0:
            raise ValueError(f"max_threads must be a positive integer, got {self.max_threads}")

        if self.max_threads > 10:
            raise ValueError(
                f"max_threads ({self.max_threads}) is too high. "
                "Recommended maximum is 10 to avoid overwhelming APIs"
            )

        if not isinstance(self.timeout, int) or self.timeout <= 0:
            raise ValueError(f"timeout must be a positive integer, got {self.timeout}")

    def process_experiments_parallel(
        self,
        experiments: List[Dict],
        agent_en_fr: TranslationAgent,
        agent_fr_he: TranslationAgent,
        agent_he_en: TranslationAgent,
    ) -> List[Dict]:
        """
        Process multiple experiments in parallel using multithreading.

        This method is optimized for I/O-bound operations (agent API calls).
        It runs multiple experiments concurrently while respecting the
        max_threads limit.

        Input Data:
        - experiments: List of dicts with keys: 'text', 'error_rate', etc.

        Output Data:
        - results: List of completed experiment dicts with translations added

        Args:
            experiments: List of experiment configurations
            agent_en_fr: English to French translation agent
            agent_fr_he: French to Hebrew translation agent
            agent_he_en: Hebrew to English translation agent

        Returns:
            List of completed experiments with results

        Raises:
            TypeError: If experiments is not a list
            ValueError: If experiments is empty or invalid
        """
        # Input validation
        self._validate_experiments_input(experiments)

        print(f"\nProcessing {len(experiments)} experiments in parallel "
              f"(max {self.max_threads} concurrent)...")

        threads = []
        start_time = time.time()

        # Create and start threads for each experiment
        for i, experiment in enumerate(experiments):
            thread = threading.Thread(
                target=self._process_single_experiment,
                args=(i, experiment, agent_en_fr, agent_fr_he, agent_he_en),
                daemon=True  # Daemon threads for clean shutdown
            )
            threads.append(thread)
            thread.start()

            # Brief pause to stagger starts
            time.sleep(0.1)

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=self.timeout + 10)  # Extra time for safety

        # Collect results from queue
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())

        # Sort results by index to maintain order
        results.sort(key=lambda x: x.get('_index', 0))

        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r.get('status') == 'completed')
        failed = len(results) - successful

        print(f"\nParallel processing complete!")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Successful: {successful}/{len(experiments)}")
        print(f"  Failed: {failed}/{len(experiments)}")
        print(f"  Avg time/experiment: {elapsed/len(experiments):.2f}s")

        return results

    def _process_single_experiment(
        self,
        index: int,
        experiment: Dict,
        agent_en_fr: TranslationAgent,
        agent_fr_he: TranslationAgent,
        agent_he_en: TranslationAgent,
    ):
        """
        Process a single experiment in a thread.

        This method runs in a separate thread and handles the full
        translation pipeline for one experiment with proper error handling
        and thread safety.

        Args:
            index: Experiment index for ordering results
            experiment: Experiment configuration dict
            agent_en_fr: EN→FR agent
            agent_fr_he: FR→HE agent
            agent_he_en: HE→EN agent
        """
        # Acquire semaphore to limit concurrent executions
        with self.semaphore:
            try:
                print(f"  [Thread-{index}] Starting experiment {index+1}...")

                # Extract text to translate
                text = experiment.get('corrupted_sentence') or experiment.get('text')
                if not text:
                    raise ValueError("Experiment must have 'corrupted_sentence' or 'text' field")

                # Step 1: EN → FR (with error handling)
                print(f"  [Thread-{index}] EN → FR...")
                french = agent_en_fr.translate(text, handle_errors=True)

                # Step 2: FR → HE
                print(f"  [Thread-{index}] FR → HE...")
                hebrew = agent_fr_he.translate(french)

                # Step 3: HE → EN
                print(f"  [Thread-{index}] HE → EN...")
                final_english = agent_he_en.translate(hebrew)

                # Thread-safe result storage
                with self.lock:
                    experiment['_index'] = index
                    experiment['french_translation'] = french
                    experiment['hebrew_translation'] = hebrew
                    experiment['final_english'] = final_english
                    experiment['status'] = 'completed'
                    self.results_queue.put(experiment)

                print(f"  [Thread-{index}] ✓ Completed successfully")

            except Exception as e:
                # Thread-safe error recording
                with self.lock:
                    experiment['_index'] = index
                    experiment['status'] = 'failed'
                    experiment['error'] = str(e)
                    experiment['error_type'] = type(e).__name__
                    self.results_queue.put(experiment)

                print(f"  [Thread-{index}] ✗ Failed: {e}")

    def _validate_experiments_input(self, experiments: List[Dict]):
        """
        Validate experiments input comprehensively.

        Args:
            experiments: List of experiment dicts to validate

        Raises:
            TypeError: If input type is invalid
            ValueError: If input value is invalid
        """
        # Type checking
        if not isinstance(experiments, list):
            raise TypeError(f"experiments must be a list, got {type(experiments).__name__}")

        # Value checking
        if not experiments:
            raise ValueError("experiments cannot be empty")

        # Element validation
        for i, exp in enumerate(experiments):
            if not isinstance(exp, dict):
                raise TypeError(
                    f"experiments[{i}] must be a dict, got {type(exp).__name__}"
                )

            # Check required fields
            has_text = 'text' in exp or 'corrupted_sentence' in exp
            if not has_text:
                raise ValueError(
                    f"experiments[{i}] must have 'text' or 'corrupted_sentence' field"
                )


class ThreadSafeCounter:
    """
    Thread-safe counter for tracking progress across threads.

    This is a simple building block demonstrating thread safety
    using locks.
    """

    def __init__(self, initial: int = 0):
        """
        Initialize counter.

        Args:
            initial: Initial counter value
        """
        self._value = initial
        self._lock = threading.Lock()

    def increment(self) -> int:
        """
        Increment counter by 1 (thread-safe).

        Returns:
            New counter value
        """
        with self._lock:
            self._value += 1
            return self._value

    def decrement(self) -> int:
        """
        Decrement counter by 1 (thread-safe).

        Returns:
            New counter value
        """
        with self._lock:
            self._value -= 1
            return self._value

    def get(self) -> int:
        """
        Get current counter value (thread-safe).

        Returns:
            Current value
        """
        with self._lock:
            return self._value


def benchmark_parallel_vs_sequential_agents(
    experiments: List[Dict],
    agent_en_fr: TranslationAgent,
    agent_fr_he: TranslationAgent,
    agent_he_en: TranslationAgent,
    max_threads: int = 3
) -> dict:
    """
    Benchmark parallel vs sequential agent processing.

    Args:
        experiments: List of experiment configurations
        agent_en_fr: EN→FR agent
        agent_fr_he: FR→HE agent
        agent_he_en: HE→EN agent
        max_threads: Number of threads for parallel processing

    Returns:
        Dictionary with benchmark results
    """
    import time

    # Sequential processing
    print("\n=== SEQUENTIAL PROCESSING ===")
    start = time.time()
    for i, exp in enumerate(experiments):
        text = exp.get('corrupted_sentence') or exp.get('text')
        print(f"  Processing {i+1}/{len(experiments)}...")
        french = agent_en_fr.translate(text, handle_errors=True)
        hebrew = agent_fr_he.translate(french)
        final = agent_he_en.translate(hebrew)
    sequential_time = time.time() - start
    print(f"Total time: {sequential_time:.2f}s")

    # Parallel processing
    print("\n=== PARALLEL PROCESSING ===")
    orchestrator = ParallelAgentOrchestrator(max_threads=max_threads)
    start = time.time()
    results = orchestrator.process_experiments_parallel(
        experiments, agent_en_fr, agent_fr_he, agent_he_en
    )
    parallel_time = time.time() - start

    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0

    print(f"\n=== RESULTS ===")
    print(f"Sequential: {sequential_time:.2f}s ({sequential_time/len(experiments):.2f}s/experiment)")
    print(f"Parallel:   {parallel_time:.2f}s ({parallel_time/len(experiments):.2f}s/experiment)")
    print(f"Speedup:    {speedup:.2f}x")
    print(f"Threads:    {max_threads}")

    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
        "num_experiments": len(experiments),
        "max_threads": max_threads,
    }
