"""
Unit tests for agents parallel processing module.

Tests the ParallelAgentOrchestrator and ThreadSafeCounter with mocked threading
to avoid actual thread execution in tests.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from queue import Queue
import threading

from tri_lingual_agents.agents.parallel import (
    ParallelAgentOrchestrator,
    ThreadSafeCounter,
    benchmark_parallel_vs_sequential_agents,
)


class TestParallelAgentOrchestratorValidation:
    """Test suite for ParallelAgentOrchestrator input validation."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        orchestrator = ParallelAgentOrchestrator()

        assert orchestrator.max_threads == 3
        assert orchestrator.timeout == 300
        assert isinstance(orchestrator.results_queue, Queue)
        assert isinstance(orchestrator.lock, type(threading.Lock()))
        assert isinstance(orchestrator.semaphore, type(threading.Semaphore(1)))

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        orchestrator = ParallelAgentOrchestrator(max_threads=5, timeout=600)

        assert orchestrator.max_threads == 5
        assert orchestrator.timeout == 600

    def test_init_invalid_max_threads_type(self):
        """Test that non-integer max_threads raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            ParallelAgentOrchestrator(max_threads="5")

    def test_init_invalid_max_threads_zero(self):
        """Test that zero max_threads raises ValueError."""
        with pytest.raises(ValueError, match="max_threads must be a positive integer"):
            ParallelAgentOrchestrator(max_threads=0)

    def test_init_invalid_max_threads_negative(self):
        """Test that negative max_threads raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            ParallelAgentOrchestrator(max_threads=-1)

    def test_init_max_threads_too_high(self):
        """Test that max_threads > 10 raises ValueError."""
        with pytest.raises(ValueError, match="max_threads .* is too high"):
            ParallelAgentOrchestrator(max_threads=15)

    def test_init_invalid_timeout_type(self):
        """Test that non-integer timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be a positive integer"):
            ParallelAgentOrchestrator(timeout="300")

    def test_init_invalid_timeout_zero(self):
        """Test that zero timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be a positive integer"):
            ParallelAgentOrchestrator(timeout=0)


class TestParallelAgentOrchestratorFunctionality:
    """Test suite for ParallelAgentOrchestrator processing methods."""

    @pytest.fixture
    def orchestrator(self):
        """Fixture providing an initialized orchestrator."""
        return ParallelAgentOrchestrator(max_threads=2, timeout=60)

    @pytest.fixture
    def mock_agents(self):
        """Fixture providing mock translation agents."""
        agent_en_fr = Mock()
        agent_en_fr.translate.return_value = "Bonjour"

        agent_fr_he = Mock()
        agent_fr_he.translate.return_value = "שלום"

        agent_he_en = Mock()
        agent_he_en.translate.return_value = "Hello"

        return agent_en_fr, agent_fr_he, agent_he_en

    def test_validate_experiments_input_valid(self, orchestrator):
        """Test validation with valid experiments input."""
        experiments = [
            {'text': 'Hello world'},
            {'corrupted_sentence': 'Helo wrld'}
        ]

        # Should not raise
        orchestrator._validate_experiments_input(experiments)

    def test_validate_experiments_input_invalid_type(self, orchestrator):
        """Test that non-list experiments raises TypeError."""
        with pytest.raises(TypeError, match="experiments must be a list"):
            orchestrator._validate_experiments_input("not a list")

    def test_validate_experiments_input_empty(self, orchestrator):
        """Test that empty experiments list raises ValueError."""
        with pytest.raises(ValueError, match="experiments cannot be empty"):
            orchestrator._validate_experiments_input([])

    def test_validate_experiments_input_invalid_element_type(self, orchestrator):
        """Test that non-dict elements raise TypeError."""
        with pytest.raises(TypeError, match="experiments\\[0\\] must be a dict"):
            orchestrator._validate_experiments_input(["not a dict"])

    def test_validate_experiments_input_missing_text_field(self, orchestrator):
        """Test that experiments without text field raise ValueError."""
        experiments = [{'error_rate': 0.1}]  # Missing 'text' and 'corrupted_sentence'

        with pytest.raises(ValueError, match="must have 'text' or 'corrupted_sentence' field"):
            orchestrator._validate_experiments_input(experiments)

    @patch('tri_lingual_agents.agents.parallel.threading.Thread')
    @patch('tri_lingual_agents.agents.parallel.time.sleep')
    def test_process_experiments_parallel_creates_threads(
        self, mock_sleep, mock_thread_class, orchestrator, mock_agents
    ):
        """Test that threads are created for each experiment."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        experiments = [
            {'text': 'First experiment'},
            {'text': 'Second experiment'}
        ]

        # Mock thread instances
        mock_thread1 = Mock()
        mock_thread2 = Mock()
        mock_thread_class.side_effect = [mock_thread1, mock_thread2]

        # Mock the _process_single_experiment to immediately succeed
        with patch.object(orchestrator, '_process_single_experiment'):
            # Put mock results in queue
            orchestrator.results_queue.put({'_index': 0, 'status': 'completed'})
            orchestrator.results_queue.put({'_index': 1, 'status': 'completed'})

            results = orchestrator.process_experiments_parallel(
                experiments, agent_en_fr, agent_fr_he, agent_he_en
            )

        # Verify threads were created
        assert mock_thread_class.call_count == 2
        mock_thread1.start.assert_called_once()
        mock_thread2.start.assert_called_once()
        mock_thread1.join.assert_called_once()
        mock_thread2.join.assert_called_once()

    def test_process_single_experiment_success(self, orchestrator, mock_agents):
        """Test successful processing of single experiment."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        experiment = {'text': 'Hello world', 'error_rate': 0.0}

        orchestrator._process_single_experiment(
            0, experiment, agent_en_fr, agent_fr_he, agent_he_en
        )

        # Verify agents were called in sequence
        agent_en_fr.translate.assert_called_once_with('Hello world', handle_errors=True)
        agent_fr_he.translate.assert_called_once_with('Bonjour')
        agent_he_en.translate.assert_called_once_with('שלום')

        # Verify result was queued
        assert not orchestrator.results_queue.empty()
        result = orchestrator.results_queue.get()
        assert result['status'] == 'completed'
        assert result['french_translation'] == 'Bonjour'
        assert result['hebrew_translation'] == 'שלום'
        assert result['final_english'] == 'Hello'

    def test_process_single_experiment_with_corrupted_sentence(self, orchestrator, mock_agents):
        """Test that corrupted_sentence is used if present."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        experiment = {
            'text': 'Hello world',
            'corrupted_sentence': 'Helo wrld'
        }

        orchestrator._process_single_experiment(
            0, experiment, agent_en_fr, agent_fr_he, agent_he_en
        )

        # Verify first agent was called with corrupted_sentence
        agent_en_fr.translate.assert_called_once_with('Helo wrld', handle_errors=True)

    def test_process_single_experiment_handles_error(self, orchestrator, mock_agents):
        """Test error handling in single experiment processing."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        experiment = {'text': 'Hello world'}

        # Make first agent raise an error
        agent_en_fr.translate.side_effect = ValueError("API error")

        orchestrator._process_single_experiment(
            0, experiment, agent_en_fr, agent_fr_he, agent_he_en
        )

        # Verify error was recorded in result
        result = orchestrator.results_queue.get()
        assert result['status'] == 'failed'
        assert result['error'] == 'API error'
        assert result['error_type'] == 'ValueError'

    def test_process_single_experiment_missing_text_field(self, orchestrator, mock_agents):
        """Test error when experiment has no text field."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        experiment = {'error_rate': 0.1}  # No text field

        orchestrator._process_single_experiment(
            0, experiment, agent_en_fr, agent_fr_he, agent_he_en
        )

        # Verify error was recorded
        result = orchestrator.results_queue.get()
        assert result['status'] == 'failed'
        assert 'must have' in result['error']

    @patch('tri_lingual_agents.agents.parallel.threading.Thread')
    def test_process_experiments_parallel_results_sorted_by_index(
        self, mock_thread_class, orchestrator, mock_agents
    ):
        """Test that results are sorted by index."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        experiments = [
            {'text': 'First'},
            {'text': 'Second'},
            {'text': 'Third'}
        ]

        # Mock threads
        mock_threads = [Mock() for _ in range(3)]
        mock_thread_class.side_effect = mock_threads

        # Mock processing - add results out of order
        with patch.object(orchestrator, '_process_single_experiment'):
            orchestrator.results_queue.put({'_index': 2, 'text': 'Third', 'status': 'completed'})
            orchestrator.results_queue.put({'_index': 0, 'text': 'First', 'status': 'completed'})
            orchestrator.results_queue.put({'_index': 1, 'text': 'Second', 'status': 'completed'})

            results = orchestrator.process_experiments_parallel(
                experiments, agent_en_fr, agent_fr_he, agent_he_en
            )

        # Verify results are sorted by index
        assert results[0]['_index'] == 0
        assert results[1]['_index'] == 1
        assert results[2]['_index'] == 2


class TestThreadSafeCounter:
    """Test suite for ThreadSafeCounter."""

    def test_init_default(self):
        """Test initialization with default value."""
        counter = ThreadSafeCounter()
        assert counter.get() == 0

    def test_init_custom(self):
        """Test initialization with custom value."""
        counter = ThreadSafeCounter(initial=10)
        assert counter.get() == 10

    def test_increment(self):
        """Test increment operation."""
        counter = ThreadSafeCounter(0)

        result = counter.increment()
        assert result == 1
        assert counter.get() == 1

    def test_increment_multiple_times(self):
        """Test multiple increments."""
        counter = ThreadSafeCounter(0)

        counter.increment()
        counter.increment()
        counter.increment()

        assert counter.get() == 3

    def test_decrement(self):
        """Test decrement operation."""
        counter = ThreadSafeCounter(10)

        result = counter.decrement()
        assert result == 9
        assert counter.get() == 9

    def test_decrement_multiple_times(self):
        """Test multiple decrements."""
        counter = ThreadSafeCounter(10)

        counter.decrement()
        counter.decrement()
        counter.decrement()

        assert counter.get() == 7

    def test_increment_and_decrement(self):
        """Test combination of increment and decrement."""
        counter = ThreadSafeCounter(5)

        counter.increment()
        counter.increment()
        counter.decrement()

        assert counter.get() == 6

    def test_get_does_not_modify(self):
        """Test that get() doesn't modify the counter."""
        counter = ThreadSafeCounter(5)

        value1 = counter.get()
        value2 = counter.get()

        assert value1 == 5
        assert value2 == 5


class TestBenchmarkParallelVsSequential:
    """Test suite for benchmark_parallel_vs_sequential_agents function."""

    @pytest.fixture
    def mock_agents(self):
        """Fixture providing mock translation agents."""
        agent_en_fr = Mock()
        agent_en_fr.translate.return_value = "Bonjour"

        agent_fr_he = Mock()
        agent_fr_he.translate.return_value = "שלום"

        agent_he_en = Mock()
        agent_he_en.translate.return_value = "Hello"

        return agent_en_fr, agent_fr_he, agent_he_en

    @patch('tri_lingual_agents.agents.parallel.time.time')
    @patch('tri_lingual_agents.agents.parallel.ParallelAgentOrchestrator')
    def test_benchmark_returns_correct_structure(
        self, mock_orchestrator_class, mock_time, mock_agents
    ):
        """Test that benchmark returns correct result structure."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        experiments = [{'text': 'Test'}]

        # Mock time to simulate duration
        mock_time.side_effect = [0, 10, 10, 15]  # sequential: 10s, parallel: 5s

        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.process_experiments_parallel.return_value = []
        mock_orchestrator_class.return_value = mock_orchestrator

        result = benchmark_parallel_vs_sequential_agents(
            experiments, agent_en_fr, agent_fr_he, agent_he_en, max_threads=3
        )

        # Verify result structure
        assert 'sequential_time' in result
        assert 'parallel_time' in result
        assert 'speedup' in result
        assert 'num_experiments' in result
        assert 'max_threads' in result

    @patch('tri_lingual_agents.agents.parallel.time.time')
    @patch('tri_lingual_agents.agents.parallel.ParallelAgentOrchestrator')
    def test_benchmark_calculates_speedup(
        self, mock_orchestrator_class, mock_time, mock_agents
    ):
        """Test that speedup is calculated correctly."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        experiments = [{'text': 'Test'}]

        # Mock time: sequential = 10s, parallel = 2s
        mock_time.side_effect = [0, 10, 10, 12]

        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.process_experiments_parallel.return_value = []
        mock_orchestrator_class.return_value = mock_orchestrator

        result = benchmark_parallel_vs_sequential_agents(
            experiments, agent_en_fr, agent_fr_he, agent_he_en
        )

        # Speedup should be 10/2 = 5.0
        assert result['speedup'] == pytest.approx(5.0)

    @patch('tri_lingual_agents.agents.parallel.ParallelAgentOrchestrator')
    def test_benchmark_processes_all_experiments_sequentially(
        self, mock_orchestrator_class, mock_agents
    ):
        """Test that all experiments are processed in sequential mode."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        experiments = [
            {'text': 'First'},
            {'text': 'Second'},
            {'text': 'Third'}
        ]

        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.process_experiments_parallel.return_value = []
        mock_orchestrator_class.return_value = mock_orchestrator

        benchmark_parallel_vs_sequential_agents(
            experiments, agent_en_fr, agent_fr_he, agent_he_en
        )

        # Verify each experiment was processed sequentially
        assert agent_en_fr.translate.call_count == 3
        assert agent_fr_he.translate.call_count == 3
        assert agent_he_en.translate.call_count == 3
