"""
Unit tests for embeddings parallel processing module.

Tests the ParallelEmbeddingProcessor with mocked multiprocessing
to avoid actual process execution in tests.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import multiprocessing as mp

from tri_lingual_agents.embeddings.parallel import (
    ParallelEmbeddingProcessor,
    benchmark_parallel_vs_sequential,
)


class TestParallelEmbeddingProcessorValidation:
    """Test suite for ParallelEmbeddingProcessor input validation."""

    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_init_default_params(self, mock_model_class):
        """Test initialization with default parameters."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        processor = ParallelEmbeddingProcessor()

        assert processor.model_name == "all-MiniLM-L6-v2"
        assert processor.n_processes == mp.cpu_count()
        assert processor.model == mock_model

    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_init_custom_params(self, mock_model_class):
        """Test initialization with custom parameters."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        processor = ParallelEmbeddingProcessor(
            model_name="custom-model",
            n_processes=4
        )

        assert processor.model_name == "custom-model"
        assert processor.n_processes == 4

    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_init_invalid_n_processes_type(self, mock_model_class):
        """Test that non-integer n_processes raises ValueError."""
        with pytest.raises(ValueError, match="n_processes must be a positive integer"):
            ParallelEmbeddingProcessor(n_processes="4")

    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_init_invalid_n_processes_negative(self, mock_model_class):
        """Test that negative n_processes raises ValueError."""
        with pytest.raises(ValueError, match="n_processes must be a positive integer"):
            ParallelEmbeddingProcessor(n_processes=-1)

    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_init_n_processes_exceeds_cpu_count(self, mock_model_class):
        """Test that n_processes > CPU count raises ValueError."""
        max_cpus = mp.cpu_count()
        with pytest.raises(ValueError, match="exceeds available CPU count"):
            ParallelEmbeddingProcessor(n_processes=max_cpus + 1)

    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_init_invalid_model_name_empty(self, mock_model_class):
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            ParallelEmbeddingProcessor(model_name="")

    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_init_invalid_model_name_whitespace(self, mock_model_class):
        """Test that whitespace-only model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            ParallelEmbeddingProcessor(model_name="   ")

    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_init_invalid_model_name_type(self, mock_model_class):
        """Test that non-string model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            ParallelEmbeddingProcessor(model_name=123)


class TestParallelEmbeddingProcessorFunctionality:
    """Test suite for ParallelEmbeddingProcessor processing methods."""

    @pytest.fixture
    def processor(self):
        """Fixture providing an initialized processor."""
        with patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel'):
            return ParallelEmbeddingProcessor(n_processes=2)

    def test_validate_texts_input_valid(self, processor):
        """Test validation with valid texts input."""
        texts = ['Hello world', 'This is a test']

        # Should not raise
        processor._validate_texts_input(texts)

    def test_validate_texts_input_invalid_type(self, processor):
        """Test that non-list texts raises TypeError."""
        with pytest.raises(TypeError, match="texts must be a list"):
            processor._validate_texts_input("not a list")

    def test_validate_texts_input_empty(self, processor):
        """Test that empty texts list raises ValueError."""
        with pytest.raises(ValueError, match="texts cannot be empty"):
            processor._validate_texts_input([])

    def test_validate_texts_input_invalid_element_type(self, processor):
        """Test that non-string elements raise TypeError."""
        with pytest.raises(TypeError, match="texts\\[0\\] must be a string"):
            processor._validate_texts_input([123])

    def test_validate_texts_input_empty_string(self, processor):
        """Test that empty string elements raise ValueError."""
        with pytest.raises(ValueError, match="texts\\[0\\] cannot be empty"):
            processor._validate_texts_input([''])

    def test_validate_texts_input_whitespace_only(self, processor):
        """Test that whitespace-only strings raise ValueError."""
        with pytest.raises(ValueError, match="texts\\[1\\] cannot be empty"):
            processor._validate_texts_input(['Valid text', '   '])

    def test_validate_texts_input_custom_param_name(self, processor):
        """Test validation with custom parameter name."""
        with pytest.raises(TypeError, match="custom_param must be a list"):
            processor._validate_texts_input("not a list", param_name="custom_param")

    def test_process_batch_small_uses_sequential(self, processor):
        """Test that small batches use sequential processing."""
        texts = ['Hello', 'World']
        mock_embedding = np.random.rand(384)

        processor.model.encode.return_value = mock_embedding

        # Small batch (< n_processes * 2) should use sequential
        embeddings = processor.process_batch(texts, show_progress=False)

        # Verify model.encode was called for each text
        assert processor.model.encode.call_count == 2
        assert len(embeddings) == 2

    @patch('tri_lingual_agents.embeddings.parallel.mp.Pool')
    def test_process_batch_large_uses_parallel(self, mock_pool_class, processor):
        """Test that large batches use parallel processing."""
        texts = ['Text {}'.format(i) for i in range(10)]
        mock_embeddings = [np.random.rand(384) for _ in range(10)]

        # Mock pool
        mock_pool = Mock()
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.map.return_value = mock_embeddings
        mock_pool_class.return_value = mock_pool

        embeddings = processor.process_batch(texts, show_progress=False)

        # Verify pool was created and used
        mock_pool_class.assert_called_once_with(processes=2)
        mock_pool.map.assert_called_once()
        assert len(embeddings) == 10

    @patch('tri_lingual_agents.embeddings.parallel.mp.Pool')
    @patch('tqdm.tqdm')
    def test_process_batch_with_progress(self, mock_tqdm, mock_pool_class, processor):
        """Test that progress bar is shown when show_progress=True."""
        texts = ['Text {}'.format(i) for i in range(10)]
        mock_embeddings = [np.random.rand(384) for _ in range(10)]

        # Mock pool
        mock_pool = Mock()
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.imap.return_value = mock_embeddings
        mock_pool_class.return_value = mock_pool

        # Mock tqdm
        mock_tqdm.return_value = mock_embeddings

        processor.process_batch(texts, show_progress=True)

        # Verify tqdm was called
        mock_tqdm.assert_called_once()

    @patch('tri_lingual_agents.embeddings.parallel.mp.Pool')
    def test_calculate_distances_parallel_valid_inputs(self, mock_pool_class, processor):
        """Test distance calculation with valid inputs."""
        original_texts = ['Hello', 'World']
        final_texts = ['Bonjour', 'Monde']

        # Mock embeddings
        mock_embeddings = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            np.array([0.15, 0.25, 0.35]),
            np.array([0.45, 0.55, 0.65])
        ]

        # Mock pool
        mock_pool = Mock()
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.map.return_value = mock_embeddings
        mock_pool_class.return_value = mock_pool

        distances = processor.calculate_distances_parallel(
            original_texts, final_texts, show_progress=False
        )

        # Verify distances were calculated
        assert len(distances) == 2
        assert all(isinstance(d, float) for d in distances)

    def test_calculate_distances_parallel_mismatched_lengths(self, processor):
        """Test that mismatched text list lengths raise ValueError."""
        original_texts = ['Hello', 'World']
        final_texts = ['Bonjour']

        with pytest.raises(ValueError, match="must have the same length"):
            processor.calculate_distances_parallel(original_texts, final_texts)

    def test_calculate_distances_parallel_validates_original_texts(self, processor):
        """Test that original_texts are validated."""
        original_texts = "not a list"
        final_texts = ['Bonjour']

        with pytest.raises(TypeError, match="original_texts must be a list"):
            processor.calculate_distances_parallel(original_texts, final_texts)

    def test_calculate_distances_parallel_validates_final_texts(self, processor):
        """Test that final_texts are validated."""
        original_texts = ['Hello']
        final_texts = "not a list"

        with pytest.raises(TypeError, match="final_texts must be a list"):
            processor.calculate_distances_parallel(original_texts, final_texts)

    def test_calculate_distances_parallel_combines_texts(self, processor):
        """Test that original and final texts are combined for processing."""
        original_texts = ['Hello world this is a test sentence']
        final_texts = ['Bonjour monde ceci est une phrase test']

        # Mock embeddings (2 total: 1 original + 1 final)
        mock_embeddings = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.15, 0.25, 0.35])
        ]

        # Mock the process_batch method to return our embeddings
        with patch.object(processor, 'process_batch', return_value=mock_embeddings):
            distances = processor.calculate_distances_parallel(
                original_texts, final_texts, show_progress=False
            )

            # Verify process_batch was called with combined texts
            processor.process_batch.assert_called_once()
            call_args = processor.process_batch.call_args[0][0]
            # Combined list should have both texts
            assert len(call_args) == 2
            assert original_texts[0] in call_args
            assert final_texts[0] in call_args

        # Verify we got a distance result
        assert len(distances) == 1
        assert isinstance(distances[0], float)


class TestBenchmarkParallelVsSequential:
    """Test suite for benchmark_parallel_vs_sequential function."""

    @patch('time.time')
    @patch('tri_lingual_agents.embeddings.parallel.ParallelEmbeddingProcessor')
    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_benchmark_returns_correct_structure(
        self, mock_model_class, mock_processor_class, mock_time
    ):
        """Test that benchmark returns correct result structure."""
        texts = ['Test text']

        # Mock time to simulate duration
        mock_time.side_effect = [0, 10, 10, 15]  # sequential: 10s, parallel: 5s

        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_model_class.return_value = mock_model

        # Mock processor
        mock_processor = Mock()
        mock_processor.n_processes = 4
        mock_processor.process_batch.return_value = [np.random.rand(384)]
        mock_processor_class.return_value = mock_processor

        result = benchmark_parallel_vs_sequential(texts)

        # Verify result structure
        assert 'sequential_time' in result
        assert 'parallel_time' in result
        assert 'speedup' in result
        assert 'num_texts' in result
        assert 'num_processes' in result

    @patch('time.time')
    @patch('tri_lingual_agents.embeddings.parallel.ParallelEmbeddingProcessor')
    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_benchmark_calculates_speedup(
        self, mock_model_class, mock_processor_class, mock_time
    ):
        """Test that speedup is calculated correctly."""
        texts = ['Test text']

        # Mock time: sequential = 10s, parallel = 2s
        mock_time.side_effect = [0, 10, 10, 12]

        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_model_class.return_value = mock_model

        # Mock processor
        mock_processor = Mock()
        mock_processor.n_processes = 4
        mock_processor.process_batch.return_value = [np.random.rand(384)]
        mock_processor_class.return_value = mock_processor

        result = benchmark_parallel_vs_sequential(texts)

        # Speedup should be 10/2 = 5.0
        assert result['speedup'] == pytest.approx(5.0)

    @patch('tri_lingual_agents.embeddings.parallel.ParallelEmbeddingProcessor')
    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_benchmark_processes_all_texts_sequentially(
        self, mock_model_class, mock_processor_class
    ):
        """Test that all texts are processed in sequential mode."""
        texts = ['First', 'Second', 'Third']

        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_model_class.return_value = mock_model

        # Mock processor
        mock_processor = Mock()
        mock_processor.n_processes = 4
        mock_processor.process_batch.return_value = [np.random.rand(384) for _ in range(3)]
        mock_processor_class.return_value = mock_processor

        benchmark_parallel_vs_sequential(texts)

        # Verify each text was encoded sequentially
        assert mock_model.encode.call_count == 3

    @patch('tri_lingual_agents.embeddings.parallel.ParallelEmbeddingProcessor')
    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_benchmark_uses_custom_model_name(
        self, mock_model_class, mock_processor_class
    ):
        """Test benchmark with custom model name."""
        texts = ['Test']
        custom_model = 'custom-model'

        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_model_class.return_value = mock_model

        # Mock processor
        mock_processor = Mock()
        mock_processor.n_processes = 4
        mock_processor.process_batch.return_value = [np.random.rand(384)]
        mock_processor_class.return_value = mock_processor

        benchmark_parallel_vs_sequential(texts, model_name=custom_model)

        # Verify custom model was used
        mock_model_class.assert_called_with(custom_model)
        mock_processor_class.assert_called_with(custom_model, None)

    @patch('tri_lingual_agents.embeddings.parallel.ParallelEmbeddingProcessor')
    @patch('tri_lingual_agents.embeddings.parallel.EmbeddingModel')
    def test_benchmark_uses_custom_n_processes(
        self, mock_model_class, mock_processor_class
    ):
        """Test benchmark with custom n_processes."""
        texts = ['Test']
        custom_processes = 8

        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_model_class.return_value = mock_model

        # Mock processor
        mock_processor = Mock()
        mock_processor.n_processes = custom_processes
        mock_processor.process_batch.return_value = [np.random.rand(384)]
        mock_processor_class.return_value = mock_processor

        result = benchmark_parallel_vs_sequential(texts, n_processes=custom_processes)

        # Verify custom n_processes was used
        mock_processor_class.assert_called_with("all-MiniLM-L6-v2", custom_processes)
        assert result['num_processes'] == custom_processes
