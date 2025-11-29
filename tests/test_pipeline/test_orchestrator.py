"""
Unit tests for pipeline orchestrator module.

Tests the experiment orchestration functions with mocked dependencies
to avoid actual API calls and model loading.
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime

from tri_lingual_agents.pipeline.orchestrator import (
    run_experiment,
    run_error_rate_sweep,
    save_experiment_results,
    print_summary,
    load_experiment_results,
)


class TestRunExperiment:
    """Test suite for run_experiment function."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock translation agents."""
        agent_en_fr = Mock()
        agent_en_fr.translate.return_value = "Bonjour le monde"

        agent_fr_he = Mock()
        agent_fr_he.translate.return_value = "שלום עולם"

        agent_he_en = Mock()
        agent_he_en.translate.return_value = "Hello world"

        return agent_en_fr, agent_fr_he, agent_he_en

    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model."""
        model = Mock()
        # Return different embeddings for different inputs
        def encode_side_effect(text):
            # Return deterministic embeddings based on text length
            dim = 384
            seed = len(text)
            np.random.seed(seed)
            return np.random.rand(dim).astype(np.float32)

        model.encode.side_effect = encode_side_effect
        return model

    def test_run_experiment_zero_error_rate(self, mock_agents, mock_embedding_model):
        """Test experiment with zero error rate (no corruption)."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"

        result = run_experiment(
            sentence=sentence,
            error_rate=0.0,
            agent_en_fr=agent_en_fr,
            agent_fr_he=agent_fr_he,
            agent_he_en=agent_he_en,
            embedding_model=mock_embedding_model,
            seed=42
        )

        # Verify result structure
        assert result['error_rate'] == 0.0
        assert result['original_sentence'] == sentence
        assert result['corrupted_sentence'] == sentence  # No corruption
        assert result['corrupted_words'] == []
        assert result['french_translation'] == "Bonjour le monde"
        assert result['hebrew_translation'] == "שלום עולם"
        assert result['final_english'] == "Hello world"
        assert 'cosine_distance' in result
        assert 'timestamp' in result

        # Verify error statistics for zero error rate
        assert result['error_statistics']['corrupted_words'] == 0
        assert result['error_statistics']['error_rate_percent'] == 0.0

    @patch('tri_lingual_agents.pipeline.orchestrator.inject_spelling_errors')
    @patch('tri_lingual_agents.pipeline.orchestrator.calculate_error_statistics')
    def test_run_experiment_with_error_injection(
        self, mock_calc_stats, mock_inject_errors, mock_agents, mock_embedding_model
    ):
        """Test experiment with error injection."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"

        # Mock error injection
        mock_inject_errors.return_value = ("Helo wrld this is a tst", ["Hello", "world", "test"])
        mock_calc_stats.return_value = {
            'total_words': 6,
            'corrupted_words': 3,
            'error_rate_percent': 50.0,
            'corrupted_list': ["Hello", "world", "test"],
            'original_sentence': sentence,
            'corrupted_sentence': "Helo wrld this is a tst"
        }

        result = run_experiment(
            sentence=sentence,
            error_rate=0.3,
            agent_en_fr=agent_en_fr,
            agent_fr_he=agent_fr_he,
            agent_he_en=agent_he_en,
            embedding_model=mock_embedding_model,
            seed=42
        )

        # Verify error injection was called
        mock_inject_errors.assert_called_once_with(sentence, 0.3, 42)
        mock_calc_stats.assert_called_once()

        # Verify result contains corruption info
        assert result['error_rate'] == 0.3
        assert result['corrupted_sentence'] == "Helo wrld this is a tst"
        assert len(result['corrupted_words']) == 3
        assert result['error_statistics']['corrupted_words'] == 3

    def test_run_experiment_agents_called_in_order(self, mock_agents, mock_embedding_model):
        """Test that agents are called in correct sequence."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"

        run_experiment(
            sentence=sentence,
            error_rate=0.0,
            agent_en_fr=agent_en_fr,
            agent_fr_he=agent_fr_he,
            agent_he_en=agent_he_en,
            embedding_model=mock_embedding_model
        )

        # Verify agents were called with correct inputs
        agent_en_fr.translate.assert_called_once()
        assert agent_en_fr.translate.call_args[0][0] == sentence

        agent_fr_he.translate.assert_called_once()
        assert agent_fr_he.translate.call_args[0][0] == "Bonjour le monde"

        agent_he_en.translate.assert_called_once()
        assert agent_he_en.translate.call_args[0][0] == "שלום עולם"

    def test_run_experiment_first_agent_error_handling(self, mock_agents, mock_embedding_model):
        """Test that first agent is called with handle_errors=True."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"

        run_experiment(
            sentence=sentence,
            error_rate=0.2,
            agent_en_fr=agent_en_fr,
            agent_fr_he=agent_fr_he,
            agent_he_en=agent_he_en,
            embedding_model=mock_embedding_model
        )

        # First agent should be called with handle_errors=True
        call_kwargs = agent_en_fr.translate.call_args[1]
        assert call_kwargs.get('handle_errors') == True

    def test_run_experiment_embeddings_calculated(self, mock_agents, mock_embedding_model):
        """Test that embeddings are calculated for original and final sentences."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"

        result = run_experiment(
            sentence=sentence,
            error_rate=0.0,
            agent_en_fr=agent_en_fr,
            agent_fr_he=agent_fr_he,
            agent_he_en=agent_he_en,
            embedding_model=mock_embedding_model
        )

        # Verify encode was called twice (original and final)
        assert mock_embedding_model.encode.call_count == 2

        # Verify distance is a float
        assert isinstance(result['cosine_distance'], float)
        assert result['cosine_distance'] >= 0

    def test_run_experiment_timestamp_included(self, mock_agents, mock_embedding_model):
        """Test that result includes ISO format timestamp."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"

        result = run_experiment(
            sentence=sentence,
            error_rate=0.0,
            agent_en_fr=agent_en_fr,
            agent_fr_he=agent_fr_he,
            agent_he_en=agent_he_en,
            embedding_model=mock_embedding_model
        )

        # Verify timestamp is valid ISO format
        assert 'timestamp' in result
        timestamp = datetime.fromisoformat(result['timestamp'])
        assert isinstance(timestamp, datetime)

    def test_run_experiment_with_seed(self, mock_agents, mock_embedding_model):
        """Test that seed is passed to error injection."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"

        with patch('tri_lingual_agents.pipeline.orchestrator.inject_spelling_errors') as mock_inject:
            mock_inject.return_value = (sentence, [])

            run_experiment(
                sentence=sentence,
                error_rate=0.2,
                agent_en_fr=agent_en_fr,
                agent_fr_he=agent_fr_he,
                agent_he_en=agent_he_en,
                embedding_model=mock_embedding_model,
                seed=123
            )

            # Verify seed was passed
            assert mock_inject.call_args[0][2] == 123

    def test_run_experiment_result_completeness(self, mock_agents, mock_embedding_model):
        """Test that result contains all required fields."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"

        result = run_experiment(
            sentence=sentence,
            error_rate=0.1,
            agent_en_fr=agent_en_fr,
            agent_fr_he=agent_fr_he,
            agent_he_en=agent_he_en,
            embedding_model=mock_embedding_model
        )

        # Verify all required fields are present
        required_fields = [
            'error_rate',
            'error_rate_percent',
            'original_sentence',
            'corrupted_sentence',
            'corrupted_words',
            'error_statistics',
            'french_translation',
            'hebrew_translation',
            'final_english',
            'cosine_distance',
            'timestamp'
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_run_experiment_error_rate_percent_calculation(self, mock_agents, mock_embedding_model):
        """Test that error_rate_percent is correctly calculated."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"

        result = run_experiment(
            sentence=sentence,
            error_rate=0.25,
            agent_en_fr=agent_en_fr,
            agent_fr_he=agent_fr_he,
            agent_he_en=agent_he_en,
            embedding_model=mock_embedding_model
        )

        assert result['error_rate_percent'] == 25.0


class TestRunErrorRateSweep:
    """Test suite for run_error_rate_sweep function."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock translation agents."""
        agent_en_fr = Mock()
        agent_en_fr.translate.return_value = "Bonjour"

        agent_fr_he = Mock()
        agent_fr_he.translate.return_value = "שלום"

        agent_he_en = Mock()
        agent_he_en.translate.return_value = "Hello"

        return agent_en_fr, agent_fr_he, agent_he_en

    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model."""
        model = Mock()
        model.encode.return_value = np.random.rand(384).astype(np.float32)
        return model

    def test_run_error_rate_sweep_single_rate(self, mock_agents, mock_embedding_model):
        """Test sweep with single error rate."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"
        error_rates = [0.0]

        with patch('tri_lingual_agents.pipeline.orchestrator.save_experiment_results'):
            results = run_error_rate_sweep(
                base_sentence=sentence,
                error_rates=error_rates,
                agent_en_fr=agent_en_fr,
                agent_fr_he=agent_fr_he,
                agent_he_en=agent_he_en,
                embedding_model=mock_embedding_model,
                save_results=False
            )

        assert len(results) == 1
        assert results[0]['error_rate'] == 0.0

    def test_run_error_rate_sweep_multiple_rates(self, mock_agents, mock_embedding_model):
        """Test sweep with multiple error rates."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"
        error_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        with patch('tri_lingual_agents.pipeline.orchestrator.save_experiment_results'):
            results = run_error_rate_sweep(
                base_sentence=sentence,
                error_rates=error_rates,
                agent_en_fr=agent_en_fr,
                agent_fr_he=agent_fr_he,
                agent_he_en=agent_he_en,
                embedding_model=mock_embedding_model,
                save_results=False
            )

        assert len(results) == 6
        for i, result in enumerate(results):
            assert result['error_rate'] == error_rates[i]

    def test_run_error_rate_sweep_seeds_increment(self, mock_agents, mock_embedding_model):
        """Test that seeds increment for each experiment."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"
        error_rates = [0.1, 0.2, 0.3]

        with patch('tri_lingual_agents.pipeline.orchestrator.run_experiment') as mock_run_exp:
            mock_run_exp.return_value = {
                'error_rate': 0.0,
                'cosine_distance': 0.0,
                'error_rate_percent': 0.0
            }

            run_error_rate_sweep(
                base_sentence=sentence,
                error_rates=error_rates,
                agent_en_fr=agent_en_fr,
                agent_fr_he=agent_fr_he,
                agent_he_en=agent_he_en,
                embedding_model=mock_embedding_model,
                save_results=False
            )

            # Verify seeds: 42, 43, 44
            calls = mock_run_exp.call_args_list
            assert calls[0][1]['seed'] == 42
            assert calls[1][1]['seed'] == 43
            assert calls[2][1]['seed'] == 44

    def test_run_error_rate_sweep_saves_results(self, mock_agents, mock_embedding_model):
        """Test that results are saved when save_results=True."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"
        error_rates = [0.0, 0.1]

        with patch('tri_lingual_agents.pipeline.orchestrator.save_experiment_results') as mock_save:
            results = run_error_rate_sweep(
                base_sentence=sentence,
                error_rates=error_rates,
                agent_en_fr=agent_en_fr,
                agent_fr_he=agent_fr_he,
                agent_he_en=agent_he_en,
                embedding_model=mock_embedding_model,
                save_results=True,
                output_dir='test_results'
            )

            # Verify save was called with correct arguments
            mock_save.assert_called_once()
            assert mock_save.call_args[0][0] == results
            assert mock_save.call_args[0][1] == sentence
            assert mock_save.call_args[0][2] == 'test_results'

    def test_run_error_rate_sweep_no_save(self, mock_agents, mock_embedding_model):
        """Test that results are not saved when save_results=False."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"
        error_rates = [0.0]

        with patch('tri_lingual_agents.pipeline.orchestrator.save_experiment_results') as mock_save:
            run_error_rate_sweep(
                base_sentence=sentence,
                error_rates=error_rates,
                agent_en_fr=agent_en_fr,
                agent_fr_he=agent_fr_he,
                agent_he_en=agent_he_en,
                embedding_model=mock_embedding_model,
                save_results=False
            )

            # Verify save was not called
            mock_save.assert_not_called()

    def test_run_error_rate_sweep_prints_summary(self, mock_agents, mock_embedding_model):
        """Test that summary is printed after sweep."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"
        error_rates = [0.0]

        with patch('tri_lingual_agents.pipeline.orchestrator.print_summary') as mock_print:
            with patch('tri_lingual_agents.pipeline.orchestrator.save_experiment_results'):
                results = run_error_rate_sweep(
                    base_sentence=sentence,
                    error_rates=error_rates,
                    agent_en_fr=agent_en_fr,
                    agent_fr_he=agent_fr_he,
                    agent_he_en=agent_he_en,
                    embedding_model=mock_embedding_model,
                    save_results=False
                )

                # Verify print_summary was called with results
                mock_print.assert_called_once_with(results)

    def test_run_error_rate_sweep_empty_error_rates(self, mock_agents, mock_embedding_model):
        """Test sweep with empty error rates list."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"
        error_rates = []

        with patch('tri_lingual_agents.pipeline.orchestrator.save_experiment_results'):
            results = run_error_rate_sweep(
                base_sentence=sentence,
                error_rates=error_rates,
                agent_en_fr=agent_en_fr,
                agent_fr_he=agent_fr_he,
                agent_he_en=agent_he_en,
                embedding_model=mock_embedding_model,
                save_results=False
            )

        assert len(results) == 0

    def test_run_error_rate_sweep_default_output_dir(self, mock_agents, mock_embedding_model):
        """Test that default output directory is 'results'."""
        agent_en_fr, agent_fr_he, agent_he_en = mock_agents
        sentence = "Hello world this is a test"
        error_rates = [0.0]

        with patch('tri_lingual_agents.pipeline.orchestrator.save_experiment_results') as mock_save:
            run_error_rate_sweep(
                base_sentence=sentence,
                error_rates=error_rates,
                agent_en_fr=agent_en_fr,
                agent_fr_he=agent_fr_he,
                agent_he_en=agent_he_en,
                embedding_model=mock_embedding_model,
                save_results=True
            )

            # Default output_dir should be 'results'
            assert mock_save.call_args[0][2] == 'results'


class TestSaveLoadResults:
    """Test suite for save_experiment_results and load_experiment_results."""

    def test_save_experiment_results_creates_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'new_results_dir')
            results = [{'error_rate': 0.0, 'cosine_distance': 0.0}]

            save_experiment_results(results, "Test sentence", output_dir)

            # Verify directory was created
            assert os.path.exists(output_dir)

    def test_save_experiment_results_creates_json_file(self):
        """Test that JSON file is created with correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [
                {'error_rate': 0.0, 'cosine_distance': 0.1},
                {'error_rate': 0.2, 'cosine_distance': 0.3}
            ]
            base_sentence = "Test sentence"

            save_experiment_results(results, base_sentence, tmpdir)

            # Find the created JSON file
            json_files = [f for f in os.listdir(tmpdir) if f.endswith('.json')]
            assert len(json_files) == 1

            # Read and verify structure
            filepath = os.path.join(tmpdir, json_files[0])
            with open(filepath, 'r') as f:
                data = json.load(f)

            assert 'experiment_metadata' in data
            assert 'results' in data
            assert data['experiment_metadata']['base_sentence'] == base_sentence
            assert data['experiment_metadata']['num_error_rates'] == 2
            assert data['results'] == results

    def test_save_experiment_results_filename_format(self):
        """Test that filename includes timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [{'error_rate': 0.0}]

            save_experiment_results(results, "Test", tmpdir)

            json_files = [f for f in os.listdir(tmpdir) if f.endswith('.json')]
            assert len(json_files) == 1

            # Filename should match pattern: experiment_results_YYYYMMDD_HHMMSS.json
            filename = json_files[0]
            assert filename.startswith('experiment_results_')
            assert filename.endswith('.json')

    def test_save_experiment_results_utf8_encoding(self):
        """Test that file is saved with UTF-8 encoding (supports Hebrew, French)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = [{
                'error_rate': 0.0,
                'hebrew_translation': 'שלום עולם',
                'french_translation': 'Bonjour le monde'
            }]

            save_experiment_results(results, "Test", tmpdir)

            json_files = [f for f in os.listdir(tmpdir) if f.endswith('.json')]
            filepath = os.path.join(tmpdir, json_files[0])

            # Read and verify UTF-8 encoding preserved
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert data['results'][0]['hebrew_translation'] == 'שלום עולם'
            assert data['results'][0]['french_translation'] == 'Bonjour le monde'

    def test_load_experiment_results_valid_file(self):
        """Test loading results from valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test results file
            test_data = {
                'experiment_metadata': {
                    'base_sentence': 'Test sentence',
                    'num_error_rates': 2,
                    'error_rates': [0.0, 0.1]
                },
                'results': [
                    {'error_rate': 0.0, 'cosine_distance': 0.1},
                    {'error_rate': 0.1, 'cosine_distance': 0.2}
                ]
            }

            filepath = os.path.join(tmpdir, 'test_results.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(test_data, f)

            # Load and verify
            metadata, results = load_experiment_results(filepath)

            assert metadata == test_data['experiment_metadata']
            assert results == test_data['results']
            assert len(results) == 2

    def test_save_load_roundtrip(self):
        """Test that saved data can be loaded correctly (roundtrip)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_results = [
                {
                    'error_rate': 0.0,
                    'cosine_distance': 0.15,
                    'french_translation': 'Bonjour',
                    'hebrew_translation': 'שלום'
                },
                {
                    'error_rate': 0.3,
                    'cosine_distance': 0.45,
                    'french_translation': 'Au revoir',
                    'hebrew_translation': 'להתראות'
                }
            ]
            base_sentence = "Hello world"

            # Save
            save_experiment_results(original_results, base_sentence, tmpdir)

            # Find saved file
            json_files = [f for f in os.listdir(tmpdir) if f.endswith('.json')]
            filepath = os.path.join(tmpdir, json_files[0])

            # Load
            metadata, loaded_results = load_experiment_results(filepath)

            # Verify
            assert metadata['base_sentence'] == base_sentence
            assert metadata['num_error_rates'] == 2
            assert loaded_results == original_results


class TestPrintSummary:
    """Test suite for print_summary function."""

    def test_print_summary_basic_output(self, capsys):
        """Test that summary prints basic structure."""
        results = [
            {'error_rate_percent': 0.0, 'cosine_distance': 0.05},
            {'error_rate_percent': 20.0, 'cosine_distance': 0.25},
            {'error_rate_percent': 50.0, 'cosine_distance': 0.65}
        ]

        print_summary(results)

        captured = capsys.readouterr()
        output = captured.out

        # Verify header is present
        assert 'EXPERIMENT SUMMARY' in output
        assert 'Error Rate' in output
        assert 'Distance' in output
        assert 'Interpretation' in output

    def test_print_summary_all_error_rates_shown(self, capsys):
        """Test that all error rates are displayed."""
        results = [
            {'error_rate_percent': 0.0, 'cosine_distance': 0.05},
            {'error_rate_percent': 10.0, 'cosine_distance': 0.15},
            {'error_rate_percent': 20.0, 'cosine_distance': 0.25}
        ]

        print_summary(results)

        captured = capsys.readouterr()
        output = captured.out

        assert '0.0%' in output
        assert '10.0%' in output
        assert '20.0%' in output

    def test_print_summary_interpretations_very_similar(self, capsys):
        """Test 'Very similar' interpretation for distance < 0.1."""
        results = [{'error_rate_percent': 0.0, 'cosine_distance': 0.05}]

        print_summary(results)

        captured = capsys.readouterr()
        assert 'Very similar' in captured.out

    def test_print_summary_interpretations_similar(self, capsys):
        """Test 'Similar' interpretation for 0.1 <= distance < 0.3."""
        results = [{'error_rate_percent': 10.0, 'cosine_distance': 0.15}]

        print_summary(results)

        captured = capsys.readouterr()
        assert 'Similar' in captured.out

    def test_print_summary_interpretations_moderate_drift(self, capsys):
        """Test 'Moderate drift' interpretation for 0.3 <= distance < 0.5."""
        results = [{'error_rate_percent': 20.0, 'cosine_distance': 0.35}]

        print_summary(results)

        captured = capsys.readouterr()
        assert 'Moderate drift' in captured.out

    def test_print_summary_interpretations_high_drift(self, capsys):
        """Test 'High drift' interpretation for distance >= 0.5."""
        results = [{'error_rate_percent': 50.0, 'cosine_distance': 0.65}]

        print_summary(results)

        captured = capsys.readouterr()
        assert 'High drift' in captured.out

    def test_print_summary_empty_results(self, capsys):
        """Test print_summary with empty results list."""
        results = []

        print_summary(results)

        captured = capsys.readouterr()
        output = captured.out

        # Should still print header
        assert 'EXPERIMENT SUMMARY' in output
