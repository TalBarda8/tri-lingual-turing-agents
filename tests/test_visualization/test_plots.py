"""
Unit tests for visualization module.

Tests plotting functions with mocked matplotlib to avoid actual rendering.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

from tri_lingual_agents.visualization.plots import (
    plot_error_vs_distance,
    plot_from_results,
    create_summary_figure,
    generate_all_visualizations,
)


class TestPlotErrorVsDistance:
    """Test suite for plot_error_vs_distance function."""

    @pytest.fixture
    def sample_data(self):
        """Sample error rates and distances."""
        return {
            'error_rates': [0, 10, 20, 30, 40, 50],
            'distances': [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
        }

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_plot_creates_figure(self, mock_plt, sample_data):
        """Test that figure and axis are created."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            plot_error_vs_distance(
                sample_data['error_rates'],
                sample_data['distances'],
                output_path=output_path,
                show=False
            )

        # Verify figure was created
        mock_plt.subplots.assert_called_once()
        call_kwargs = mock_plt.subplots.call_args[1]
        assert 'figsize' in call_kwargs

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_plot_calls_plot_method(self, mock_plt, sample_data):
        """Test that ax.plot is called with data."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            plot_error_vs_distance(
                sample_data['error_rates'],
                sample_data['distances'],
                output_path=output_path,
                show=False
            )

        # Verify plot was called with data
        mock_ax.plot.assert_called_once()
        call_args = mock_ax.plot.call_args[0]
        assert call_args[0] == sample_data['error_rates']
        assert call_args[1] == sample_data['distances']

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_plot_sets_labels(self, mock_plt, sample_data):
        """Test that axis labels are set."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            plot_error_vs_distance(
                sample_data['error_rates'],
                sample_data['distances'],
                output_path=output_path,
                show=False
            )

        # Verify labels were set
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_ylabel.assert_called_once()
        mock_ax.set_title.assert_called_once()

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_plot_saves_to_file(self, mock_plt, sample_data):
        """Test that plot is saved to file."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            plot_error_vs_distance(
                sample_data['error_rates'],
                sample_data['distances'],
                output_path=output_path,
                show=False
            )

        # Verify savefig was called
        mock_plt.savefig.assert_called_once()
        call_args = mock_plt.savefig.call_args
        assert call_args[0][0] == output_path

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_plot_creates_output_directory(self, mock_plt, sample_data):
        """Test that output directory is created if it doesn't exist."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'new_dir', 'test.png')
            plot_error_vs_distance(
                sample_data['error_rates'],
                sample_data['distances'],
                output_path=output_path,
                show=False
            )

            # Verify directory was created (check within tempdir context)
            assert os.path.exists(os.path.dirname(output_path))

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_plot_closes_figure_when_show_false(self, mock_plt, sample_data):
        """Test that figure is closed when show=False."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            plot_error_vs_distance(
                sample_data['error_rates'],
                sample_data['distances'],
                output_path=output_path,
                show=False
            )

        # Verify close was called
        mock_plt.close.assert_called_once()
        # show should not be called
        mock_plt.show.assert_not_called()

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_plot_shows_when_show_true(self, mock_plt, sample_data):
        """Test that plot.show is called when show=True."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            plot_error_vs_distance(
                sample_data['error_rates'],
                sample_data['distances'],
                output_path=output_path,
                show=True
            )

        # Verify show was called
        mock_plt.show.assert_called_once()
        # close should not be called
        mock_plt.close.assert_not_called()

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_plot_custom_title(self, mock_plt, sample_data):
        """Test that custom title is used."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        custom_title = "My Custom Title"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            plot_error_vs_distance(
                sample_data['error_rates'],
                sample_data['distances'],
                output_path=output_path,
                title=custom_title,
                show=False
            )

        # Verify custom title was used
        mock_ax.set_title.assert_called_once()
        call_args = mock_ax.set_title.call_args[0]
        assert call_args[0] == custom_title

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_plot_custom_figsize(self, mock_plt, sample_data):
        """Test that custom figsize is used."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        custom_figsize = (10, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            plot_error_vs_distance(
                sample_data['error_rates'],
                sample_data['distances'],
                output_path=output_path,
                figsize=custom_figsize,
                show=False
            )

        # Verify custom figsize was used
        call_kwargs = mock_plt.subplots.call_args[1]
        assert call_kwargs['figsize'] == custom_figsize

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_plot_custom_dpi(self, mock_plt, sample_data):
        """Test that custom DPI is used."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        custom_dpi = 150

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            plot_error_vs_distance(
                sample_data['error_rates'],
                sample_data['distances'],
                output_path=output_path,
                dpi=custom_dpi,
                show=False
            )

        # Verify custom DPI was used in savefig
        call_kwargs = mock_plt.savefig.call_args[1]
        assert call_kwargs['dpi'] == custom_dpi


class TestPlotFromResults:
    """Test suite for plot_from_results function."""

    @pytest.fixture
    def sample_results(self):
        """Sample experiment results."""
        return [
            {'error_rate_percent': 0.0, 'cosine_distance': 0.05},
            {'error_rate_percent': 10.0, 'cosine_distance': 0.15},
            {'error_rate_percent': 20.0, 'cosine_distance': 0.25},
            {'error_rate_percent': 30.0, 'cosine_distance': 0.35},
        ]

    @patch('tri_lingual_agents.visualization.plots.plot_error_vs_distance')
    def test_plot_from_results_extracts_data(self, mock_plot, sample_results):
        """Test that data is extracted from results."""
        output_path = 'test.png'

        plot_from_results(sample_results, output_path=output_path)

        # Verify plot_error_vs_distance was called
        mock_plot.assert_called_once()

        # Verify correct data extraction
        call_args = mock_plot.call_args[0]
        error_rates = call_args[0]
        distances = call_args[1]
        output = call_args[2]

        assert error_rates == [0.0, 10.0, 20.0, 30.0]
        assert distances == [0.05, 0.15, 0.25, 0.35]
        assert output == output_path

    @patch('tri_lingual_agents.visualization.plots.plot_error_vs_distance')
    def test_plot_from_results_passes_kwargs(self, mock_plot, sample_results):
        """Test that additional kwargs are passed through."""
        plot_from_results(
            sample_results,
            output_path='test.png',
            title='Custom Title',
            dpi=150,
            show=True
        )

        # Verify kwargs were passed
        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs['title'] == 'Custom Title'
        assert call_kwargs['dpi'] == 150
        assert call_kwargs['show'] == True

    @patch('tri_lingual_agents.visualization.plots.plot_error_vs_distance')
    def test_plot_from_results_empty_list(self, mock_plot):
        """Test with empty results list."""
        plot_from_results([], output_path='test.png')

        # Verify plot was called with empty lists
        call_args = mock_plot.call_args[0]
        assert call_args[0] == []
        assert call_args[1] == []

    @patch('tri_lingual_agents.visualization.plots.plot_error_vs_distance')
    def test_plot_from_results_single_result(self, mock_plot):
        """Test with single result."""
        results = [{'error_rate_percent': 0.0, 'cosine_distance': 0.1}]

        plot_from_results(results, output_path='test.png')

        # Verify correct extraction
        call_args = mock_plot.call_args[0]
        assert call_args[0] == [0.0]
        assert call_args[1] == [0.1]

    @patch('tri_lingual_agents.visualization.plots.plot_error_vs_distance')
    def test_plot_from_results_default_output_path(self, mock_plot, sample_results):
        """Test default output path."""
        plot_from_results(sample_results)

        # Verify default path was used
        call_args = mock_plot.call_args[0]
        assert call_args[2] == 'results/error_rate_vs_distance.png'


class TestCreateSummaryFigure:
    """Test suite for create_summary_figure function."""

    @pytest.fixture
    def sample_results(self):
        """Sample experiment results."""
        return [
            {'error_rate_percent': 0.0, 'cosine_distance': 0.05},
            {'error_rate_percent': 20.0, 'cosine_distance': 0.25},
            {'error_rate_percent': 40.0, 'cosine_distance': 0.45},
        ]

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_create_summary_figure_creates_subplots(self, mock_plt, sample_results):
        """Test that subplots are created."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'summary.png')
            create_summary_figure(sample_results, output_path=output_path)

        # Verify subplots was called with correct layout
        mock_plt.subplots.assert_called_once()
        call_args = mock_plt.subplots.call_args[0]
        assert call_args[0] == 1  # 1 row
        assert call_args[1] == 2  # 2 columns

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_create_summary_figure_plots_line_chart(self, mock_plt, sample_results):
        """Test that first subplot creates line chart."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'summary.png')
            create_summary_figure(sample_results, output_path=output_path)

        # Verify ax1.plot was called (line chart)
        mock_ax1.plot.assert_called_once()

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_create_summary_figure_plots_bar_chart(self, mock_plt, sample_results):
        """Test that second subplot creates bar chart."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'summary.png')
            create_summary_figure(sample_results, output_path=output_path)

        # Verify ax2.bar was called (bar chart)
        mock_ax2.bar.assert_called_once()

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_create_summary_figure_sets_overall_title(self, mock_plt, sample_results):
        """Test that overall figure title is set."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'summary.png')
            create_summary_figure(sample_results, output_path=output_path)

        # Verify suptitle was called
        mock_fig.suptitle.assert_called_once()

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_create_summary_figure_saves_file(self, mock_plt, sample_results):
        """Test that figure is saved to file."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'summary.png')
            create_summary_figure(sample_results, output_path=output_path)

        # Verify savefig was called
        mock_plt.savefig.assert_called_once()
        call_args = mock_plt.savefig.call_args[0]
        assert call_args[0] == output_path

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_create_summary_figure_custom_dpi(self, mock_plt, sample_results):
        """Test that custom DPI is used."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        custom_dpi = 200

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'summary.png')
            create_summary_figure(sample_results, output_path=output_path, dpi=custom_dpi)

        # Verify custom DPI was used
        call_kwargs = mock_plt.savefig.call_args[1]
        assert call_kwargs['dpi'] == custom_dpi

    @patch('tri_lingual_agents.visualization.plots.plt')
    def test_create_summary_figure_closes_plot(self, mock_plt, sample_results):
        """Test that plot is closed after saving."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'summary.png')
            create_summary_figure(sample_results, output_path=output_path)

        # Verify close was called
        mock_plt.close.assert_called_once()


class TestGenerateAllVisualizations:
    """Test suite for generate_all_visualizations function."""

    @pytest.fixture
    def sample_results(self):
        """Sample experiment results."""
        return [
            {'error_rate_percent': 0.0, 'cosine_distance': 0.05},
            {'error_rate_percent': 20.0, 'cosine_distance': 0.25},
        ]

    @patch('tri_lingual_agents.visualization.plots.plot_from_results')
    @patch('tri_lingual_agents.visualization.plots.create_summary_figure')
    def test_generate_all_calls_plot_from_results(
        self, mock_summary, mock_plot, sample_results
    ):
        """Test that plot_from_results is called."""
        generate_all_visualizations(sample_results, output_dir='test_results')

        # Verify plot_from_results was called
        mock_plot.assert_called_once()
        call_args = mock_plot.call_args[0]
        assert call_args[0] == sample_results

    @patch('tri_lingual_agents.visualization.plots.plot_from_results')
    @patch('tri_lingual_agents.visualization.plots.create_summary_figure')
    def test_generate_all_calls_create_summary(
        self, mock_summary, mock_plot, sample_results
    ):
        """Test that create_summary_figure is called."""
        generate_all_visualizations(sample_results, output_dir='test_results')

        # Verify create_summary_figure was called
        mock_summary.assert_called_once()
        call_args = mock_summary.call_args[0]
        assert call_args[0] == sample_results

    @patch('tri_lingual_agents.visualization.plots.plot_from_results')
    @patch('tri_lingual_agents.visualization.plots.create_summary_figure')
    def test_generate_all_uses_output_dir(
        self, mock_summary, mock_plot, sample_results
    ):
        """Test that output directory is used for both plots."""
        output_dir = 'custom_results'

        generate_all_visualizations(sample_results, output_dir=output_dir)

        # Verify output_dir was used in paths
        plot_call_kwargs = mock_plot.call_args[1]
        summary_call_kwargs = mock_summary.call_args[1]

        assert output_dir in plot_call_kwargs['output_path']
        assert output_dir in summary_call_kwargs['output_path']

    @patch('tri_lingual_agents.visualization.plots.plot_from_results')
    @patch('tri_lingual_agents.visualization.plots.create_summary_figure')
    def test_generate_all_default_output_dir(
        self, mock_summary, mock_plot, sample_results
    ):
        """Test default output directory is 'results'."""
        generate_all_visualizations(sample_results)

        # Verify default 'results' dir was used
        plot_call_kwargs = mock_plot.call_args[1]
        summary_call_kwargs = mock_summary.call_args[1]

        assert 'results' in plot_call_kwargs['output_path']
        assert 'results' in summary_call_kwargs['output_path']

    @patch('tri_lingual_agents.visualization.plots.plot_from_results')
    @patch('tri_lingual_agents.visualization.plots.create_summary_figure')
    def test_generate_all_creates_both_files(
        self, mock_summary, mock_plot, sample_results
    ):
        """Test that both visualization files are created."""
        generate_all_visualizations(sample_results, output_dir='test_results')

        # Verify both functions were called exactly once
        assert mock_plot.call_count == 1
        assert mock_summary.call_count == 1

    @patch('tri_lingual_agents.visualization.plots.plot_from_results')
    @patch('tri_lingual_agents.visualization.plots.create_summary_figure')
    def test_generate_all_correct_filenames(
        self, mock_summary, mock_plot, sample_results
    ):
        """Test that correct filenames are used."""
        generate_all_visualizations(sample_results, output_dir='test_results')

        # Verify correct filenames
        plot_path = mock_plot.call_args[1]['output_path']
        summary_path = mock_summary.call_args[1]['output_path']

        assert plot_path.endswith('error_rate_vs_distance.png')
        assert summary_path.endswith('experiment_summary.png')
