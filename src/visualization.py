"""
Visualization Module

This module provides functions to generate graphs and visualizations
for the tri-lingual pipeline experiment results.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Dict, Optional
import os
from pathlib import Path


def plot_error_vs_distance(
    error_rates: List[float],
    distances: List[float],
    output_path: str = 'results/error_rate_vs_distance.png',
    title: str = 'Semantic Drift vs. Spelling Error Rate in Tri-Lingual Agent Pipeline',
    figsize: tuple = (12, 7),
    dpi: int = 300,
    show: bool = False
):
    """
    Plot error rate vs. semantic distance graph.

    Args:
        error_rates: List of error rates (as percentages, e.g., [0, 10, 20, ...])
        distances: List of corresponding distance values
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height) in inches
        dpi: DPI for output image
        show: Whether to display the plot interactively
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data
    ax.plot(
        error_rates,
        distances,
        marker='o',
        linewidth=2.5,
        markersize=10,
        color='#2E86AB',
        markerfacecolor='#A23B72',
        markeredgewidth=2,
        markeredgecolor='#2E86AB',
        label='Semantic Distance'
    )

    # Customize axes
    ax.set_xlabel('Spelling Error Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cosine Distance', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Set x-axis ticks to show all error rates
    ax.set_xticks(error_rates)
    ax.set_xticklabels([f'{int(r)}%' for r in error_rates])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Add legend
    ax.legend(fontsize=12, loc='upper left')

    # Tight layout
    plt.tight_layout()

    # Create output directory if it doesn't exist
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Graph saved to: {output_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def plot_from_results(
    results: List[Dict],
    output_path: str = 'results/error_rate_vs_distance.png',
    **kwargs
):
    """
    Generate plot directly from experiment results.

    Args:
        results: List of experiment result dictionaries
        output_path: Path to save the plot
        **kwargs: Additional arguments passed to plot_error_vs_distance
    """
    # Extract error rates and distances
    error_rates = [r['error_rate_percent'] for r in results]
    distances = [r['cosine_distance'] for r in results]

    # Generate plot
    plot_error_vs_distance(error_rates, distances, output_path, **kwargs)


def create_summary_figure(
    results: List[Dict],
    output_path: str = 'results/experiment_summary.png',
    dpi: int = 300
):
    """
    Create a comprehensive summary figure with multiple subplots.

    Args:
        results: List of experiment result dictionaries
        output_path: Path to save the figure
        dpi: DPI for output image
    """
    # Extract data
    error_rates = [r['error_rate_percent'] for r in results]
    distances = [r['cosine_distance'] for r in results]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Error rate vs distance (line plot)
    ax1.plot(
        error_rates,
        distances,
        marker='o',
        linewidth=2.5,
        markersize=10,
        color='#2E86AB',
        markerfacecolor='#A23B72',
        markeredgewidth=2,
        markeredgecolor='#2E86AB'
    )
    ax1.set_xlabel('Spelling Error Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cosine Distance', fontsize=12, fontweight='bold')
    ax1.set_title('Semantic Drift vs. Error Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(error_rates)

    # Subplot 2: Bar chart showing distance at each error rate
    colors = ['#2E86AB' if i == 0 else '#A23B72' for i in range(len(error_rates))]
    ax2.bar(
        [f'{int(r)}%' for r in error_rates],
        distances,
        color=colors,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    ax2.set_xlabel('Spelling Error Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cosine Distance', fontsize=12, fontweight='bold')
    ax2.set_title('Distance Distribution by Error Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Overall title
    fig.suptitle(
        'Tri-Lingual Agent Pipeline: Error Rate Impact on Semantic Drift',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    # Tight layout
    plt.tight_layout()

    # Create output directory
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Summary figure saved to: {output_path}")
    plt.close()


def generate_all_visualizations(
    results: List[Dict],
    output_dir: str = 'results'
):
    """
    Generate all standard visualizations for the experiment.

    Args:
        results: List of experiment result dictionaries
        output_dir: Directory to save visualizations
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Main error rate vs. distance plot
    plot_from_results(
        results,
        output_path=os.path.join(output_dir, 'error_rate_vs_distance.png')
    )

    # Comprehensive summary figure
    create_summary_figure(
        results,
        output_path=os.path.join(output_dir, 'experiment_summary.png')
    )

    print("="*70 + "\n")
