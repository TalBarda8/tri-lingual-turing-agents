"""
Experiment Pipeline Module

This module orchestrates the tri-lingual translation pipeline experiments
and collects data on semantic drift vs. spelling error rates.
"""

from .orchestrator import (
    run_experiment,
    run_error_rate_sweep,
    save_experiment_results,
    load_experiment_results,
    print_summary,
)

__all__ = [
    "run_experiment",
    "run_error_rate_sweep",
    "save_experiment_results",
    "load_experiment_results",
    "print_summary",
]
