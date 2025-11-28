"""
Translation Agents Module

This module provides translation agents for the tri-lingual pipeline.
Each agent handles translation between two specific languages using LLM APIs.

Includes parallel processing capabilities for I/O-bound operations.
"""

from .translators import TranslationAgent, create_agent_pipeline
from .parallel import (
    ParallelAgentOrchestrator,
    ThreadSafeCounter,
    benchmark_parallel_vs_sequential_agents,
)

__all__ = [
    # Core agent classes
    "TranslationAgent",
    "create_agent_pipeline",
    # Parallel processing
    "ParallelAgentOrchestrator",
    "ThreadSafeCounter",
    "benchmark_parallel_vs_sequential_agents",
]
