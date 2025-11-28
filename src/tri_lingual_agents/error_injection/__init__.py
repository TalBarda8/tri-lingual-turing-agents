"""
Spelling Error Injection Module

This module provides functions to inject realistic spelling errors into English sentences
at controlled rates for robustness testing.
"""

from .injector import (
    inject_spelling_errors,
    calculate_error_statistics,
    corrupt_word,
    should_corrupt_word,
)

__all__ = [
    "inject_spelling_errors",
    "calculate_error_statistics",
    "corrupt_word",
    "should_corrupt_word",
]
