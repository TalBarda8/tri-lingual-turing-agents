"""
Spelling Error Injection Module

Building Block for injecting controlled spelling errors into text for robustness testing.

This module provides functionality to corrupt text with realistic spelling errors
at configurable rates, enabling systematic testing of LLM error handling.

Input Data:
- sentence: str - Original text to corrupt
- error_rate: float - Percentage of words to corrupt (0.0 to 1.0)

Output Data:
- corrupted_sentence: str - Text with spelling errors injected
- corrupted_words: List[str] - List of "original → corrupted" transformations

Setup Data:
- seed: int - Random seed for reproducibility (optional)
- min_word_length: int - Minimum word length to corrupt (default: 3)
"""

import random
import re
from typing import List, Tuple
import math


def corrupt_word(word: str) -> str:
    """
    Apply a random spelling error to a word.

    Corruption techniques:
    - Letter substitution (e.g., "hello" → "helo")
    - Letter transposition (e.g., "receive" → "recieve")
    - Letter omission (e.g., "spelling" → "speling")
    - Letter duplication (e.g., "error" → "errror")

    Args:
        word: The word to corrupt

    Returns:
        The corrupted word
    """
    if len(word) <= 2:
        return word  # Don't corrupt very short words

    techniques = ['substitute', 'transpose', 'omit', 'duplicate']
    technique = random.choice(techniques)

    word_chars = list(word)

    if technique == 'substitute':
        # Replace a random letter with another letter
        pos = random.randint(0, len(word_chars) - 1)
        # Choose a letter that's close on the keyboard or similar
        similar_chars = {
            'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr',
            'f': 'dg', 'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk',
            'k': 'jl', 'l': 'ko', 'm': 'n', 'n': 'bm', 'o': 'ip',
            'p': 'o', 'q': 'wa', 'r': 'et', 's': 'ad', 't': 'ry',
            'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu',
            'z': 'x'
        }
        char_lower = word_chars[pos].lower()
        if char_lower in similar_chars:
            replacement = random.choice(similar_chars[char_lower])
            if word_chars[pos].isupper():
                replacement = replacement.upper()
            word_chars[pos] = replacement

    elif technique == 'transpose':
        # Swap two adjacent letters
        if len(word_chars) > 2:
            pos = random.randint(0, len(word_chars) - 2)
            word_chars[pos], word_chars[pos + 1] = word_chars[pos + 1], word_chars[pos]

    elif technique == 'omit':
        # Remove a random letter
        if len(word_chars) > 3:
            pos = random.randint(0, len(word_chars) - 1)
            word_chars.pop(pos)

    elif technique == 'duplicate':
        # Duplicate a random letter
        pos = random.randint(0, len(word_chars) - 1)
        word_chars.insert(pos, word_chars[pos])

    return ''.join(word_chars)


def should_corrupt_word(word: str) -> bool:
    """
    Determine if a word should be eligible for corruption.

    Args:
        word: The word to check

    Returns:
        True if the word should be corrupted, False otherwise
    """
    # Skip very short words (articles, prepositions, etc.)
    if len(word) <= 2:
        return False

    # Skip punctuation-only "words"
    if not any(c.isalpha() for c in word):
        return False

    # Skip common short words that would be confusing if corrupted
    skip_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'has'}
    if word.lower() in skip_words:
        return False

    return True


def tokenize_sentence(sentence: str) -> List[Tuple[str, bool]]:
    """
    Tokenize a sentence into words and punctuation, preserving structure.

    Args:
        sentence: The sentence to tokenize

    Returns:
        List of (token, is_word) tuples where is_word indicates if it's a word vs punctuation
    """
    # Split on whitespace while preserving the structure
    tokens = []
    current_word = ""

    for char in sentence:
        if char.isalnum() or char == "'":
            current_word += char
        else:
            if current_word:
                tokens.append((current_word, True))
                current_word = ""
            if not char.isspace():
                tokens.append((char, False))
            elif tokens and char.isspace():
                # Preserve spacing
                tokens.append((char, False))

    if current_word:
        tokens.append((current_word, True))

    return tokens


def inject_spelling_errors(sentence: str, error_rate: float, seed: int = None) -> Tuple[str, List[str]]:
    """
    Inject spelling errors into a sentence at a specified rate.

    Building Block: Spelling Error Injector

    Input Data:
    - sentence: str - Original text to corrupt
    - error_rate: float - Percentage of words to corrupt (0.0 to 1.0)
    - seed: int - Random seed for reproducibility (optional)

    Output Data:
    - corrupted_sentence: str - Text with spelling errors
    - corrupted_words: List[str] - List of "original → corrupted" pairs

    Args:
        sentence: The original sentence
        error_rate: Percentage of words to corrupt (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (corrupted_sentence, list_of_corrupted_words)

    Raises:
        TypeError: If sentence is not a string or error_rate is not numeric
        ValueError: If sentence is empty or error_rate is out of range

    Example:
        >>> inject_spelling_errors("The quick brown fox", 0.5, seed=42)
        ("The qiuck browwn fox", ["quick → qiuck", "brown → browwn"])
    """
    # === INPUT VALIDATION ===
    _validate_inject_errors_input(sentence, error_rate, seed)

    if seed is not None:
        random.seed(seed)

    # Tokenize the sentence
    tokens = tokenize_sentence(sentence)

    # Extract words that are eligible for corruption
    words = [(i, token) for i, (token, is_word) in enumerate(tokens) if is_word and should_corrupt_word(token)]

    if len(words) == 0:
        return sentence, []

    # Calculate how many words to corrupt
    num_to_corrupt = math.ceil(error_rate * len(words))
    num_to_corrupt = min(num_to_corrupt, len(words))

    # Randomly select words to corrupt
    words_to_corrupt = random.sample(words, num_to_corrupt)

    # Track which words were corrupted
    corrupted_words = []

    # Apply corruption
    for idx, original_word in words_to_corrupt:
        corrupted = corrupt_word(original_word)
        tokens[idx] = (corrupted, True)
        corrupted_words.append(f"{original_word} → {corrupted}")

    # Reconstruct the sentence
    result = ""
    for i, (token, is_word) in enumerate(tokens):
        result += token

    return result, corrupted_words


def calculate_error_statistics(original: str, corrupted: str, corrupted_words: List[str]) -> dict:
    """
    Calculate statistics about the error injection.

    Args:
        original: Original sentence
        corrupted: Corrupted sentence
        corrupted_words: List of corrupted word mappings

    Returns:
        Dictionary with statistics

    Raises:
        TypeError: If inputs are not correct types
        ValueError: If inputs are invalid
    """
    # Input validation
    if not isinstance(original, str):
        raise TypeError(f"original must be str, got {type(original).__name__}")
    if not isinstance(corrupted, str):
        raise TypeError(f"corrupted must be str, got {type(corrupted).__name__}")
    if not isinstance(corrupted_words, list):
        raise TypeError(f"corrupted_words must be list, got {type(corrupted_words).__name__}")

    # Count words (simple whitespace split for word count)
    original_words = [w for w in re.findall(r'\b\w+\b', original)]
    total_words = len(original_words)
    num_corrupted = len(corrupted_words)

    actual_error_rate = (num_corrupted / total_words * 100) if total_words > 0 else 0

    return {
        'total_words': total_words,
        'corrupted_words': num_corrupted,
        'error_rate_percent': round(actual_error_rate, 2),
        'corrupted_list': corrupted_words,
        'original_sentence': original,
        'corrupted_sentence': corrupted
    }


def _validate_inject_errors_input(sentence: str, error_rate: float, seed: int = None):
    """
    Validate inputs for inject_spelling_errors function.

    Performs comprehensive validation including type checking, value range
    checking, and precondition checking.

    Args:
        sentence: Text to validate
        error_rate: Error rate to validate
        seed: Seed to validate

    Raises:
        TypeError: If types are incorrect
        ValueError: If values are out of valid ranges
    """
    # === TYPE CHECKING ===
    if not isinstance(sentence, str):
        raise TypeError(f"sentence must be str, got {type(sentence).__name__}")

    if not isinstance(error_rate, (int, float)):
        raise TypeError(f"error_rate must be numeric, got {type(error_rate).__name__}")

    if seed is not None and not isinstance(seed, int):
        raise TypeError(f"seed must be int or None, got {type(seed).__name__}")

    # === VALUE RANGE CHECKING ===
    if not 0.0 <= error_rate <= 1.0:
        raise ValueError(
            f"error_rate must be in range [0.0, 1.0], got {error_rate}"
        )

    # === PRECONDITION CHECKING ===
    if not sentence.strip():
        raise ValueError("sentence cannot be empty or whitespace only")

    # Check minimum word count (at least 3 words for meaningful corruption)
    words = re.findall(r'\b\w+\b', sentence)
    if len(words) < 3:
        raise ValueError(
            f"sentence must contain at least 3 words, got {len(words)}"
        )
