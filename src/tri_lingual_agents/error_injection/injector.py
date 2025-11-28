"""
Spelling Error Injection Module

This module provides functions to inject realistic spelling errors into English sentences
at controlled rates for robustness testing.
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

    Args:
        sentence: The original sentence
        error_rate: Percentage of words to corrupt (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (corrupted_sentence, list_of_corrupted_words)
    """
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
    """
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
