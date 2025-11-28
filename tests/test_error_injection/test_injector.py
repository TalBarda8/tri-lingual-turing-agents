"""
Unit tests for error injection module.

Tests the SpellingErrorInjector building block with comprehensive coverage
of normal operation, edge cases, and error handling.
"""

import pytest
from tri_lingual_agents.error_injection import (
    inject_spelling_errors,
    calculate_error_statistics,
    corrupt_word,
    should_corrupt_word,
)


class TestInjectSpellingErrors:
    """Test suite for inject_spelling_errors function."""

    def test_inject_errors_zero_rate(self):
        """Test with 0% error rate - no corruption should occur."""
        sentence = "The quick brown fox jumps over the lazy dog"
        corrupted, words = inject_spelling_errors(sentence, 0.0, seed=42)

        assert corrupted == sentence  # No changes
        assert len(words) == 0  # No words corrupted

    def test_inject_errors_full_rate(self):
        """Test with 100% error rate - all eligible words corrupted."""
        sentence = "The quick brown fox jumps"
        corrupted, words = inject_spelling_errors(sentence, 1.0, seed=42)

        assert corrupted != sentence  # Should be different
        assert len(words) > 0  # Some words should be corrupted

    def test_inject_errors_partial_rate(self):
        """Test with 50% error rate."""
        sentence = "The remarkable transformation of artificial intelligence"
        corrupted, words = inject_spelling_errors(sentence, 0.5, seed=42)

        assert corrupted != sentence
        assert len(words) > 0
        # Should corrupt roughly half the eligible words
        assert len(words) >= 2  # At least some words corrupted

    def test_inject_errors_reproducible_with_seed(self):
        """Test that same seed produces same results."""
        sentence = "The quick brown fox jumps over the lazy dog"

        result1, words1 = inject_spelling_errors(sentence, 0.5, seed=42)
        result2, words2 = inject_spelling_errors(sentence, 0.5, seed=42)

        assert result1 == result2  # Reproducible
        assert words1 == words2

    def test_inject_errors_different_with_different_seed(self):
        """Test that different seeds produce different results."""
        sentence = "The quick brown fox jumps over the lazy dog"

        result1, _ = inject_spelling_errors(sentence, 0.5, seed=42)
        result2, _ = inject_spelling_errors(sentence, 0.5, seed=99)

        assert result1 != result2  # Different results

    def test_inject_errors_preserves_structure(self):
        """Test that sentence structure is preserved."""
        sentence = "Hello, world! How are you today?"
        corrupted, _ = inject_spelling_errors(sentence, 0.5, seed=42)

        # Should have same number of punctuation
        assert corrupted.count(',') == sentence.count(',')
        assert corrupted.count('!') == sentence.count('!')
        assert corrupted.count('?') == sentence.count('?')

    # === INPUT VALIDATION TESTS ===

    def test_inject_errors_invalid_type_sentence(self):
        """Test that non-string sentence raises TypeError."""
        with pytest.raises(TypeError, match="sentence must be str"):
            inject_spelling_errors(123, 0.5)

    def test_inject_errors_invalid_type_error_rate(self):
        """Test that non-numeric error_rate raises TypeError."""
        with pytest.raises(TypeError, match="error_rate must be numeric"):
            inject_spelling_errors("Hello world today", "invalid")

    def test_inject_errors_invalid_type_seed(self):
        """Test that non-int seed raises TypeError."""
        with pytest.raises(TypeError, match="seed must be int"):
            inject_spelling_errors("Hello world today", 0.5, seed="42")

    def test_inject_errors_error_rate_too_low(self):
        """Test that error_rate < 0 raises ValueError."""
        with pytest.raises(ValueError, match="error_rate must be in range"):
            inject_spelling_errors("Hello world today", -0.1)

    def test_inject_errors_error_rate_too_high(self):
        """Test that error_rate > 1 raises ValueError."""
        with pytest.raises(ValueError, match="error_rate must be in range"):
            inject_spelling_errors("Hello world today", 1.5)

    def test_inject_errors_empty_sentence(self):
        """Test that empty sentence raises ValueError."""
        with pytest.raises(ValueError, match="sentence cannot be empty"):
            inject_spelling_errors("", 0.5)

    def test_inject_errors_whitespace_only(self):
        """Test that whitespace-only sentence raises ValueError."""
        with pytest.raises(ValueError, match="sentence cannot be empty"):
            inject_spelling_errors("   ", 0.5)

    def test_inject_errors_too_few_words(self):
        """Test that sentence with < 3 words raises ValueError."""
        with pytest.raises(ValueError, match="must contain at least 3 words"):
            inject_spelling_errors("Hello world", 0.5)

    # === EDGE CASE TESTS ===

    def test_inject_errors_minimum_valid_sentence(self):
        """Test with exactly 3 words (minimum)."""
        sentence = "Hello world today"
        corrupted, words = inject_spelling_errors(sentence, 0.5, seed=42)

        assert isinstance(corrupted, str)
        assert isinstance(words, list)

    def test_inject_errors_long_sentence(self):
        """Test with a long sentence."""
        sentence = " ".join(["word"] * 50)  # 50 words
        corrupted, words = inject_spelling_errors(sentence, 0.2, seed=42)

        assert corrupted != sentence
        # Should corrupt roughly 20% of eligible words
        assert len(words) >= 5

    def test_inject_errors_with_punctuation(self):
        """Test sentence with various punctuation marks."""
        sentence = "Hello, world! How are you? I'm fine, thanks."
        corrupted, words = inject_spelling_errors(sentence, 0.3, seed=42)

        assert isinstance(corrupted, str)
        # Punctuation should be preserved
        assert ',' in corrupted
        assert '!' in corrupted
        assert '?' in corrupted

    def test_inject_errors_with_numbers(self):
        """Test sentence containing numbers."""
        sentence = "The year 2023 was great for research and development"
        corrupted, words = inject_spelling_errors(sentence, 0.3, seed=42)

        assert '2023' in corrupted  # Numbers should not be corrupted


class TestCorruptWord:
    """Test suite for corrupt_word function."""

    def test_corrupt_word_changes_word(self):
        """Test that corrupt_word modifies the word."""
        word = "hello"
        # Try multiple times since it's random
        results = set()
        for seed in range(10):
            # Can't set seed directly, but corruption should happen
            corrupted = corrupt_word(word)
            results.add(corrupted)

        # Should have at least some variation
        assert len(results) > 1 or list(results)[0] != word

    def test_corrupt_word_very_short_words(self):
        """Test that very short words (<= 2 chars) are not corrupted."""
        assert corrupt_word("a") == "a"
        assert corrupt_word("an") == "an"

    def test_corrupt_word_preserves_length_roughly(self):
        """Test that corrupted word length is similar to original."""
        word = "testing"
        corrupted = corrupt_word(word)

        # Length should be within reasonable range
        assert len(corrupted) >= len(word) - 1
        assert len(corrupted) <= len(word) + 1


class TestShouldCorruptWord:
    """Test suite for should_corrupt_word function."""

    def test_should_corrupt_long_words(self):
        """Test that long words are eligible for corruption."""
        assert should_corrupt_word("hello") is True
        assert should_corrupt_word("transformation") is True

    def test_should_not_corrupt_short_words(self):
        """Test that very short words are not corrupted."""
        assert should_corrupt_word("a") is False
        assert should_corrupt_word("an") is False
        assert should_corrupt_word("I") is False

    def test_should_not_corrupt_common_words(self):
        """Test that common short words are excluded."""
        excluded = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her']
        for word in excluded:
            assert should_corrupt_word(word) is False

    def test_should_not_corrupt_punctuation(self):
        """Test that punctuation-only is not corrupted."""
        assert should_corrupt_word("!!!") is False
        assert should_corrupt_word(",,,") is False


class TestCalculateErrorStatistics:
    """Test suite for calculate_error_statistics function."""

    def test_calculate_stats_basic(self):
        """Test basic statistics calculation."""
        original = "The quick brown fox"
        corrupted = "The qiuck brown fox"
        corrupted_words = ["quick → qiuck"]

        stats = calculate_error_statistics(original, corrupted, corrupted_words)

        assert stats['total_words'] == 4
        assert stats['corrupted_words'] == 1
        assert stats['error_rate_percent'] == 25.0
        assert stats['original_sentence'] == original
        assert stats['corrupted_sentence'] == corrupted

    def test_calculate_stats_no_errors(self):
        """Test statistics with no errors."""
        original = "The quick brown fox"
        corrupted = original
        corrupted_words = []

        stats = calculate_error_statistics(original, corrupted, corrupted_words)

        assert stats['corrupted_words'] == 0
        assert stats['error_rate_percent'] == 0.0

    def test_calculate_stats_all_corrupted(self):
        """Test statistics with all words corrupted."""
        original = "Hello world"
        corrupted = "Helo wrld"
        corrupted_words = ["Hello → Helo", "world → wrld"]

        stats = calculate_error_statistics(original, corrupted, corrupted_words)

        assert stats['corrupted_words'] == 2
        assert stats['error_rate_percent'] == 100.0

    # === INPUT VALIDATION TESTS ===

    def test_calculate_stats_invalid_type_original(self):
        """Test that non-string original raises TypeError."""
        with pytest.raises(TypeError, match="original must be str"):
            calculate_error_statistics(123, "test", [])

    def test_calculate_stats_invalid_type_corrupted(self):
        """Test that non-string corrupted raises TypeError."""
        with pytest.raises(TypeError, match="corrupted must be str"):
            calculate_error_statistics("test", 123, [])

    def test_calculate_stats_invalid_type_corrupted_words(self):
        """Test that non-list corrupted_words raises TypeError."""
        with pytest.raises(TypeError, match="corrupted_words must be list"):
            calculate_error_statistics("test", "test", "not a list")


# Integration test
class TestErrorInjectionIntegration:
    """Integration tests for the complete error injection workflow."""

    def test_full_workflow(self):
        """Test complete workflow from injection to statistics."""
        original = "The remarkable transformation of artificial intelligence systems"
        error_rate = 0.3

        # Inject errors
        corrupted, corrupted_words = inject_spelling_errors(original, error_rate, seed=42)

        # Calculate statistics
        stats = calculate_error_statistics(original, corrupted, corrupted_words)

        # Verify workflow
        assert stats['original_sentence'] == original
        assert stats['corrupted_sentence'] == corrupted
        assert len(stats['corrupted_list']) > 0
        assert 0 < stats['error_rate_percent'] <= 100

    def test_reproducibility_across_workflow(self):
        """Test that the entire workflow is reproducible."""
        sentence = "The quick brown fox jumps over the lazy dog"

        # Run 1
        corrupted1, words1 = inject_spelling_errors(sentence, 0.5, seed=42)
        stats1 = calculate_error_statistics(sentence, corrupted1, words1)

        # Run 2
        corrupted2, words2 = inject_spelling_errors(sentence, 0.5, seed=42)
        stats2 = calculate_error_statistics(sentence, corrupted2, words2)

        # Should be identical
        assert stats1 == stats2
