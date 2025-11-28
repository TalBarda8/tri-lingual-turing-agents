"""
Unit tests for translation agents module.

Tests the TranslationAgent building block with focus on validation
and error handling (not actual API calls).
"""

import pytest
from tri_lingual_agents.agents import TranslationAgent, create_agent_pipeline


class TestTranslationAgentValidation:
    """Test suite for TranslationAgent input validation."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        agent = TranslationAgent('en', 'fr', provider='anthropic')

        assert agent.source_lang == 'en'
        assert agent.target_lang == 'fr'
        assert agent.provider == 'anthropic'
        assert agent.model is not None

    def test_init_all_language_combinations(self):
        """Test all valid language pair combinations."""
        valid_pairs = [
            ('en', 'fr'),
            ('en', 'he'),
            ('fr', 'en'),
            ('fr', 'he'),
            ('he', 'en'),
            ('he', 'fr'),
        ]

        for source, target in valid_pairs:
            agent = TranslationAgent(source, target)
            assert agent.source_lang == source
            assert agent.target_lang == target

    def test_init_both_providers(self):
        """Test initialization with both providers."""
        agent_anthropic = TranslationAgent('en', 'fr', provider='anthropic')
        assert agent_anthropic.provider == 'anthropic'

        agent_openai = TranslationAgent('en', 'fr', provider='openai')
        assert agent_openai.provider == 'openai'

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        agent = TranslationAgent('en', 'fr', model='custom-model')
        assert agent.model == 'custom-model'

    # === INPUT VALIDATION TESTS ===

    def test_init_invalid_type_source_lang(self):
        """Test that non-string source_lang raises TypeError."""
        with pytest.raises(TypeError, match="source_lang must be str"):
            TranslationAgent(123, 'fr')

    def test_init_invalid_type_target_lang(self):
        """Test that non-string target_lang raises TypeError."""
        with pytest.raises(TypeError, match="target_lang must be str"):
            TranslationAgent('en', 123)

    def test_init_invalid_type_provider(self):
        """Test that non-string provider raises TypeError."""
        with pytest.raises(TypeError, match="provider must be str"):
            TranslationAgent('en', 'fr', provider=123)

    def test_init_invalid_source_lang(self):
        """Test that invalid source language raises ValueError."""
        with pytest.raises(ValueError, match="source_lang must be one of"):
            TranslationAgent('spanish', 'fr')

    def test_init_invalid_target_lang(self):
        """Test that invalid target language raises ValueError."""
        with pytest.raises(ValueError, match="target_lang must be one of"):
            TranslationAgent('en', 'german')

    def test_init_invalid_provider(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="provider must be one of"):
            TranslationAgent('en', 'fr', provider='google')

    def test_init_same_source_and_target(self):
        """Test that same source and target language raises ValueError."""
        with pytest.raises(ValueError, match="source_lang and target_lang cannot be the same"):
            TranslationAgent('en', 'en')

    def test_translate_invalid_type_text(self):
        """Test that non-string text raises TypeError."""
        agent = TranslationAgent('en', 'fr')

        with pytest.raises(TypeError, match="text must be str"):
            agent.translate(123)

    def test_translate_invalid_type_handle_errors(self):
        """Test that non-bool handle_errors raises TypeError."""
        agent = TranslationAgent('en', 'fr')

        with pytest.raises(TypeError, match="handle_errors must be bool"):
            agent.translate("Hello", handle_errors="yes")

    def test_translate_invalid_type_max_retries(self):
        """Test that non-int max_retries raises TypeError."""
        agent = TranslationAgent('en', 'fr')

        with pytest.raises(TypeError, match="max_retries must be int"):
            agent.translate("Hello", max_retries="3")

    def test_translate_empty_text(self):
        """Test that empty text raises ValueError."""
        agent = TranslationAgent('en', 'fr')

        with pytest.raises(ValueError, match="text cannot be empty"):
            agent.translate("")

    def test_translate_whitespace_only(self):
        """Test that whitespace-only text raises ValueError."""
        agent = TranslationAgent('en', 'fr')

        with pytest.raises(ValueError, match="text cannot be empty"):
            agent.translate("   ")

    def test_translate_max_retries_too_low(self):
        """Test that max_retries < 1 raises ValueError."""
        agent = TranslationAgent('en', 'fr')

        with pytest.raises(ValueError, match="max_retries must be at least 1"):
            agent.translate("Hello", max_retries=0)

    def test_translate_max_retries_too_high(self):
        """Test that max_retries > 10 raises ValueError."""
        agent = TranslationAgent('en', 'fr')

        with pytest.raises(ValueError, match="max_retries cannot exceed 10"):
            agent.translate("Hello", max_retries=20)


class TestTranslationAgentMethods:
    """Test suite for TranslationAgent methods."""

    def test_get_prompt_basic(self):
        """Test prompt generation without error handling."""
        agent = TranslationAgent('en', 'fr')
        prompt = agent._get_prompt("Hello world", handle_errors=False)

        assert "English" in prompt
        assert "French" in prompt
        assert "Hello world" in prompt
        assert "spelling errors" not in prompt.lower()

    def test_get_prompt_with_error_handling(self):
        """Test prompt generation with error handling."""
        agent = TranslationAgent('en', 'fr')
        prompt = agent._get_prompt("Hello world", handle_errors=True)

        assert "spelling errors" in prompt.lower()
        assert "Hello world" in prompt

    def test_repr(self):
        """Test string representation of agent."""
        agent = TranslationAgent('en', 'fr', provider='anthropic')
        repr_str = repr(agent)

        assert 'en' in repr_str
        assert 'fr' in repr_str
        assert 'anthropic' in repr_str


class TestCreateAgentPipeline:
    """Test suite for create_agent_pipeline function."""

    def test_create_pipeline_default(self):
        """Test pipeline creation with default parameters."""
        agent1, agent2, agent3 = create_agent_pipeline()

        assert agent1.source_lang == 'en'
        assert agent1.target_lang == 'fr'
        assert agent2.source_lang == 'fr'
        assert agent2.target_lang == 'he'
        assert agent3.source_lang == 'he'
        assert agent3.target_lang == 'en'

    def test_create_pipeline_with_provider(self):
        """Test pipeline creation with specific provider."""
        agent1, agent2, agent3 = create_agent_pipeline(provider='openai')

        assert all(agent.provider == 'openai' for agent in [agent1, agent2, agent3])

    def test_create_pipeline_with_model(self):
        """Test pipeline creation with custom model."""
        agent1, agent2, agent3 = create_agent_pipeline(model='custom-model')

        assert all(agent.model == 'custom-model' for agent in [agent1, agent2, agent3])

    def test_create_pipeline_returns_three_agents(self):
        """Test that pipeline returns exactly three agents."""
        result = create_agent_pipeline()

        assert len(result) == 3
        assert all(isinstance(agent, TranslationAgent) for agent in result)


# Note: Actual translation tests with API calls are excluded as they would
# require real API keys and network access. Those should be tested separately
# in integration tests or with proper mocking.
