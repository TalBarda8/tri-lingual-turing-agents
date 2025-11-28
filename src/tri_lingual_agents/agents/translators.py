"""
Translation Agents Module

Building Block for LLM-based translation between language pairs.

This module provides translation agents that wrap LLM APIs to perform
translations between specific language pairs with error handling and retry logic.

Input Data:
- text: str - Text to translate
- handle_errors: bool - Whether to handle spelling errors in input

Output Data:
- translated_text: str - Translated text in target language

Setup Data:
- source_lang: str - Source language code (e.g., 'en', 'fr', 'he')
- target_lang: str - Target language code
- provider: str - LLM provider ('anthropic' or 'openai')
- model: str - Model name (optional, uses defaults if None)
- api_key: str - API key (optional, uses environment variable if None)
"""

import os
from typing import Optional, Literal
import time


class TranslationAgent:
    """
    A translation agent that uses an LLM API to translate text between languages.

    Building Block: Translation Agent

    Input Data:
    - text: str - Text to translate
    - handle_errors: bool - Whether to handle spelling errors

    Output Data:
    - translated_text: str - Translated text

    Setup Data:
    - source_lang: str - Source language code
    - target_lang: str - Target language code
    - provider: str - LLM provider
    - model: str - Model name
    - api_key: str - API credentials

    Attributes:
        source_lang (str): Source language code (e.g., 'en', 'fr', 'he')
        target_lang (str): Target language code (e.g., 'en', 'fr', 'he')
        model (str): The LLM model to use
        provider (str): The LLM provider ('anthropic' or 'openai')
    """

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        provider: Literal['anthropic', 'openai'] = 'anthropic',
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize a translation agent.

        Args:
            source_lang: Source language code
            target_lang: Target language code
            provider: LLM provider to use ('anthropic' or 'openai')
            model: Specific model name (uses default if None)
            api_key: API key (uses environment variable if None)

        Raises:
            TypeError: If arguments have incorrect types
            ValueError: If arguments have invalid values
        """
        # === INPUT VALIDATION ===
        self._validate_init_params(source_lang, target_lang, provider)

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.provider = provider

        # Set default models if not specified
        if model is None:
            if provider == 'anthropic':
                self.model = 'claude-3-5-sonnet-20241022'
            else:  # openai
                self.model = 'gpt-4'
        else:
            self.model = model

        # Initialize API client
        if provider == 'anthropic':
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
            )
        else:  # openai
            import openai
            self.client = openai.OpenAI(
                api_key=api_key or os.getenv('OPENAI_API_KEY')
            )

    def _get_prompt(self, text: str, handle_errors: bool = False) -> str:
        """
        Generate the translation prompt based on source and target languages.

        Args:
            text: Text to translate
            handle_errors: Whether to instruct the model to handle spelling errors

        Returns:
            The formatted prompt string
        """
        lang_names = {
            'en': 'English',
            'fr': 'French',
            'he': 'Hebrew'
        }

        source_name = lang_names.get(self.source_lang, self.source_lang)
        target_name = lang_names.get(self.target_lang, self.target_lang)

        if handle_errors and self.source_lang == 'en':
            return f"""Translate the following {source_name} text to {target_name}.
The text may contain spelling errors; please interpret the intended meaning and translate accurately.
Do not correct the spelling errors - just translate the intended meaning.

Text to translate:
{text}

Provide only the {target_name} translation, nothing else."""
        else:
            return f"""Translate the following {source_name} text to {target_name} accurately, preserving the meaning.

Text to translate:
{text}

Provide only the {target_name} translation, nothing else."""

    def translate(self, text: str, handle_errors: bool = False, max_retries: int = 3) -> str:
        """
        Translate text from source language to target language.

        Input Data:
        - text: str - Text to translate
        - handle_errors: bool - Whether to handle spelling errors
        - max_retries: int - Maximum retry attempts

        Output Data:
        - translated_text: str - Translated text in target language

        Args:
            text: Text to translate
            handle_errors: Whether to handle spelling errors (for EN→FR agent)
            max_retries: Maximum number of API retry attempts

        Returns:
            Translated text

        Raises:
            TypeError: If text is not a string
            ValueError: If text is empty or max_retries is invalid
            Exception: If translation fails after all retries
        """
        # === INPUT VALIDATION ===
        self._validate_translate_input(text, handle_errors, max_retries)

        prompt = self._get_prompt(text, handle_errors)

        for attempt in range(max_retries):
            try:
                if self.provider == 'anthropic':
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return response.content[0].text.strip()

                else:  # openai
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1024
                    )
                    return response.choices[0].message.content.strip()

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Translation attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Translation failed after {max_retries} attempts: {e}")

    def _validate_init_params(self, source_lang: str, target_lang: str, provider: str):
        """
        Validate initialization parameters.

        Args:
            source_lang: Source language to validate
            target_lang: Target language to validate
            provider: Provider to validate

        Raises:
            TypeError: If types are incorrect
            ValueError: If values are invalid
        """
        # Type checking
        if not isinstance(source_lang, str):
            raise TypeError(f"source_lang must be str, got {type(source_lang).__name__}")
        if not isinstance(target_lang, str):
            raise TypeError(f"target_lang must be str, got {type(target_lang).__name__}")
        if not isinstance(provider, str):
            raise TypeError(f"provider must be str, got {type(provider).__name__}")

        # Value checking
        valid_langs = {'en', 'fr', 'he'}
        if source_lang not in valid_langs:
            raise ValueError(f"source_lang must be one of {valid_langs}, got '{source_lang}'")
        if target_lang not in valid_langs:
            raise ValueError(f"target_lang must be one of {valid_langs}, got '{target_lang}'")

        valid_providers = {'anthropic', 'openai'}
        if provider not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}, got '{provider}'")

        # Precondition checking
        if source_lang == target_lang:
            raise ValueError(f"source_lang and target_lang cannot be the same: '{source_lang}'")

    def _validate_translate_input(self, text: str, handle_errors: bool, max_retries: int):
        """
        Validate translate method inputs.

        Args:
            text: Text to validate
            handle_errors: Boolean flag to validate
            max_retries: Retry count to validate

        Raises:
            TypeError: If types are incorrect
            ValueError: If values are invalid
        """
        # Type checking
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not isinstance(handle_errors, bool):
            raise TypeError(f"handle_errors must be bool, got {type(handle_errors).__name__}")
        if not isinstance(max_retries, int):
            raise TypeError(f"max_retries must be int, got {type(max_retries).__name__}")

        # Value checking
        if not text.strip():
            raise ValueError("text cannot be empty or whitespace only")
        if max_retries < 1:
            raise ValueError(f"max_retries must be at least 1, got {max_retries}")
        if max_retries > 10:
            raise ValueError(f"max_retries cannot exceed 10, got {max_retries}")

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"TranslationAgent({self.source_lang}→{self.target_lang}, {self.provider}:{self.model})"


def create_agent_pipeline(
    provider: Literal['anthropic', 'openai'] = 'anthropic',
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> tuple[TranslationAgent, TranslationAgent, TranslationAgent]:
    """
    Create the three-agent pipeline for the tri-lingual experiment.

    Args:
        provider: LLM provider to use
        model: Specific model name (uses default if None)
        api_key: API key (uses environment variable if None)

    Returns:
        Tuple of (agent_en_fr, agent_fr_he, agent_he_en)
    """
    agent1 = TranslationAgent('en', 'fr', provider, model, api_key)
    agent2 = TranslationAgent('fr', 'he', provider, model, api_key)
    agent3 = TranslationAgent('he', 'en', provider, model, api_key)

    return agent1, agent2, agent3
