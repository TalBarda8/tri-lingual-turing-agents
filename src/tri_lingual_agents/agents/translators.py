"""
Translation Agents Module

This module implements the translation agents for the tri-lingual pipeline.
Each agent handles translation between two specific languages using an LLM API.
"""

import os
from typing import Optional, Literal
import time


class TranslationAgent:
    """
    A translation agent that uses an LLM API to translate text between languages.

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
        """
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

        Args:
            text: Text to translate
            handle_errors: Whether to handle spelling errors (for EN→FR agent)
            max_retries: Maximum number of API retry attempts

        Returns:
            Translated text

        Raises:
            Exception: If translation fails after all retries
        """
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
