"""
OpenRouter API Client
Provides integration with OpenRouter's AI models
"""

import requests
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self, api_key: str, model: str = "90b"):
        self.api_url = f"https://api.openrouter.ai/v1/models/{model}"
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def call_model(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """
        Call OpenRouter API model.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens in response

        Returns:
            API response as dictionary

        Raises:
            Exception: If API call fails
        """
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API call failed: {str(e)}")
            raise Exception(f"Error connecting to OpenRouter API: {e}")
