"""
OpenRouter client for fallback inference and evaluation.
Provides a simple interface to OpenRouter API with automatic fallback.
"""

import os
import time
import json
import httpx
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import asyncio
from pathlib import Path

from .logging_config import get_logger

logger = get_logger("openrouter")


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter API."""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openrouter/auto"
    api_key: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9


class OpenRouterClient:
    """Client for OpenRouter API with retry logic and error handling."""
    
    def __init__(self, config: Optional[OpenRouterConfig] = None):
        """
        Initialize OpenRouter client.
        
        Args:
            config: OpenRouter configuration
        """
        self.config = config or OpenRouterConfig()
        
        # Override with environment variables
        self.config.api_key = os.getenv("OPENROUTER_API_KEY", self.config.api_key)
        self.config.base_url = os.getenv("OPENROUTER_BASE_URL", self.config.base_url)
        self.config.model = os.getenv("OPENROUTER_MODEL", self.config.model)
        
        # Initialize HTTP client
        self.client = httpx.Client(
            timeout=self.config.timeout,
            headers=self._get_headers()
        )
        
        logger.info(f"Initialized OpenRouter client with model: {self.config.model}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "LLM-Finetuning-Pipeline/1.0"
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        return headers
    
    def is_available(self) -> bool:
        """
        Check if OpenRouter API is available.
        
        Returns:
            True if API is available, False otherwise
        """
        try:
            response = self.client.get(f"{self.config.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"OpenRouter API not available: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.
        
        Returns:
            List of available models
        """
        try:
            response = self.client.get(f"{self.config.base_url}/models")
            response.raise_for_status()
            
            data = response.json()
            models = data.get("data", [])
            
            logger.info(f"Found {len(models)} available models")
            return models
        
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def complete_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        model: Optional[str] = None
    ) -> Optional[str]:
        """
        Complete text using OpenRouter API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            model: Model to use (overrides default)
        
        Returns:
            Generated text or None if failed
        """
        # Prepare request data
        request_data = {
            "model": model or self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": top_p if top_p is not None else self.config.top_p
        }
        
        if stop:
            request_data["stop"] = stop if isinstance(stop, list) else [stop]
        
        # Make request with retries
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Making OpenRouter request (attempt {attempt + 1})")
                
                response = self.client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=request_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "choices" in data and len(data["choices"]) > 0:
                        message = data["choices"][0].get("message", {})
                        content = message.get("content", "")
                        
                        logger.debug(f"Generated {len(content)} characters")
                        return content.strip()
                    else:
                        logger.warning("No choices in OpenRouter response")
                        return None
                
                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                    if attempt == self.config.max_retries - 1:
                        return None
                    time.sleep(1)
            
            except httpx.TimeoutException:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
            
            except Exception as e:
                logger.error(f"OpenRouter request failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    def batch_complete(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        delay_between_requests: float = 0.5
    ) -> List[Optional[str]]:
        """
        Complete multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per completion
            temperature: Sampling temperature
            delay_between_requests: Delay between API calls to avoid rate limits
        
        Returns:
            List of generated texts (None for failed requests)
        """
        logger.info(f"Processing {len(prompts)} prompts in batch")
        
        results = []
        for i, prompt in enumerate(prompts):
            logger.debug(f"Processing prompt {i + 1}/{len(prompts)}")
            
            result = self.complete_text(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            results.append(result)
            
            # Add delay to avoid rate limits
            if i < len(prompts) - 1 and delay_between_requests > 0:
                time.sleep(delay_between_requests)
        
        success_rate = sum(1 for r in results if r is not None) / len(results) * 100
        logger.info(f"Batch completion finished with {success_rate:.1f}% success rate")
        
        return results
    
    def close(self):
        """Close the HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class FallbackInferenceClient:
    """
    Inference client with automatic fallback from OpenAI to OpenRouter to local.
    """
    
    def __init__(
        self,
        local_model=None,
        local_tokenizer=None,
        openai_api_key: Optional[str] = None,
        openrouter_config: Optional[OpenRouterConfig] = None
    ):
        """
        Initialize fallback client.
        
        Args:
            local_model: Local model for fallback
            local_tokenizer: Local tokenizer for fallback
            openai_api_key: OpenAI API key
            openrouter_config: OpenRouter configuration
        """
        self.local_model = local_model
        self.local_tokenizer = local_tokenizer
        
        # OpenAI setup
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        
        if self.openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not available")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # OpenRouter setup
        self.openrouter_client = None
        if openrouter_config or os.getenv("OPENROUTER_BASE_URL"):
            self.openrouter_client = OpenRouterClient(openrouter_config)
            
            if self.openrouter_client.is_available():
                logger.info("OpenRouter client initialized and available")
            else:
                logger.warning("OpenRouter client initialized but not available")
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        prefer_provider: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate text with automatic fallback.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            prefer_provider: Preferred provider ("openai", "openrouter", "local")
        
        Returns:
            Generated text or None if all methods fail
        """
        providers = ["openai", "openrouter", "local"]
        
        # Reorder based on preference
        if prefer_provider and prefer_provider in providers:
            providers.remove(prefer_provider)
            providers.insert(0, prefer_provider)
        
        for provider in providers:
            try:
                if provider == "openai" and self.openai_client:
                    result = self._generate_openai(prompt, max_tokens, temperature)
                    if result:
                        logger.info(f"Generated text using OpenAI ({len(result)} chars)")
                        return result
                
                elif provider == "openrouter" and self.openrouter_client:
                    result = self.openrouter_client.complete_text(
                        prompt, max_tokens=max_tokens, temperature=temperature
                    )
                    if result:
                        logger.info(f"Generated text using OpenRouter ({len(result)} chars)")
                        return result
                
                elif provider == "local" and self.local_model and self.local_tokenizer:
                    result = self._generate_local(prompt, max_tokens, temperature)
                    if result:
                        logger.info(f"Generated text using local model ({len(result)} chars)")
                        return result
                
            except Exception as e:
                logger.warning(f"Failed to generate with {provider}: {e}")
                continue
        
        logger.error("All generation methods failed")
        return None
    
    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Generate text using OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if response.choices:
                return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"OpenAI generation failed: {e}")
        
        return None
    
    def _generate_local(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Generate text using local model."""
        try:
            import torch
            
            # Tokenize input
            inputs = self.local_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Generate
            device = next(self.local_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.local_tokenizer.pad_token_id,
                    eos_token_id=self.local_tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.local_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
        
        except Exception as e:
            logger.warning(f"Local generation failed: {e}")
        
        return None
    
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> List[Optional[str]]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
        
        Returns:
            List of generated texts
        """
        logger.info(f"Batch generating for {len(prompts)} prompts")
        
        results = []
        for prompt in prompts:
            result = self.generate_text(prompt, max_tokens, temperature)
            results.append(result)
        
        return results
    
    def close(self):
        """Close all clients."""
        if self.openrouter_client:
            self.openrouter_client.close()


# Utility functions

def create_fallback_client(
    openai_key: Optional[str] = None,
    openrouter_url: Optional[str] = None,
    local_model=None,
    local_tokenizer=None
) -> FallbackInferenceClient:
    """
    Create a fallback inference client with auto-configuration.
    
    Args:
        openai_key: OpenAI API key
        openrouter_url: OpenRouter base URL
        local_model: Local model
        local_tokenizer: Local tokenizer
    
    Returns:
        Configured fallback client
    """
    openrouter_config = None
    if openrouter_url or os.getenv("OPENROUTER_BASE_URL"):
        openrouter_config = OpenRouterConfig(
            base_url=openrouter_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    
    return FallbackInferenceClient(
        local_model=local_model,
        local_tokenizer=local_tokenizer,
        openai_api_key=openai_key,
        openrouter_config=openrouter_config
    )


def test_openrouter_connection() -> bool:
    """
    Test OpenRouter API connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with OpenRouterClient() as client:
            return client.is_available()
    except Exception as e:
        logger.error(f"Failed to test OpenRouter connection: {e}")
        return False