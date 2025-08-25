"""LLM provider implementations for language module."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
import asyncio

import openai
import anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from nitroagi.utils.config import get_config
from nitroagi.utils.logging import get_logger


config = get_config()
logger = get_logger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text stream from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Generated text chunks
        """
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
        """
        self.api_key = api_key or config.ai_models.openai_api_key
        self.model = model
        self.client = None
        
        if self.api_key:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> str:
        """Generate text using OpenAI API."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                **kwargs
            )
            
            if stream:
                # For streaming, use generate_stream instead
                raise ValueError("Use generate_stream for streaming responses")
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using OpenAI API."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except:
            # Rough estimate if tiktoken not available
            return len(text.split()) * 1.3


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        """Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model: Model to use
        """
        self.api_key = api_key or config.ai_models.anthropic_api_key
        self.model = model
        self.client = None
        
        if self.api_key:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> str:
        """Generate text using Anthropic API."""
        if not self.client:
            raise ValueError("Anthropic API key not configured")
        
        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using Anthropic API."""
        if not self.client:
            raise ValueError("Anthropic API key not configured")
        
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens (approximate for Claude)."""
        # Rough estimate for Claude
        return len(text.split()) * 1.2


class HuggingFaceProvider(LLMProvider):
    """HuggingFace local model provider."""
    
    def __init__(self, model_name: str = "microsoft/phi-2"):
        """Initialize HuggingFace provider.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    async def _load_model(self):
        """Load model and tokenizer."""
        if self.model is None:
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> str:
        """Generate text using local HuggingFace model."""
        await self._load_model()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=kwargs.get("top_p", 0.9),
                    **kwargs
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the generated text
            generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using local HuggingFace model."""
        # Simulate streaming by chunking the output
        full_text = await self.generate(prompt, max_tokens, temperature, **kwargs)
        
        # Yield in chunks
        chunk_size = 10  # words
        words = full_text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming delay
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer."""
        await self._load_model()
        return len(self.tokenizer.encode(text))


class MultiProviderLLM(LLMProvider):
    """Multi-provider LLM with fallback support."""
    
    def __init__(self, providers: List[LLMProvider]):
        """Initialize multi-provider LLM.
        
        Args:
            providers: List of providers in priority order
        """
        self.providers = providers
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> str:
        """Generate text using first available provider."""
        for provider in self.providers:
            try:
                return await provider.generate(prompt, max_tokens, temperature, stream, **kwargs)
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed: {e}")
                continue
        
        raise RuntimeError("All LLM providers failed")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using first available provider."""
        for provider in self.providers:
            try:
                async for chunk in provider.generate_stream(prompt, max_tokens, temperature, **kwargs):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} streaming failed: {e}")
                continue
        
        raise RuntimeError("All LLM providers failed for streaming")
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using first available provider."""
        for provider in self.providers:
            try:
                return await provider.count_tokens(text)
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} token counting failed: {e}")
                continue
        
        # Fallback to word count estimate
        return len(text.split()) * 1.3