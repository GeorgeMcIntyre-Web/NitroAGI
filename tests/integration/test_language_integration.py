"""Integration tests for language module with providers."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import os

from nitroagi.modules.language.language_module import LanguageModule
from nitroagi.modules.language.providers import (
    OpenAIProvider,
    AnthropicProvider,
    HuggingFaceProvider,
    MultiProviderLLM
)
from nitroagi.core.base import ModuleRequest, ModuleCapability


@pytest.mark.integration
@pytest.mark.network
class TestLanguageIntegration:
    """Integration tests for language module."""
    
    @pytest.mark.asyncio
    async def test_openai_integration(self):
        """Test OpenAI provider integration."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")
        
        provider = OpenAIProvider()
        
        result = await provider.generate(
            "Write a haiku about artificial intelligence",
            max_tokens=50,
            temperature=0.7
        )
        
        assert result is not None
        assert len(result) > 0
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_anthropic_integration(self):
        """Test Anthropic provider integration."""
        # Skip if no API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Anthropic API key not available")
        
        provider = AnthropicProvider()
        
        result = await provider.generate(
            "Explain quantum computing in one sentence",
            max_tokens=100,
            temperature=0.5
        )
        
        assert result is not None
        assert len(result) > 0
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_huggingface_integration(self):
        """Test HuggingFace provider integration."""
        provider = HuggingFaceProvider(model_name="microsoft/phi-2")
        
        # This will download the model if not cached
        result = await provider.generate(
            "The future of AI is",
            max_tokens=30,
            temperature=0.7
        )
        
        assert result is not None
        assert len(result) > 0
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_multi_provider_fallback(self):
        """Test multi-provider fallback mechanism."""
        # Create providers with mock failures
        provider1 = AsyncMock()
        provider1.generate = AsyncMock(side_effect=Exception("API limit reached"))
        
        provider2 = AsyncMock()
        provider2.generate = AsyncMock(side_effect=Exception("Service unavailable"))
        
        provider3 = AsyncMock()
        provider3.generate = AsyncMock(return_value="Fallback response")
        
        multi_provider = MultiProviderLLM([provider1, provider2, provider3])
        
        result = await multi_provider.generate("Test prompt")
        
        assert result == "Fallback response"
        assert provider1.generate.called
        assert provider2.generate.called
        assert provider3.generate.called
    
    @pytest.mark.asyncio
    async def test_language_module_full_flow(self, module_config, create_module_request):
        """Test complete language module flow."""
        module = LanguageModule(module_config)
        
        # Mock provider
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(
            return_value="This is a comprehensive response to your query."
        )
        module.llm_provider = mock_provider
        
        await module.initialize()
        
        # Test text generation
        gen_request = create_module_request(
            data={
                "prompt": "Tell me about AGI",
                "temperature": 0.7,
                "max_tokens": 100
            },
            capabilities=[ModuleCapability.TEXT_GENERATION]
        )
        
        gen_response = await module.process(gen_request)
        
        assert gen_response.status == "success"
        assert gen_response.data is not None
        assert len(gen_response.data) > 0
        
        # Test text analysis
        mock_provider.generate = AsyncMock(
            return_value='{"topic": "AGI", "sentiment": "neutral", "entities": ["AGI"], "summary": "Discussion about AGI"}'
        )
        
        analysis_request = create_module_request(
            data="Artificial General Intelligence represents the next frontier",
            capabilities=[ModuleCapability.TEXT_UNDERSTANDING]
        )
        
        analysis_response = await module.process(analysis_request)
        
        assert analysis_response.status == "success"
        assert isinstance(analysis_response.data, dict)
        assert analysis_response.data["topic"] == "AGI"
        
        await module.shutdown()
    
    @pytest.mark.asyncio
    async def test_conversation_context(self, module_config, create_module_request):
        """Test conversation context handling."""
        module = LanguageModule(module_config)
        
        mock_provider = AsyncMock()
        responses = [
            "Hello! I'm NitroAGI.",
            "AGI stands for Artificial General Intelligence.",
            "It aims to match human cognitive abilities."
        ]
        mock_provider.generate = AsyncMock(side_effect=responses)
        module.llm_provider = mock_provider
        
        await module.initialize()
        
        conversation_id = "test-conversation-123"
        
        # Multi-turn conversation
        messages_sequence = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "What is AGI?"}],
            [{"role": "user", "content": "Tell me more"}]
        ]
        
        for i, messages in enumerate(messages_sequence):
            request = create_module_request(
                data={"messages": messages},
                capabilities=[ModuleCapability.TEXT_GENERATION]
            )
            request.context.conversation_id = conversation_id
            
            response = await module.process(request)
            
            assert response.status == "success"
            assert response.data == responses[i]
        
        # Check conversation history
        assert conversation_id in module.conversation_history
        assert len(module.conversation_history[conversation_id]) > 0
        
        await module.shutdown()
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self):
        """Test streaming text generation."""
        provider = HuggingFaceProvider()
        provider.generate = AsyncMock(
            return_value="This is a long text that will be streamed in chunks."
        )
        
        chunks = []
        async for chunk in provider.generate_stream("Test prompt"):
            chunks.append(chunk)
            assert isinstance(chunk, str)
            assert len(chunk) > 0
        
        # Verify we got multiple chunks
        assert len(chunks) > 1
        
        # Verify complete text
        complete_text = "".join(chunks)
        assert len(complete_text) > 0
    
    @pytest.mark.asyncio
    async def test_token_counting(self):
        """Test token counting across providers."""
        text = "This is a sample text for counting tokens in different providers."
        
        # Test OpenAI provider
        openai_provider = OpenAIProvider(api_key="test")
        openai_count = await openai_provider.count_tokens(text)
        assert openai_count > 0
        
        # Test Anthropic provider
        anthropic_provider = AnthropicProvider(api_key="test")
        anthropic_count = await anthropic_provider.count_tokens(text)
        assert anthropic_count > 0
        
        # Counts should be relatively similar
        assert abs(openai_count - anthropic_count) < 5
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, module_config, create_module_request):
        """Test error recovery in language module."""
        module = LanguageModule(module_config)
        
        # Provider that fails then succeeds
        attempt_count = 0
        
        async def mock_generate(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise Exception("Temporary failure")
            return "Success after retry"
        
        mock_provider = AsyncMock()
        mock_provider.generate = mock_generate
        module.llm_provider = mock_provider
        
        await module.initialize()
        
        request = create_module_request(
            data="Test prompt",
            capabilities=[ModuleCapability.TEXT_GENERATION]
        )
        
        # First attempt should fail
        response = await module.process(request)
        assert response.status == "error"
        
        # Second attempt should succeed
        response = await module.process(request)
        assert response.status == "success"
        assert response.data == "Success after retry"
        
        await module.shutdown()