"""Unit tests for language module."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json

from nitroagi.modules.language.language_module import LanguageModule
from nitroagi.modules.language.providers import (
    OpenAIProvider,
    AnthropicProvider,
    HuggingFaceProvider,
    MultiProviderLLM
)
from nitroagi.core.base import ModuleRequest, ModuleCapability


class TestLanguageModule:
    """Test LanguageModule functionality."""
    
    @pytest.mark.asyncio
    async def test_module_initialization(self, module_config):
        """Test language module initialization."""
        module = LanguageModule(module_config)
        
        with patch.object(module, 'llm_provider', AsyncMock()):
            await module.initialize()
            
            assert module._initialized is True
            assert module.llm_provider is not None
    
    @pytest.mark.asyncio
    async def test_text_generation(self, module_config, create_module_request, mock_llm_provider):
        """Test text generation capability."""
        module = LanguageModule(module_config)
        module.llm_provider = mock_llm_provider
        module._initialized = True
        
        request = create_module_request(
            data="Generate a story about AI",
            capabilities=[ModuleCapability.TEXT_GENERATION]
        )
        
        response = await module.process(request)
        
        assert response.status == "success"
        assert response.data == "Generated text response"
        assert response.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_text_understanding(self, module_config, create_module_request):
        """Test text understanding capability."""
        module = LanguageModule(module_config)
        
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(
            return_value='{"topic": "AI", "sentiment": "positive", "entities": ["AI"], "summary": "Text about AI"}'
        )
        module.llm_provider = mock_provider
        module._initialized = True
        
        request = create_module_request(
            data="Analyze this text about artificial intelligence",
            capabilities=[ModuleCapability.TEXT_UNDERSTANDING]
        )
        
        response = await module.process(request)
        
        assert response.status == "success"
        assert isinstance(response.data, dict)
        assert "topic" in response.data
        assert "sentiment" in response.data
    
    @pytest.mark.asyncio
    async def test_conversation_history(self, module_config, create_module_request):
        """Test conversation history management."""
        module = LanguageModule(module_config)
        module.llm_provider = AsyncMock()
        module.llm_provider.generate = AsyncMock(return_value="Response")
        module._initialized = True
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        request = create_module_request(
            data={"messages": messages},
            capabilities=[ModuleCapability.TEXT_GENERATION]
        )
        
        await module.process(request)
        
        # Check conversation history was updated
        conv_id = request.context.conversation_id
        assert conv_id in module.conversation_history
        assert len(module.conversation_history[conv_id]) == len(messages)
    
    @pytest.mark.asyncio
    async def test_prompt_enhancement(self, module_config):
        """Test prompt enhancement functionality."""
        module = LanguageModule(module_config)
        
        simple_prompt = "Tell me about AI"
        enhanced = module._enhance_prompt(simple_prompt)
        
        assert "NitroAGI" in enhanced
        assert simple_prompt in enhanced
        assert len(enhanced) > len(simple_prompt)
    
    @pytest.mark.asyncio
    async def test_response_post_processing(self, module_config):
        """Test response post-processing."""
        module = LanguageModule(module_config)
        
        raw_response = "Assistant: Here is the response"
        processed = module._post_process_response(raw_response)
        
        assert not processed.startswith("Assistant:")
        assert processed == "Here is the response"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, module_config, create_module_request):
        """Test error handling in language module."""
        module = LanguageModule(module_config)
        module.llm_provider = AsyncMock()
        module.llm_provider.generate = AsyncMock(side_effect=Exception("API Error"))
        module._initialized = True
        
        request = create_module_request(data="Test prompt")
        
        response = await module.process(request)
        
        assert response.status == "error"
        assert "API Error" in response.error


class TestOpenAIProvider:
    """Test OpenAI provider."""
    
    @pytest.mark.asyncio
    async def test_openai_generation(self, mock_openai_client):
        """Test OpenAI text generation."""
        provider = OpenAIProvider(api_key="test-key")
        provider.client = mock_openai_client
        
        result = await provider.generate("Test prompt")
        
        assert result == "OpenAI response"
        mock_openai_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_openai_token_counting(self):
        """Test OpenAI token counting."""
        provider = OpenAIProvider(api_key="test-key")
        
        # Fallback token counting (without tiktoken)
        count = await provider.count_tokens("This is a test text")
        assert count > 0


class TestAnthropicProvider:
    """Test Anthropic provider."""
    
    @pytest.mark.asyncio
    async def test_anthropic_generation(self, mock_anthropic_client):
        """Test Anthropic text generation."""
        provider = AnthropicProvider(api_key="test-key")
        provider.client = mock_anthropic_client
        
        result = await provider.generate("Test prompt")
        
        assert result == "Anthropic response"
        mock_anthropic_client.messages.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_anthropic_token_counting(self):
        """Test Anthropic token counting."""
        provider = AnthropicProvider(api_key="test-key")
        
        count = await provider.count_tokens("This is a test text")
        assert count > 0


class TestHuggingFaceProvider:
    """Test HuggingFace provider."""
    
    @pytest.mark.asyncio
    @patch('nitroagi.modules.language.providers.AutoModelForCausalLM')
    @patch('nitroagi.modules.language.providers.AutoTokenizer')
    async def test_huggingface_generation(self, mock_tokenizer, mock_model):
        """Test HuggingFace text generation."""
        # Setup mocks
        tokenizer_instance = MagicMock()
        tokenizer_instance.return_value = {"input_ids": MagicMock()}
        tokenizer_instance.decode = MagicMock(return_value="Test promptGenerated text")
        mock_tokenizer.from_pretrained.return_value = tokenizer_instance
        
        model_instance = MagicMock()
        model_instance.generate = MagicMock(return_value=[[1, 2, 3]])
        mock_model.from_pretrained.return_value = model_instance
        
        provider = HuggingFaceProvider()
        
        result = await provider.generate("Test prompt")
        
        assert result == "Generated text"
    
    @pytest.mark.asyncio
    async def test_huggingface_streaming(self):
        """Test HuggingFace streaming generation."""
        provider = HuggingFaceProvider()
        provider.generate = AsyncMock(return_value="This is a long generated text")
        
        chunks = []
        async for chunk in provider.generate_stream("Test prompt"):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0


class TestMultiProviderLLM:
    """Test multi-provider LLM."""
    
    @pytest.mark.asyncio
    async def test_provider_fallback(self):
        """Test fallback between providers."""
        # Create providers with different behaviors
        provider1 = AsyncMock()
        provider1.generate = AsyncMock(side_effect=Exception("Provider 1 failed"))
        
        provider2 = AsyncMock()
        provider2.generate = AsyncMock(return_value="Provider 2 response")
        
        multi_provider = MultiProviderLLM([provider1, provider2])
        
        result = await multi_provider.generate("Test prompt")
        
        assert result == "Provider 2 response"
        provider1.generate.assert_called_once()
        provider2.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_all_providers_fail(self):
        """Test when all providers fail."""
        provider1 = AsyncMock()
        provider1.generate = AsyncMock(side_effect=Exception("Provider 1 failed"))
        
        provider2 = AsyncMock()
        provider2.generate = AsyncMock(side_effect=Exception("Provider 2 failed"))
        
        multi_provider = MultiProviderLLM([provider1, provider2])
        
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await multi_provider.generate("Test prompt")