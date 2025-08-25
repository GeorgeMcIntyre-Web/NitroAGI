"""Language processing module implementation."""

from typing import Any, Dict, List, Optional
import json

from nitroagi.core.base import ModuleConfig, ModuleCapability, ModuleRequest
from nitroagi.modules.base import BaseAIModule
from nitroagi.modules.language.providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    HuggingFaceProvider,
    MultiProviderLLM
)
from nitroagi.utils.config import get_config
from nitroagi.utils.logging import get_logger


logger = get_logger(__name__)
config = get_config()


class LanguageModule(BaseAIModule):
    """Language processing module using LLMs."""
    
    def __init__(self, module_config: Optional[ModuleConfig] = None):
        """Initialize language module.
        
        Args:
            module_config: Module configuration
        """
        if module_config is None:
            module_config = ModuleConfig(
                name="language",
                version="1.0.0",
                description="Language processing module with multi-provider LLM support",
                capabilities=[
                    ModuleCapability.TEXT_GENERATION,
                    ModuleCapability.TEXT_UNDERSTANDING,
                ],
                max_workers=5,
                timeout_seconds=30.0,
                cache_enabled=True,
                cache_ttl_seconds=600,
            )
        
        super().__init__(module_config)
        self.llm_provider: Optional[LLMProvider] = None
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        self.max_history_length = 20
    
    async def _initialize_impl(self) -> None:
        """Initialize the language module."""
        # Initialize LLM providers based on available API keys
        providers = []
        
        if config.ai_models.openai_api_key:
            logger.info("Initializing OpenAI provider")
            providers.append(OpenAIProvider(
                api_key=config.ai_models.openai_api_key,
                model=config.ai_models.default_llm_model
            ))
        
        if config.ai_models.anthropic_api_key:
            logger.info("Initializing Anthropic provider")
            providers.append(AnthropicProvider(
                api_key=config.ai_models.anthropic_api_key
            ))
        
        # Always add HuggingFace as fallback
        logger.info("Initializing HuggingFace provider as fallback")
        providers.append(HuggingFaceProvider())
        
        if not providers:
            raise RuntimeError("No LLM providers available")
        
        # Use multi-provider with fallback
        self.llm_provider = MultiProviderLLM(providers)
        logger.info(f"Language module initialized with {len(providers)} providers")
    
    async def _process_impl(self, request: ModuleRequest) -> Any:
        """Process language request.
        
        Args:
            request: Module request
            
        Returns:
            Generated text or analysis result
        """
        data = request.data
        
        # Handle different input formats
        if isinstance(data, str):
            prompt = data
            messages = None
        elif isinstance(data, dict):
            prompt = data.get("prompt", "")
            messages = data.get("messages", None)
            temperature = data.get("temperature", config.ai_models.temperature)
            max_tokens = data.get("max_tokens", config.ai_models.max_tokens)
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
        
        # Build conversation context if messages provided
        if messages:
            prompt = self._build_prompt_from_messages(messages)
            
            # Store in conversation history
            conversation_id = request.context.conversation_id
            if conversation_id:
                self._update_conversation_history(conversation_id, messages)
        
        # Check for specific capabilities
        if ModuleCapability.TEXT_UNDERSTANDING in request.required_capabilities:
            # Text understanding/analysis task
            return await self._analyze_text(prompt)
        else:
            # Default to text generation
            return await self._generate_text(
                prompt,
                temperature=temperature if 'temperature' in locals() else config.ai_models.temperature,
                max_tokens=max_tokens if 'max_tokens' in locals() else config.ai_models.max_tokens
            )
    
    async def _generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate text using LLM provider.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if not self.llm_provider:
            raise RuntimeError("LLM provider not initialized")
        
        try:
            # Apply any prompt engineering
            enhanced_prompt = self._enhance_prompt(prompt)
            
            # Generate response
            response = await self.llm_provider.generate(
                enhanced_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=config.ai_models.top_p,
                frequency_penalty=config.ai_models.frequency_penalty,
                presence_penalty=config.ai_models.presence_penalty,
            )
            
            # Post-process response
            response = self._post_process_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for understanding.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        # Create analysis prompt
        analysis_prompt = f"""Analyze the following text and provide:
1. Main topic/theme
2. Sentiment (positive/negative/neutral)
3. Key entities mentioned
4. Summary (2-3 sentences)

Text: {text}

Provide the analysis in JSON format."""
        
        try:
            response = await self.llm_provider.generate(
                analysis_prompt,
                temperature=0.3,  # Lower temperature for analysis
                max_tokens=500
            )
            
            # Try to parse as JSON
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # Fallback to structured text parsing
                analysis = self._parse_analysis_response(response)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {
                "error": str(e),
                "topic": "unknown",
                "sentiment": "neutral",
                "entities": [],
                "summary": text[:200] + "..." if len(text) > 200 else text
            }
    
    def _build_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """Build prompt from conversation messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add instruction for response
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _enhance_prompt(self, prompt: str) -> str:
        """Enhance prompt with context and instructions.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Enhanced prompt
        """
        # Add system context if not already present
        if not prompt.startswith("System:") and not prompt.startswith("User:"):
            enhanced = f"""You are NitroAGI, an advanced AI assistant with multi-modal capabilities.
You are helpful, accurate, and concise in your responses.

User: {prompt}Assistant:"""
            return enhanced
        return prompt
    
    def _post_process_response(self, response: str) -> str:
        """Post-process generated response.
        
        Args:
            response: Generated response
            
        Returns:
            Processed response
        """
        # Remove any unwanted prefixes
        if response.startswith("Assistant:"):
            response = response[10:].strip()
        
        # Clean up whitespace
        response = response.strip()
        
        return response
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse analysis response from text format.
        
        Args:
            response: Text response
            
        Returns:
            Parsed analysis dictionary
        """
        analysis = {
            "topic": "unknown",
            "sentiment": "neutral",
            "entities": [],
            "summary": ""
        }
        
        lines = response.split("\n")
        for line in lines:
            lower_line = line.lower()
            if "topic" in lower_line or "theme" in lower_line:
                analysis["topic"] = line.split(":", 1)[-1].strip()
            elif "sentiment" in lower_line:
                if "positive" in lower_line:
                    analysis["sentiment"] = "positive"
                elif "negative" in lower_line:
                    analysis["sentiment"] = "negative"
                else:
                    analysis["sentiment"] = "neutral"
            elif "entities" in lower_line or "entity" in lower_line:
                entities_text = line.split(":", 1)[-1].strip()
                analysis["entities"] = [e.strip() for e in entities_text.split(",")]
            elif "summary" in lower_line:
                analysis["summary"] = line.split(":", 1)[-1].strip()
        
        return analysis
    
    def _update_conversation_history(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
        """Update conversation history.
        
        Args:
            conversation_id: Conversation identifier
            messages: Messages to add
        """
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        
        self.conversation_history[conversation_id].extend(messages)
        
        # Trim history if too long
        if len(self.conversation_history[conversation_id]) > self.max_history_length:
            self.conversation_history[conversation_id] = self.conversation_history[conversation_id][-self.max_history_length:]
    
    def _calculate_confidence(self, result: Any) -> float:
        """Calculate confidence score for language result.
        
        Args:
            result: Processing result
            
        Returns:
            Confidence score
        """
        # Base confidence
        confidence = 0.8
        
        if isinstance(result, str):
            # Check response quality indicators
            if len(result) > 50:
                confidence += 0.05
            if not result.endswith("..."):
                confidence += 0.05
            if result.count("\n") > 0:
                confidence += 0.05
        elif isinstance(result, dict):
            # For analysis results
            if "error" not in result:
                confidence = 0.9
            if result.get("entities") and len(result["entities"]) > 0:
                confidence += 0.05
        
        return min(confidence, 0.99)
    
    async def _shutdown_impl(self) -> None:
        """Cleanup on shutdown."""
        # Clear conversation history
        self.conversation_history.clear()
        
        # Clear any loaded models (for HuggingFace provider)
        if hasattr(self.llm_provider, 'providers'):
            for provider in self.llm_provider.providers:
                if isinstance(provider, HuggingFaceProvider):
                    if provider.model is not None:
                        del provider.model
                        provider.model = None
                    if provider.tokenizer is not None:
                        del provider.tokenizer
                        provider.tokenizer = None