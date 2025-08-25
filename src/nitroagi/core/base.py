"""Base classes and interfaces for NitroAGI modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
import asyncio
import logging
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class ModuleStatus(Enum):
    """Status of an AI module."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    MAINTENANCE = "maintenance"


class ModuleCapability(Enum):
    """Capabilities that modules can provide."""
    # Language capabilities
    TEXT_GENERATION = "text_generation"
    TEXT_UNDERSTANDING = "text_understanding"
    TRANSLATION = "translation"
    
    # Vision capabilities
    IMAGE_PROCESSING = "image_processing"
    IMAGE_GENERATION = "image_generation"
    IMAGE_UNDERSTANDING = "image_understanding"
    OBJECT_DETECTION = "object_detection"
    SCENE_ANALYSIS = "scene_analysis"
    TEXT_EXTRACTION = "text_extraction"  # OCR
    
    # Audio capabilities
    AUDIO_PROCESSING = "audio_processing"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    
    # Video capabilities
    VIDEO_PROCESSING = "video_processing"
    VIDEO_UNDERSTANDING = "video_understanding"
    
    # Cognitive capabilities
    REASONING = "reasoning"
    PLANNING = "planning"
    LOGICAL_REASONING = "logical_reasoning"
    CAUSAL_INFERENCE = "causal_inference"
    
    # Memory capabilities
    MEMORY_STORAGE = "memory_storage"
    MEMORY_RETRIEVAL = "memory_retrieval"
    
    # Learning capabilities
    LEARNING = "learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    
    # Tool capabilities
    TOOL_USE = "tool_use"


class ModuleConfig(BaseModel):
    """Configuration for an AI module."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(..., description="Module name")
    version: str = Field(default="1.0.0", description="Module version")
    description: str = Field(default="", description="Module description")
    capabilities: List[ModuleCapability] = Field(default_factory=list)
    max_workers: int = Field(default=1, ge=1, description="Maximum concurrent workers")
    timeout_seconds: float = Field(default=30.0, gt=0, description="Processing timeout")
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts")
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=300, ge=0, description="Cache TTL in seconds")
    resource_limits: Dict[str, Any] = Field(default_factory=dict)
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class ProcessingContext:
    """Context information for processing requests."""
    request_id: UUID = field(default_factory=uuid4)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_context: Optional['ProcessingContext'] = None
    
    def create_child_context(self) -> 'ProcessingContext':
        """Create a child context for sub-tasks."""
        return ProcessingContext(
            request_id=uuid4(),
            user_id=self.user_id,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            metadata={**self.metadata},
            parent_context=self
        )


@dataclass
class ModuleRequest:
    """Request to be processed by a module."""
    data: Any
    context: ProcessingContext = field(default_factory=ProcessingContext)
    priority: int = field(default=0)
    required_capabilities: List[ModuleCapability] = field(default_factory=list)
    timeout_override: Optional[float] = None


@dataclass
class ModuleResponse:
    """Response from a module after processing."""
    request_id: UUID
    module_name: str
    status: str
    data: Any
    processing_time_ms: float
    confidence_score: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIModule(ABC):
    """Abstract base class for all AI modules in NitroAGI."""
    
    def __init__(self, config: ModuleConfig):
        """Initialize the AI module.
        
        Args:
            config: Module configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"nitroagi.modules.{config.name}")
        self.status = ModuleStatus.INITIALIZING
        self._cache: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {
            "requests_processed": 0,
            "errors": 0,
            "total_processing_time_ms": 0.0,
            "average_processing_time_ms": 0.0,
        }
        self._shutdown_event = asyncio.Event()
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the module and load any required resources.
        
        This method should:
        - Load AI models
        - Set up connections
        - Prepare any required resources
        - Set status to READY when complete
        """
        pass
    
    @abstractmethod
    async def process(self, request: ModuleRequest) -> ModuleResponse:
        """Process a request and return a response.
        
        Args:
            request: The request to process
            
        Returns:
            ModuleResponse containing the results
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the module.
        
        This method should:
        - Save any state if needed
        - Close connections
        - Release resources
        - Set status to SHUTDOWN
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the module.
        
        Returns:
            Dictionary containing health status information
        """
        return {
            "module": self.config.name,
            "status": self.status.value,
            "version": self.config.version,
            "metrics": self._metrics.copy(),
            "capabilities": [cap.value for cap in self.config.capabilities],
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def supports_capability(self, capability: ModuleCapability) -> bool:
        """Check if the module supports a specific capability.
        
        Args:
            capability: The capability to check
            
        Returns:
            True if the capability is supported
        """
        return capability in self.config.capabilities
    
    async def validate_request(self, request: ModuleRequest) -> bool:
        """Validate a request before processing.
        
        Args:
            request: The request to validate
            
        Returns:
            True if the request is valid
        """
        # Check if module supports required capabilities
        for capability in request.required_capabilities:
            if not self.supports_capability(capability):
                self.logger.warning(
                    f"Module {self.config.name} does not support capability {capability.value}"
                )
                return False
        
        # Additional validation can be implemented by subclasses
        return True
    
    def update_metrics(self, processing_time_ms: float, error: bool = False) -> None:
        """Update module metrics.
        
        Args:
            processing_time_ms: Time taken to process the request
            error: Whether an error occurred
        """
        self._metrics["requests_processed"] += 1
        if error:
            self._metrics["errors"] += 1
        self._metrics["total_processing_time_ms"] += processing_time_ms
        self._metrics["average_processing_time_ms"] = (
            self._metrics["total_processing_time_ms"] / self._metrics["requests_processed"]
        )
    
    def get_cache_key(self, request: ModuleRequest) -> str:
        """Generate a cache key for a request.
        
        Args:
            request: The request to generate a key for
            
        Returns:
            Cache key string
        """
        # Simple implementation - can be overridden by subclasses
        import hashlib
        import json
        
        key_data = {
            "module": self.config.name,
            "data": str(request.data),
            "capabilities": [cap.value for cap in request.required_capabilities],
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get_cached_response(self, request: ModuleRequest) -> Optional[ModuleResponse]:
        """Get cached response if available.
        
        Args:
            request: The request to check cache for
            
        Returns:
            Cached response if available, None otherwise
        """
        if not self.config.cache_enabled:
            return None
            
        cache_key = self.get_cache_key(request)
        cached_data = self._cache.get(cache_key)
        
        if cached_data:
            # Check if cache is still valid
            cache_time = cached_data.get("timestamp", 0)
            current_time = datetime.utcnow().timestamp()
            if current_time - cache_time < self.config.cache_ttl_seconds:
                self.logger.debug(f"Cache hit for key {cache_key}")
                return cached_data.get("response")
        
        return None
    
    def cache_response(self, request: ModuleRequest, response: ModuleResponse) -> None:
        """Cache a response.
        
        Args:
            request: The original request
            response: The response to cache
        """
        if not self.config.cache_enabled:
            return
            
        cache_key = self.get_cache_key(request)
        self._cache[cache_key] = {
            "response": response,
            "timestamp": datetime.utcnow().timestamp(),
        }
        self.logger.debug(f"Cached response for key {cache_key}")
    
    async def wait_for_shutdown(self) -> None:
        """Wait for the shutdown signal."""
        await self._shutdown_event.wait()
    
    def signal_shutdown(self) -> None:
        """Signal that the module should shutdown."""
        self._shutdown_event.set()


class ModuleRegistry:
    """Registry for managing AI modules."""
    
    def __init__(self):
        self._modules: Dict[str, AIModule] = {}
        self._module_types: Dict[str, Type[AIModule]] = {}
        self.logger = logging.getLogger("nitroagi.core.registry")
    
    def register_module_type(self, name: str, module_class: Type[AIModule]) -> None:
        """Register a module type.
        
        Args:
            name: Name of the module type
            module_class: Class of the module
        """
        self._module_types[name] = module_class
        self.logger.info(f"Registered module type: {name}")
    
    async def create_module(self, name: str, config: ModuleConfig) -> AIModule:
        """Create and initialize a module.
        
        Args:
            name: Name/type of the module to create
            config: Configuration for the module
            
        Returns:
            Initialized module instance
        """
        if name not in self._module_types:
            raise ValueError(f"Unknown module type: {name}")
        
        module_class = self._module_types[name]
        module = module_class(config)
        await module.initialize()
        
        self._modules[config.name] = module
        self.logger.info(f"Created and initialized module: {config.name}")
        
        return module
    
    def get_module(self, name: str) -> Optional[AIModule]:
        """Get a module by name.
        
        Args:
            name: Name of the module
            
        Returns:
            Module instance if found, None otherwise
        """
        return self._modules.get(name)
    
    def get_modules_by_capability(self, capability: ModuleCapability) -> List[AIModule]:
        """Get all modules that support a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of modules supporting the capability
        """
        return [
            module for module in self._modules.values()
            if module.supports_capability(capability)
        ]
    
    async def shutdown_all(self) -> None:
        """Shutdown all registered modules."""
        self.logger.info("Shutting down all modules...")
        shutdown_tasks = [module.shutdown() for module in self._modules.values()]
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        self.logger.info("All modules shut down")
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all modules.
        
        Returns:
            Dictionary containing health status of all modules
        """
        health_tasks = {
            name: module.health_check()
            for name, module in self._modules.items()
        }
        
        results = {}
        for name, task in health_tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results