"""Base module implementation with common functionality."""

import asyncio
from typing import Any, Dict, Optional
import time

from nitroagi.core.base import AIModule, ModuleRequest, ModuleResponse, ModuleStatus
from nitroagi.utils.logging import get_logger, log_execution_time


class BaseAIModule(AIModule):
    """Base implementation of AI module with common functionality."""
    
    def __init__(self, config):
        """Initialize base module.
        
        Args:
            config: Module configuration
        """
        super().__init__(config)
        self.logger = get_logger(f"nitroagi.modules.{config.name}")
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the module."""
        self.logger.info(f"Initializing {self.config.name} module...")
        
        try:
            # Perform module-specific initialization
            await self._initialize_impl()
            
            self._initialized = True
            self.status = ModuleStatus.READY
            
            self.logger.info(f"{self.config.name} module initialized successfully")
            
        except Exception as e:
            self.status = ModuleStatus.ERROR
            self.logger.error(f"Failed to initialize {self.config.name}: {e}", exc_info=True)
            raise
    
    async def _initialize_impl(self) -> None:
        """Module-specific initialization (override in subclasses)."""
        pass
    
    @log_execution_time
    async def process(self, request: ModuleRequest) -> ModuleResponse:
        """Process a request.
        
        Args:
            request: The request to process
            
        Returns:
            Module response
        """
        if not self._initialized:
            raise RuntimeError(f"Module {self.config.name} not initialized")
        
        if self.status != ModuleStatus.READY:
            raise RuntimeError(f"Module {self.config.name} not ready: {self.status.value}")
        
        start_time = time.time()
        
        # Check cache
        cached_response = await self.get_cached_response(request)
        if cached_response:
            self.logger.debug(f"Returning cached response for request {request.context.request_id}")
            return cached_response
        
        # Validate request
        if not await self.validate_request(request):
            raise ValueError(f"Invalid request for module {self.config.name}")
        
        self.status = ModuleStatus.PROCESSING
        
        try:
            # Process with timeout
            timeout = request.timeout_override or self.config.timeout_seconds
            
            result = await asyncio.wait_for(
                self._process_impl(request),
                timeout=timeout
            )
            
            # Create response
            processing_time = (time.time() - start_time) * 1000
            
            response = ModuleResponse(
                request_id=request.context.request_id,
                module_name=self.config.name,
                status="success",
                data=result,
                processing_time_ms=processing_time,
                confidence_score=self._calculate_confidence(result),
                metadata=self._get_response_metadata(request, result)
            )
            
            # Cache response
            self.cache_response(request, response)
            
            # Update metrics
            self.update_metrics(processing_time)
            
            self.status = ModuleStatus.READY
            return response
            
        except asyncio.TimeoutError:
            processing_time = (time.time() - start_time) * 1000
            self.update_metrics(processing_time, error=True)
            self.status = ModuleStatus.READY
            
            return ModuleResponse(
                request_id=request.context.request_id,
                module_name=self.config.name,
                status="timeout",
                data=None,
                processing_time_ms=processing_time,
                error=f"Processing timeout after {timeout} seconds"
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.update_metrics(processing_time, error=True)
            self.status = ModuleStatus.ERROR
            
            self.logger.error(f"Error processing request: {e}", exc_info=True)
            
            return ModuleResponse(
                request_id=request.context.request_id,
                module_name=self.config.name,
                status="error",
                data=None,
                processing_time_ms=processing_time,
                error=str(e)
            )
    
    async def _process_impl(self, request: ModuleRequest) -> Any:
        """Module-specific processing (override in subclasses).
        
        Args:
            request: The request to process
            
        Returns:
            Processing result
        """
        raise NotImplementedError(f"Module {self.config.name} must implement _process_impl")
    
    def _calculate_confidence(self, result: Any) -> float:
        """Calculate confidence score for result.
        
        Args:
            result: Processing result
            
        Returns:
            Confidence score between 0 and 1
        """
        # Default implementation - override for specific logic
        return 0.95
    
    def _get_response_metadata(self, request: ModuleRequest, result: Any) -> Dict[str, Any]:
        """Get metadata for response.
        
        Args:
            request: Original request
            result: Processing result
            
        Returns:
            Response metadata
        """
        return {
            "module_version": self.config.version,
            "capabilities_used": [cap.value for cap in request.required_capabilities],
            "cache_hit": False,
        }
    
    async def shutdown(self) -> None:
        """Shutdown the module."""
        self.logger.info(f"Shutting down {self.config.name} module...")
        
        try:
            # Perform module-specific shutdown
            await self._shutdown_impl()
            
            self.status = ModuleStatus.SHUTDOWN
            self._initialized = False
            
            self.logger.info(f"{self.config.name} module shut down successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
            raise
    
    async def _shutdown_impl(self) -> None:
        """Module-specific shutdown (override in subclasses)."""
        pass