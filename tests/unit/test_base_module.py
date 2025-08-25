"""Unit tests for base module functionality."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import time

from nitroagi.modules.base import BaseAIModule
from nitroagi.core.base import (
    ModuleStatus,
    ModuleRequest,
    ModuleResponse,
    ModuleCapability
)


class TestModule(BaseAIModule):
    """Test implementation of BaseAIModule."""
    
    async def _initialize_impl(self) -> None:
        """Test initialization."""
        self.test_initialized = True
    
    async def _process_impl(self, request: ModuleRequest):
        """Test processing."""
        return {"processed": request.data}
    
    async def _shutdown_impl(self) -> None:
        """Test shutdown."""
        self.test_shutdown = True


class TestBaseAIModule:
    """Test BaseAIModule functionality."""
    
    @pytest.mark.asyncio
    async def test_module_initialization(self, module_config):
        """Test module initialization."""
        module = TestModule(module_config)
        
        assert module.status == ModuleStatus.INITIALIZING
        assert not module._initialized
        
        await module.initialize()
        
        assert module.status == ModuleStatus.READY
        assert module._initialized
        assert module.test_initialized
    
    @pytest.mark.asyncio
    async def test_module_processing(self, module_config, create_module_request):
        """Test module request processing."""
        module = TestModule(module_config)
        await module.initialize()
        
        request = create_module_request(data={"test": "data"})
        response = await module.process(request)
        
        assert response.status == "success"
        assert response.data == {"processed": {"test": "data"}}
        assert response.processing_time_ms > 0
        assert response.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_module_not_initialized_error(self, module_config, create_module_request):
        """Test error when processing without initialization."""
        module = TestModule(module_config)
        request = create_module_request()
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await module.process(request)
    
    @pytest.mark.asyncio
    async def test_module_timeout(self, module_config, create_module_request):
        """Test request timeout handling."""
        class SlowModule(TestModule):
            async def _process_impl(self, request):
                await asyncio.sleep(5)  # Longer than timeout
                return {"processed": request.data}
        
        module = SlowModule(module_config)
        await module.initialize()
        
        request = create_module_request(data={"test": "data"})
        request.timeout_override = 0.1  # 100ms timeout
        
        response = await module.process(request)
        
        assert response.status == "timeout"
        assert response.error is not None
        assert "timeout" in response.error.lower()
    
    @pytest.mark.asyncio
    async def test_module_error_handling(self, module_config, create_module_request):
        """Test error handling during processing."""
        class ErrorModule(TestModule):
            async def _process_impl(self, request):
                raise ValueError("Processing error")
        
        module = ErrorModule(module_config)
        await module.initialize()
        
        request = create_module_request()
        response = await module.process(request)
        
        assert response.status == "error"
        assert response.error == "Processing error"
        assert module.status == ModuleStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_module_caching(self, module_config, create_module_request):
        """Test response caching."""
        module = TestModule(module_config)
        module.config.cache_enabled = True
        await module.initialize()
        
        request = create_module_request(data={"test": "data"})
        
        # First call - no cache
        response1 = await module.process(request)
        
        # Mock cache hit
        module.get_cached_response = AsyncMock(return_value=response1)
        
        # Second call - should use cache
        response2 = await module.process(request)
        
        module.get_cached_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_module_shutdown(self, module_config):
        """Test module shutdown."""
        module = TestModule(module_config)
        await module.initialize()
        
        assert module.status == ModuleStatus.READY
        assert module._initialized
        
        await module.shutdown()
        
        assert module.status == ModuleStatus.SHUTDOWN
        assert not module._initialized
        assert module.test_shutdown
    
    @pytest.mark.asyncio
    async def test_module_metrics(self, module_config, create_module_request):
        """Test metrics collection."""
        module = TestModule(module_config)
        await module.initialize()
        
        # Process multiple requests
        for _ in range(3):
            request = create_module_request()
            await module.process(request)
        
        metrics = module.get_metrics()
        
        assert metrics["total_requests"] == 3
        assert metrics["successful_requests"] == 3
        assert metrics["failed_requests"] == 0
        assert metrics["average_processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_module_health_check(self, module_config):
        """Test health check functionality."""
        module = TestModule(module_config)
        await module.initialize()
        
        health = await module.health_check()
        
        assert health["status"] == "healthy"
        assert health["module"] == module_config.name
        assert "uptime" in health
        assert "total_requests" in health
    
    @pytest.mark.asyncio
    async def test_module_validation(self, module_config, create_module_request):
        """Test request validation."""
        module = TestModule(module_config)
        module.config.capabilities = [ModuleCapability.TEXT_GENERATION]
        await module.initialize()
        
        # Valid request
        valid_request = create_module_request(
            capabilities=[ModuleCapability.TEXT_GENERATION]
        )
        assert await module.validate_request(valid_request)
        
        # Invalid request - wrong capability
        invalid_request = create_module_request(
            capabilities=[ModuleCapability.IMAGE_GENERATION]
        )
        assert not await module.validate_request(invalid_request)
    
    def test_confidence_calculation(self, module_config):
        """Test confidence score calculation."""
        module = TestModule(module_config)
        
        # Default confidence
        confidence = module._calculate_confidence({"result": "test"})
        assert 0 <= confidence <= 1
        assert confidence == 0.95  # Default value
    
    def test_response_metadata(self, module_config, create_module_request):
        """Test response metadata generation."""
        module = TestModule(module_config)
        request = create_module_request(
            capabilities=[ModuleCapability.TEXT_GENERATION]
        )
        
        metadata = module._get_response_metadata(request, {"result": "test"})
        
        assert metadata["module_version"] == module_config.version
        assert metadata["capabilities_used"] == ["text_generation"]
        assert metadata["cache_hit"] is False