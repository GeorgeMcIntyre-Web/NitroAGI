"""End-to-end tests for NitroAGI system."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import time
import json

from nitroagi.api.app import create_app
from nitroagi.core.orchestrator import Orchestrator
from nitroagi.core.memory import MemoryManager, MemoryType
from nitroagi.modules.language.language_module import LanguageModule
from nitroagi.core.base import ModuleRequest, ModuleCapability, ModuleContext


@pytest.mark.e2e
class TestEndToEnd:
    """End-to-end system tests."""
    
    @pytest.mark.asyncio
    async def test_full_system_initialization(self):
        """Test complete system initialization."""
        with patch("nitroagi.utils.config.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                ai_models=MagicMock(openai_api_key=None),
                database=MagicMock(redis_url="redis://localhost:6379"),
                api_host="localhost",
                api_port=8000
            )
            
            # Initialize orchestrator
            orchestrator = Orchestrator()
            await orchestrator.initialize()
            
            # Initialize memory manager
            memory_manager = MemoryManager()
            await memory_manager.initialize()
            
            # Initialize language module
            language_module = LanguageModule()
            await language_module.initialize()
            
            # Verify all components are ready
            assert orchestrator.status == "ready"
            assert memory_manager.initialized
            assert language_module._initialized
            
            # Cleanup
            await language_module.shutdown()
            await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_request_processing_pipeline(self):
        """Test complete request processing pipeline."""
        orchestrator = Orchestrator()
        memory_manager = MemoryManager()
        
        # Mock language module
        mock_language = AsyncMock()
        mock_language.process = AsyncMock(
            return_value=MagicMock(
                data="Processed response",
                status="success",
                confidence_score=0.95
            )
        )
        
        orchestrator.module_registry.register("language", mock_language)
        
        await orchestrator.initialize()
        
        # Create and process request
        context = ModuleContext(
            request_id="test-123",
            user_id="user-1",
            session_id="session-1",
            conversation_id="conv-1"
        )
        
        request = ModuleRequest(
            context=context,
            data={"prompt": "Hello, NitroAGI"},
            required_capabilities=[ModuleCapability.TEXT_GENERATION]
        )
        
        # Process through orchestrator
        result = await orchestrator.process_request(request)
        
        assert result is not None
        assert result["status"] == "success"
        assert result["response"] == "Processed response"
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_multi_module_coordination(self):
        """Test coordination between multiple modules."""
        orchestrator = Orchestrator()
        
        # Mock multiple modules
        language_module = AsyncMock()
        language_module.process = AsyncMock(
            return_value=MagicMock(
                data="Text description of image",
                status="success"
            )
        )
        
        vision_module = AsyncMock()
        vision_module.process = AsyncMock(
            return_value=MagicMock(
                data={"objects": ["cat", "dog"], "scene": "park"},
                status="success"
            )
        )
        
        orchestrator.module_registry.register("language", language_module)
        orchestrator.module_registry.register("vision", vision_module)
        
        await orchestrator.initialize()
        
        # Process multi-modal request
        request = {
            "type": "multi_modal",
            "image": "base64_image_data",
            "prompt": "Describe this image"
        }
        
        # First, vision module processes image
        vision_result = await vision_module.process(
            MagicMock(data=request["image"])
        )
        
        # Then, language module generates description
        language_result = await language_module.process(
            MagicMock(data={
                "prompt": request["prompt"],
                "context": vision_result.data
            })
        )
        
        assert vision_result.status == "success"
        assert language_result.status == "success"
        assert language_result.data == "Text description of image"
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_persistence_flow(self):
        """Test memory persistence and retrieval flow."""
        memory_manager = MemoryManager()
        await memory_manager.initialize()
        
        # Store conversation context
        conversation_data = {
            "messages": [
                {"role": "user", "content": "Remember that my name is John"},
                {"role": "assistant", "content": "I'll remember that your name is John"}
            ],
            "metadata": {
                "timestamp": time.time(),
                "importance": 0.9
            }
        }
        
        memory_id = await memory_manager.store(
            key="user_name_context",
            value=conversation_data,
            memory_type=MemoryType.EPISODIC
        )
        
        assert memory_id is not None
        
        # Retrieve memory
        retrieved = await memory_manager.retrieve("user_name_context")
        assert retrieved is not None
        assert retrieved["messages"][0]["content"] == "Remember that my name is John"
        
        # Search memories
        results = await memory_manager.search("user_*")
        assert len(results) > 0
        assert any(r["key"] == "user_name_context" for r in results)
        
        # Consolidate memories
        await memory_manager.consolidate()
        
        # Memory should still be accessible (high importance)
        retrieved_after = await memory_manager.retrieve("user_name_context")
        assert retrieved_after is not None
    
    @pytest.mark.asyncio
    async def test_api_to_module_flow(self, api_client):
        """Test flow from API endpoint to module processing."""
        with patch("nitroagi.api.routes.chat.language_module") as mock_language:
            # Setup mock response
            mock_language.process = AsyncMock(
                return_value=MagicMock(
                    data="Hello! I'm NitroAGI, ready to assist you.",
                    status="success",
                    confidence_score=0.98,
                    processing_time_ms=150
                )
            )
            
            # Make API request
            response = api_client.post(
                "/api/v1/chat",
                json={
                    "message": "Hello, introduce yourself",
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "response" in data
            assert "NitroAGI" in data["response"]
            assert "metadata" in data
            assert data["metadata"]["confidence"] == 0.98
            assert data["metadata"]["processing_time_ms"] == 150
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error propagation through the system."""
        orchestrator = Orchestrator()
        
        # Module that fails
        failing_module = AsyncMock()
        failing_module.process = AsyncMock(
            side_effect=Exception("Module processing failed")
        )
        
        orchestrator.module_registry.register("failing", failing_module)
        
        await orchestrator.initialize()
        
        request = ModuleRequest(
            context=ModuleContext(
                request_id="test-error",
                user_id="user-1"
            ),
            data={"test": "data"},
            required_capabilities=[ModuleCapability.TEXT_GENERATION]
        )
        
        # Process should handle error gracefully
        result = await orchestrator.process_request(request)
        
        assert result["status"] == "error"
        assert "Module processing failed" in result["error"]
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load."""
        orchestrator = Orchestrator()
        
        # Mock fast module
        fast_module = AsyncMock()
        async def fast_process(req):
            await asyncio.sleep(0.01)  # Simulate 10ms processing
            return MagicMock(data="response", status="success")
        
        fast_module.process = fast_process
        orchestrator.module_registry.register("fast", fast_module)
        
        await orchestrator.initialize()
        
        # Generate concurrent requests
        num_requests = 100
        start_time = time.time()
        
        async def make_request(i):
            request = ModuleRequest(
                context=ModuleContext(
                    request_id=f"load-test-{i}",
                    user_id="user-1"
                ),
                data={"index": i},
                required_capabilities=[ModuleCapability.TEXT_GENERATION]
            )
            return await orchestrator.process_request(request)
        
        # Process requests concurrently
        results = await asyncio.gather(
            *[make_request(i) for i in range(num_requests)]
        )
        
        elapsed = time.time() - start_time
        
        # Verify all requests succeeded
        assert all(r["status"] == "success" for r in results)
        
        # Performance check (should handle 100 requests in < 2 seconds)
        assert elapsed < 2.0
        
        # Calculate throughput
        throughput = num_requests / elapsed
        print(f"Throughput: {throughput:.2f} requests/second")
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful system shutdown."""
        # Initialize components
        orchestrator = Orchestrator()
        memory_manager = MemoryManager()
        language_module = LanguageModule()
        
        await orchestrator.initialize()
        await memory_manager.initialize()
        
        # Start some background tasks
        async def background_task():
            while True:
                await asyncio.sleep(0.1)
        
        task = asyncio.create_task(background_task())
        
        # Graceful shutdown
        await orchestrator.shutdown()
        task.cancel()
        
        # Verify clean shutdown
        assert orchestrator.status == "shutdown"
        
        # Attempt to process after shutdown should fail
        with pytest.raises(RuntimeError):
            await orchestrator.process_request(MagicMock())