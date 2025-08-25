"""Shared test fixtures."""

import pytest
from typing import Dict, Any, List
from unittest.mock import MagicMock, AsyncMock
import uuid

from nitroagi.core.base import (
    ModuleRequest,
    ModuleResponse,
    ModuleCapability,
    ModuleContext
)


@pytest.fixture
def create_module_request():
    """Factory for creating module requests."""
    def _create_request(
        data: Any = None,
        capabilities: List[ModuleCapability] = None,
        priority: int = 5,
        timeout: float = None
    ) -> ModuleRequest:
        return ModuleRequest(
            context=ModuleContext(
                request_id=str(uuid.uuid4()),
                user_id="test-user",
                session_id="test-session",
                conversation_id="test-conversation"
            ),
            data=data or {"test": "data"},
            required_capabilities=capabilities or [ModuleCapability.TEXT_GENERATION],
            priority=priority,
            timeout_override=timeout
        )
    return _create_request


@pytest.fixture
def create_module_response():
    """Factory for creating module responses."""
    def _create_response(
        request_id: str = None,
        module_name: str = "test_module",
        status: str = "success",
        data: Any = None,
        error: str = None,
        confidence: float = 0.95
    ) -> ModuleResponse:
        return ModuleResponse(
            request_id=request_id or str(uuid.uuid4()),
            module_name=module_name,
            status=status,
            data=data or {"result": "test"},
            processing_time_ms=100.0,
            confidence_score=confidence,
            error=error,
            metadata={"test": "metadata"}
        )
    return _create_response


@pytest.fixture
def mock_module():
    """Create mock AI module."""
    module = MagicMock()
    module.config.name = "test_module"
    module.config.version = "1.0.0"
    module.config.capabilities = [ModuleCapability.TEXT_GENERATION]
    module.initialize = AsyncMock()
    module.process = AsyncMock()
    module.shutdown = AsyncMock()
    module.health_check = AsyncMock(return_value={
        "status": "healthy",
        "module": "test_module"
    })
    return module


@pytest.fixture
def memory_items():
    """Sample memory items for testing."""
    return [
        {
            "key": "test_key_1",
            "value": {"data": "test data 1"},
            "type": "working",
            "metadata": {"importance": 0.8}
        },
        {
            "key": "test_key_2",
            "value": {"data": "test data 2"},
            "type": "episodic",
            "metadata": {"importance": 0.6}
        },
        {
            "key": "test_key_3",
            "value": {"data": "test data 3"},
            "type": "semantic",
            "metadata": {"importance": 0.9}
        }
    ]


@pytest.fixture
def api_request_data():
    """Sample API request data."""
    return {
        "chat": {
            "message": "Hello, how are you?",
            "temperature": 0.7,
            "max_tokens": 500
        },
        "vision": {
            "image_data": "base64_encoded_image",
            "task": "describe"
        },
        "memory": {
            "operation": "store",
            "key": "test_memory",
            "value": {"content": "Important information"}
        }
    }