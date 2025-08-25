"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch
import os

import redis.asyncio as redis
from fakeredis import aioredis as fakeredis

from nitroagi.core.base import ModuleConfig, ModuleCapability, ModuleContext
from nitroagi.core.memory import MemoryManager, MemoryType
from nitroagi.utils.config import Settings, AIModelConfig, DatabaseConfig
from nitroagi.utils.logging import get_logger


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    return fakeredis.FakeRedis()


@pytest.fixture
async def memory_manager(mock_redis):
    """Create MemoryManager with mock Redis."""
    manager = MemoryManager()
    manager.redis = mock_redis
    yield manager
    if manager.redis:
        await manager.redis.close()


@pytest.fixture
def test_config():
    """Create test configuration."""
    return Settings(
        ai_models=AIModelConfig(
            openai_api_key="test-key",
            default_llm_model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        ),
        database=DatabaseConfig(
            redis_url="redis://localhost:6379",
            postgres_url="postgresql://test:test@localhost/test"
        ),
        api_host="localhost",
        api_port=8000,
        debug=True,
        log_level="DEBUG"
    )


@pytest.fixture
def module_config():
    """Create test module configuration."""
    return ModuleConfig(
        name="test_module",
        version="1.0.0",
        description="Test module",
        capabilities=[ModuleCapability.TEXT_GENERATION],
        max_workers=2,
        timeout_seconds=10.0,
        cache_enabled=True,
        cache_ttl_seconds=300
    )


@pytest.fixture
def module_context():
    """Create test module context."""
    return ModuleContext(
        request_id="test-request-123",
        user_id="test-user",
        session_id="test-session",
        conversation_id="test-conversation",
        timestamp=1234567890.0,
        metadata={"test": "data"}
    )


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider."""
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value="Generated text response")
    provider.count_tokens = AsyncMock(return_value=100)
    return provider


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    with patch("openai.AsyncOpenAI") as mock:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(content="OpenAI response")
                    )
                ]
            )
        )
        mock.return_value = client
        yield client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    with patch("anthropic.AsyncAnthropic") as mock:
        client = MagicMock()
        client.messages.create = AsyncMock(
            return_value=MagicMock(
                content=[MagicMock(text="Anthropic response")]
            )
        )
        mock.return_value = client
        yield client


@pytest.fixture
async def test_app():
    """Create test FastAPI application."""
    from nitroagi.api.app import create_app
    
    # Use test configuration
    with patch("nitroagi.utils.config.get_config") as mock_config:
        mock_config.return_value = test_config()
        app = create_app()
        yield app


@pytest.fixture
def api_client(test_app):
    """Create test client for API."""
    from fastapi.testclient import TestClient
    return TestClient(test_app)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Artificial General Intelligence (AGI) represents a significant milestone
    in the evolution of artificial intelligence. Unlike narrow AI systems
    that excel at specific tasks, AGI aims to match or exceed human cognitive
    abilities across all domains of knowledge and reasoning.
    """


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is AGI?"},
        {"role": "assistant", "content": "AGI stands for Artificial General Intelligence."},
        {"role": "user", "content": "Tell me more about it."}
    ]


@pytest.fixture
def network_profile():
    """Sample network profile for testing."""
    from nitroagi.core.network import NetworkProfile
    return NetworkProfile(
        name="test_profile",
        max_latency_ms=10.0,
        min_bandwidth_mbps=1000,
        packet_loss_threshold=0.01,
        jitter_ms=1.0,
        priority=5
    )


@pytest.fixture
def mock_health_monitor():
    """Mock health monitor."""
    monitor = MagicMock()
    monitor.check_health = AsyncMock(return_value={
        "status": "healthy",
        "checks": {
            "memory": {"status": "healthy"},
            "redis": {"status": "healthy"},
            "modules": {"status": "healthy"}
        }
    })
    return monitor


# Environment setup fixtures
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("NITROAGI_ENV", "test")
    monkeypatch.setenv("NITROAGI_DEBUG", "true")
    monkeypatch.setenv("NITROAGI_LOG_LEVEL", "DEBUG")