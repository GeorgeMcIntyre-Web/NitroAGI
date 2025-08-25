"""Integration tests for API endpoints."""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import json

from nitroagi.api.app import create_app


class TestAPIEndpoints:
    """Test API endpoint functionality."""
    
    @pytest.mark.integration
    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    @pytest.mark.integration
    def test_detailed_health(self, api_client, mock_health_monitor):
        """Test detailed health check."""
        with patch("nitroagi.api.routes.health.health_monitor", mock_health_monitor):
            response = api_client.get("/health/detailed")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "checks" in data
            assert "memory" in data["checks"]
    
    @pytest.mark.integration
    def test_chat_endpoint(self, api_client):
        """Test chat endpoint."""
        with patch("nitroagi.api.routes.chat.language_module") as mock_module:
            mock_module.process = AsyncMock(
                return_value=MagicMock(
                    data="Hello! How can I help you?",
                    status="success"
                )
            )
            
            response = api_client.post(
                "/api/v1/chat",
                json={"message": "Hello"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert data["response"] == "Hello! How can I help you?"
    
    @pytest.mark.integration
    def test_chat_with_options(self, api_client):
        """Test chat with temperature and max_tokens."""
        with patch("nitroagi.api.routes.chat.language_module") as mock_module:
            mock_module.process = AsyncMock(
                return_value=MagicMock(
                    data="Generated response",
                    status="success"
                )
            )
            
            response = api_client.post(
                "/api/v1/chat",
                json={
                    "message": "Tell me a story",
                    "temperature": 0.9,
                    "max_tokens": 500
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
    
    @pytest.mark.integration
    def test_memory_store(self, api_client):
        """Test memory storage endpoint."""
        with patch("nitroagi.api.routes.memory.memory_manager") as mock_memory:
            mock_memory.store = AsyncMock(return_value="memory-id-123")
            
            response = api_client.post(
                "/api/v1/memory/store",
                json={
                    "key": "user_preference",
                    "value": {"theme": "dark"},
                    "memory_type": "working"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["memory_id"] == "memory-id-123"
            assert data["status"] == "stored"
    
    @pytest.mark.integration
    def test_memory_retrieve(self, api_client):
        """Test memory retrieval endpoint."""
        with patch("nitroagi.api.routes.memory.memory_manager") as mock_memory:
            mock_memory.retrieve = AsyncMock(
                return_value={"theme": "dark"}
            )
            
            response = api_client.get("/api/v1/memory/retrieve/user_preference")
            
            assert response.status_code == 200
            data = response.json()
            assert data["value"] == {"theme": "dark"}
    
    @pytest.mark.integration
    def test_memory_search(self, api_client):
        """Test memory search endpoint."""
        with patch("nitroagi.api.routes.memory.memory_manager") as mock_memory:
            mock_memory.search = AsyncMock(
                return_value=[
                    {"key": "user_name", "value": "John"},
                    {"key": "user_age", "value": 30}
                ]
            )
            
            response = api_client.get("/api/v1/memory/search?pattern=user_*")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 2
    
    @pytest.mark.integration
    def test_modules_list(self, api_client):
        """Test modules listing endpoint."""
        with patch("nitroagi.api.routes.modules.module_registry") as mock_registry:
            mock_registry.list_modules = MagicMock(
                return_value=["language", "vision", "audio"]
            )
            
            response = api_client.get("/api/v1/modules")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["modules"]) == 3
            assert "language" in data["modules"]
    
    @pytest.mark.integration
    def test_module_status(self, api_client):
        """Test module status endpoint."""
        with patch("nitroagi.api.routes.modules.module_registry") as mock_registry:
            mock_module = MagicMock()
            mock_module.config.name = "language"
            mock_module.config.version = "1.0.0"
            mock_module.status.value = "ready"
            mock_module.get_metrics = MagicMock(
                return_value={"total_requests": 100}
            )
            
            mock_registry.get_module = MagicMock(return_value=mock_module)
            
            response = api_client.get("/api/v1/modules/language/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "language"
            assert data["status"] == "ready"
            assert data["metrics"]["total_requests"] == 100
    
    @pytest.mark.integration
    def test_system_info(self, api_client):
        """Test system info endpoint."""
        response = api_client.get("/api/v1/system/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "environment" in data
        assert "uptime" in data
    
    @pytest.mark.integration
    def test_system_metrics(self, api_client):
        """Test system metrics endpoint."""
        response = api_client.get("/api/v1/system/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "disk_usage" in data
    
    @pytest.mark.integration
    def test_error_handling(self, api_client):
        """Test API error handling."""
        # Test 404
        response = api_client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test invalid JSON
        response = api_client.post(
            "/api/v1/chat",
            data="invalid json"
        )
        assert response.status_code == 422
    
    @pytest.mark.integration
    def test_cors_headers(self, api_client):
        """Test CORS headers."""
        response = api_client.options(
            "/api/v1/chat",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_connection(self, test_app):
        """Test WebSocket connection."""
        from fastapi.testclient import TestClient
        
        with TestClient(test_app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Send message
                websocket.send_json({"type": "ping"})
                
                # Receive response
                data = websocket.receive_json()
                assert data["type"] == "pong"