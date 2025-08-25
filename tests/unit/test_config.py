"""Unit tests for configuration management."""

import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

from nitroagi.utils.config import (
    Settings,
    AIModelConfig,
    DatabaseConfig,
    NetworkConfig,
    get_config,
    load_config
)


class TestSettings:
    """Test Settings configuration."""
    
    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()
        
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.redis_pool_size == 10
    
    def test_settings_with_env_vars(self):
        """Test settings with environment variables."""
        with patch.dict(os.environ, {
            "NITROAGI_API_HOST": "127.0.0.1",
            "NITROAGI_API_PORT": "8080",
            "NITROAGI_DEBUG": "true",
            "NITROAGI_LOG_LEVEL": "DEBUG"
        }):
            settings = Settings()
            
            assert settings.api_host == "127.0.0.1"
            assert settings.api_port == 8080
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
    
    def test_ai_model_config(self):
        """Test AI model configuration."""
        config = AIModelConfig(
            openai_api_key="test-key",
            default_llm_model="gpt-4",
            temperature=0.8,
            max_tokens=2000
        )
        
        assert config.openai_api_key == "test-key"
        assert config.default_llm_model == "gpt-4"
        assert config.temperature == 0.8
        assert config.max_tokens == 2000
    
    def test_database_config(self):
        """Test database configuration."""
        config = DatabaseConfig(
            redis_url="redis://localhost:6379",
            postgres_url="postgresql://user:pass@localhost/db",
            mongodb_url="mongodb://localhost:27017"
        )
        
        assert config.redis_url == "redis://localhost:6379"
        assert config.postgres_url == "postgresql://user:pass@localhost/db"
        assert config.mongodb_url == "mongodb://localhost:27017"
    
    def test_network_config(self):
        """Test network configuration."""
        config = NetworkConfig(
            enable_6g=True,
            max_latency_ms=1.0,
            min_bandwidth_mbps=10000,
            network_profile="ultra_low_latency"
        )
        
        assert config.enable_6g is True
        assert config.max_latency_ms == 1.0
        assert config.min_bandwidth_mbps == 10000
        assert config.network_profile == "ultra_low_latency"
    
    @patch('nitroagi.utils.config.load_dotenv')
    def test_get_config_singleton(self, mock_load_dotenv):
        """Test get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
        mock_load_dotenv.assert_called_once()
    
    def test_load_config_from_file(self, temp_dir):
        """Test loading configuration from file."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("""
api:
  host: "192.168.1.1"
  port: 9000
  
ai_models:
  default_llm_model: "claude-3"
  temperature: 0.5
  
database:
  redis_url: "redis://redis:6379"
""")
        
        config = load_config(str(config_file))
        
        assert config["api"]["host"] == "192.168.1.1"
        assert config["api"]["port"] == 9000
        assert config["ai_models"]["default_llm_model"] == "claude-3"
        assert config["database"]["redis_url"] == "redis://redis:6379"
    
    def test_settings_validation(self):
        """Test settings validation."""
        # Invalid temperature (should be 0-1)
        with pytest.raises(ValueError):
            AIModelConfig(temperature=1.5)
        
        # Invalid max_tokens (should be positive)
        with pytest.raises(ValueError):
            AIModelConfig(max_tokens=-100)
    
    def test_environment_specific_config(self):
        """Test environment-specific configuration."""
        with patch.dict(os.environ, {"NITROAGI_ENV": "production"}):
            settings = Settings()
            assert settings.debug is False
            assert settings.log_level == "INFO"
        
        with patch.dict(os.environ, {"NITROAGI_ENV": "development"}):
            settings = Settings(debug=True, log_level="DEBUG")
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
    
    def test_config_export(self):
        """Test configuration export."""
        settings = Settings()
        
        # Export as dict
        config_dict = settings.dict()
        assert isinstance(config_dict, dict)
        assert "api_host" in config_dict
        assert "ai_models" in config_dict
        
        # Export as JSON
        config_json = settings.json()
        assert isinstance(config_json, str)
        assert "api_host" in config_json