"""Configuration management for NitroAGI."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from nitroagi.core.exceptions import ConfigurationException


# Load environment variables from .env file
env_file = Path(".env")
if env_file.exists():
    load_dotenv(env_file)


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    model_config = SettingsConfigDict(env_prefix="")
    
    # PostgreSQL
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="nitroagi_dev", description="PostgreSQL database")
    postgres_user: str = Field(default="nitroagi", description="PostgreSQL user")
    postgres_password: str = Field(default="password", description="PostgreSQL password")
    database_url: Optional[str] = Field(default=None, description="Full database URL")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    
    # MongoDB
    mongodb_url: str = Field(default="mongodb://localhost:27017/nitroagi", description="MongoDB URL")
    mongodb_database: str = Field(default="nitroagi", description="MongoDB database name")
    
    # ChromaDB
    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8000, description="ChromaDB port")
    chroma_persist_directory: str = Field(default="./data/chroma", description="ChromaDB persistence directory")
    
    @field_validator("database_url")
    def construct_database_url(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if v:
            return v
        # Construct from individual components
        host = values.data.get("postgres_host", "localhost")
        port = values.data.get("postgres_port", 5432)
        db = values.data.get("postgres_db", "nitroagi_dev")
        user = values.data.get("postgres_user", "nitroagi")
        password = values.data.get("postgres_password", "password")
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"


class AIModelConfig(BaseSettings):
    """AI model configuration settings."""
    model_config = SettingsConfigDict(env_prefix="")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    huggingface_api_key: Optional[str] = Field(default=None, description="HuggingFace API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    
    # Default models
    default_llm_model: str = Field(default="gpt-4", description="Default LLM model")
    default_vision_model: str = Field(default="clip-vit-base-patch32", description="Default vision model")
    default_embedding_model: str = Field(default="text-embedding-ada-002", description="Default embedding model")
    
    # Model parameters
    max_tokens: int = Field(default=2048, ge=1, description="Maximum tokens for generation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p for nucleus sampling")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    
    # Model cache
    model_cache_dir: str = Field(default="./data/models", description="Model cache directory")
    model_cache_size_gb: int = Field(default=10, ge=1, description="Model cache size in GB")


class APIConfig(BaseSettings):
    """API server configuration settings."""
    model_config = SettingsConfigDict(env_prefix="API_")
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, ge=1, description="Number of API workers")
    reload: bool = Field(default=True, description="Enable auto-reload")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(default=60, ge=1, description="Requests per minute")
    rate_limit_per_hour: int = Field(default=1000, ge=1, description="Requests per hour")


class SecurityConfig(BaseSettings):
    """Security configuration settings."""
    model_config = SettingsConfigDict(env_prefix="")
    
    secret_key: str = Field(default="change_me_in_production", description="Secret key for encryption")
    jwt_secret_key: str = Field(default="change_me_in_production", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_delta_seconds: int = Field(default=3600, ge=60, description="JWT expiration in seconds")
    
    # Session
    session_timeout: int = Field(default=3600, ge=60, description="Session timeout in seconds")
    session_cookie_secure: bool = Field(default=False, description="Secure cookie flag")
    session_cookie_httponly: bool = Field(default=True, description="HttpOnly cookie flag")
    
    @field_validator("secret_key", "jwt_secret_key")
    def validate_secrets(cls, v: str) -> str:
        if v == "change_me_in_production":
            import warnings
            warnings.warn(
                "Using default secret key. Please set a secure secret key in production!",
                UserWarning
            )
        return v


class MemoryConfig(BaseSettings):
    """Memory system configuration settings."""
    model_config = SettingsConfigDict(env_prefix="")
    
    # Working memory
    working_memory_ttl_seconds: int = Field(default=3600, ge=1, description="Working memory TTL")
    working_memory_max_size_mb: int = Field(default=100, ge=1, description="Working memory max size in MB")
    
    # Long-term memory
    long_term_memory_enabled: bool = Field(default=True, description="Enable long-term memory")
    episodic_memory_enabled: bool = Field(default=True, description="Enable episodic memory")
    semantic_memory_enabled: bool = Field(default=True, description="Enable semantic memory")
    
    # Memory limits
    max_conversation_length: int = Field(default=100, ge=1, description="Max conversation length")
    max_memory_entries: int = Field(default=10000, ge=100, description="Max memory entries")


class PerformanceConfig(BaseSettings):
    """Performance configuration settings."""
    model_config = SettingsConfigDict(env_prefix="")
    
    # Threading and concurrency
    max_workers: int = Field(default=10, ge=1, description="Maximum worker threads")
    thread_pool_size: int = Field(default=20, ge=1, description="Thread pool size")
    async_timeout_seconds: int = Field(default=30, ge=1, description="Async operation timeout")
    
    # Caching
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl_seconds: int = Field(default=300, ge=1, description="Cache TTL in seconds")
    cache_max_size_mb: int = Field(default=500, ge=1, description="Cache max size in MB")
    
    # Request timeouts
    request_timeout_seconds: int = Field(default=60, ge=1, description="Request timeout")
    inference_timeout_seconds: int = Field(default=120, ge=1, description="Inference timeout")


class MonitoringConfig(BaseSettings):
    """Monitoring configuration settings."""
    model_config = SettingsConfigDict(env_prefix="")
    
    # Metrics
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics port")
    prometheus_enabled: bool = Field(default=False, description="Enable Prometheus metrics")
    
    # Error tracking
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN")
    sentry_environment: str = Field(default="development", description="Sentry environment")
    sentry_traces_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Sentry trace sample rate")


class FeatureFlags(BaseSettings):
    """Feature flags configuration."""
    model_config = SettingsConfigDict(env_prefix="FEATURE_")
    
    vision_module: bool = Field(default=True, description="Enable vision module")
    reasoning_engine: bool = Field(default=True, description="Enable reasoning engine")
    learning_agent: bool = Field(default=True, description="Enable learning agent")
    advanced_memory: bool = Field(default=True, description="Enable advanced memory")
    multi_modal: bool = Field(default=True, description="Enable multi-modal processing")
    real_time_learning: bool = Field(default=False, description="Enable real-time learning")


class Config(BaseSettings):
    """Main configuration class for NitroAGI."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default="logs/nitroagi.log", description="Log file path")
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ai_models: AIModelConfig = Field(default_factory=AIModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    
    @field_validator("environment")
    def validate_environment(cls, v: str) -> str:
        valid_envs = ["development", "staging", "production", "testing"]
        if v not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v
    
    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == "testing"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def validate_config(self) -> None:
        """Validate the configuration."""
        errors = []
        
        # Check required API keys in production
        if self.is_production():
            if self.security.secret_key == "change_me_in_production":
                errors.append("Secret key must be set in production")
            if self.security.jwt_secret_key == "change_me_in_production":
                errors.append("JWT secret key must be set in production")
        
        # Check database connectivity
        if not self.database.database_url:
            errors.append("Database URL is not configured")
        
        # Check required API keys for AI models
        if not any([
            self.ai_models.openai_api_key,
            self.ai_models.anthropic_api_key,
            self.ai_models.huggingface_api_key,
            self.ai_models.google_api_key
        ]):
            import warnings
            warnings.warn(
                "No AI model API keys configured. Some features may not work.",
                UserWarning
            )
        
        if errors:
            raise ConfigurationException(
                "Configuration validation failed",
                details={"errors": errors}
            )
    
    def get_database_url(self, test: bool = False) -> str:
        """Get the database URL.
        
        Args:
            test: If True, return test database URL
            
        Returns:
            Database URL string
        """
        if test and self.is_testing():
            # Use test database
            base_url = self.database.database_url
            if base_url:
                # Replace database name with test database
                parts = base_url.rsplit("/", 1)
                if len(parts) == 2:
                    return f"{parts[0]}/nitroagi_test"
        return self.database.database_url or ""


@lru_cache()
def get_config() -> Config:
    """Get the configuration singleton.
    
    Returns:
        Configuration instance
    """
    config = Config()
    
    # Validate configuration
    try:
        config.validate_config()
    except ConfigurationException as e:
        import warnings
        warnings.warn(f"Configuration validation warnings: {e.details}", UserWarning)
    
    return config


def get_settings() -> Config:
    """Alias for get_config().
    
    Returns:
        Configuration instance
    """
    return get_config()


# Create a global config instance
config = get_config()