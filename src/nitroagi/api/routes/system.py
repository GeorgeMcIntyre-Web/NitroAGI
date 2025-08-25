"""System management and configuration endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from nitroagi.utils.config import get_config
from nitroagi.utils.logging import get_logger
from nitroagi.core.network import NetworkProfile, NETWORK_PROFILES


router = APIRouter()
logger = get_logger(__name__)
config = get_config()


class SystemInfo(BaseModel):
    """System information model."""
    version: str = Field(default="0.1.0")
    environment: str = Field(...)
    capabilities: Dict[str, bool] = Field(...)
    modules: List[Dict[str, Any]] = Field(default_factory=list)
    limits: Dict[str, Any] = Field(...)
    network_profiles: Dict[str, Dict[str, Any]] = Field(...)


class ConfigUpdate(BaseModel):
    """Configuration update model."""
    section: str = Field(..., description="Configuration section to update")
    key: str = Field(..., description="Configuration key")
    value: Any = Field(..., description="New value")


@router.get("/info", response_model=SystemInfo)
async def get_system_info(request: Request) -> SystemInfo:
    """Get system information and capabilities.
    
    Returns:
        System information including capabilities and limits
    """
    # Get module information
    modules = []
    if hasattr(request.app.state, "registry"):
        registry = request.app.state.registry
        for name, module in registry._modules.items():
            modules.append({
                "name": name,
                "status": module.status.value,
                "capabilities": [cap.value for cap in module.config.capabilities]
            })
    
    # Format network profiles for response
    network_profiles = {
        name: {
            "min_bandwidth_mbps": profile.min_bandwidth_mbps,
            "max_latency_ms": profile.max_latency_ms,
            "required_capabilities": profile.required_capabilities,
            "preferred_generation": profile.preferred_generation,
        }
        for name, profile in NETWORK_PROFILES.items()
    }
    
    return SystemInfo(
        environment=config.environment,
        capabilities={
            "text_processing": True,
            "image_processing": config.features.vision_module,
            "audio_processing": False,
            "multi_modal": config.features.multi_modal,
            "memory_persistence": True,
            "real_time_learning": config.features.real_time_learning,
            "6g_ready": True,
        },
        modules=modules,
        limits={
            "max_tokens": config.ai_models.max_tokens,
            "max_conversation_length": config.memory.max_conversation_length,
            "max_memory_entries": config.memory.max_memory_entries,
            "rate_limit_per_minute": config.api.rate_limit_per_minute,
            "rate_limit_per_hour": config.api.rate_limit_per_hour,
        },
        network_profiles=network_profiles
    )


@router.get("/config")
async def get_configuration() -> Dict[str, Any]:
    """Get current configuration (non-sensitive values only).
    
    Returns:
        Safe configuration values
    """
    safe_config = {
        "environment": config.environment,
        "debug": config.debug,
        "log_level": config.log_level,
        "features": {
            "vision_module": config.features.vision_module,
            "reasoning_engine": config.features.reasoning_engine,
            "learning_agent": config.features.learning_agent,
            "multi_modal": config.features.multi_modal,
            "real_time_learning": config.features.real_time_learning,
        },
        "api": {
            "host": config.api.host,
            "port": config.api.port,
            "rate_limit_enabled": config.api.rate_limit_enabled,
            "rate_limit_per_minute": config.api.rate_limit_per_minute,
        },
        "memory": {
            "max_conversation_length": config.memory.max_conversation_length,
            "max_memory_entries": config.memory.max_memory_entries,
        },
        "performance": {
            "max_workers": config.performance.max_workers,
            "cache_enabled": config.performance.cache_enabled,
        }
    }
    
    return safe_config


@router.post("/config/update")
async def update_configuration(update: ConfigUpdate) -> Dict[str, str]:
    """Update configuration dynamically (admin only).
    
    Args:
        update: Configuration update request
        
    Returns:
        Update status
    """
    # In production, this would require admin authentication
    logger.warning(f"Configuration update requested: {update.section}.{update.key}")
    
    # This is a placeholder - actual implementation would update config
    return {
        "status": "success",
        "message": f"Configuration {update.section}.{update.key} updated",
        "note": "Dynamic config updates not fully implemented"
    }


@router.post("/shutdown")
async def shutdown_system(request: Request) -> Dict[str, str]:
    """Gracefully shutdown the system (admin only).
    
    Returns:
        Shutdown status
    """
    # In production, this would require admin authentication
    logger.warning("System shutdown requested")
    
    # Shutdown components
    if hasattr(request.app.state, "orchestrator"):
        await request.app.state.orchestrator.stop()
    
    if hasattr(request.app.state, "message_bus"):
        await request.app.state.message_bus.stop()
    
    return {
        "status": "shutting_down",
        "message": "System shutdown initiated"
    }


@router.get("/logs")
async def get_recent_logs(
    level: str = "INFO",
    limit: int = 100
) -> Dict[str, Any]:
    """Get recent system logs (admin only).
    
    Args:
        level: Minimum log level
        limit: Maximum number of logs to return
        
    Returns:
        Recent log entries
    """
    # This would read from actual log files/storage
    return {
        "logs": [],
        "total": 0,
        "level": level,
        "limit": limit,
        "note": "Log retrieval not fully implemented"
    }


@router.get("/diagnostics")
async def run_diagnostics(request: Request) -> Dict[str, Any]:
    """Run system diagnostics.
    
    Returns:
        Diagnostic results
    """
    diagnostics = {
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Check message bus
    if hasattr(request.app.state, "message_bus"):
        mb = request.app.state.message_bus
        diagnostics["checks"]["message_bus"] = {
            "running": mb._running,
            "metrics": mb.get_metrics()
        }
    
    # Check orchestrator
    if hasattr(request.app.state, "orchestrator"):
        orch = request.app.state.orchestrator
        diagnostics["checks"]["orchestrator"] = {
            "running": orch._running,
            "metrics": orch.get_metrics()
        }
    
    # Check memory
    if hasattr(request.app.state, "memory_manager"):
        mem = request.app.state.memory_manager
        diagnostics["checks"]["memory"] = mem.get_metrics()
    
    # Check network
    if hasattr(request.app.state, "network_manager"):
        net = request.app.state.network_manager
        diagnostics["checks"]["network"] = net.get_network_stats()
    
    return diagnostics