"""Health check and monitoring endpoints."""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from nitroagi.utils.config import get_config
from nitroagi.utils.logging import get_logger


router = APIRouter()
logger = get_logger(__name__)
config = get_config()


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="0.1.0")
    environment: str = Field(...)
    uptime_seconds: float = Field(...)
    services: Dict[str, str] = Field(default_factory=dict)


class SystemMetrics(BaseModel):
    """System metrics response model."""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    disk_usage: float = Field(..., description="Disk usage percentage")
    active_connections: int = Field(default=0)
    requests_per_minute: float = Field(default=0.0)
    average_response_time_ms: float = Field(default=0.0)


# Track startup time
startup_time = datetime.utcnow()


@router.get("/health", response_model=HealthStatus)
async def health_check(request: Request) -> HealthStatus:
    """Health check endpoint.
    
    Returns:
        Health status of the system
    """
    uptime = (datetime.utcnow() - startup_time).total_seconds()
    
    # Check service health
    services = {}
    
    # Check message bus
    if hasattr(request.app.state, "message_bus"):
        services["message_bus"] = "healthy" if request.app.state.message_bus._running else "unhealthy"
    
    # Check orchestrator
    if hasattr(request.app.state, "orchestrator"):
        services["orchestrator"] = "healthy" if request.app.state.orchestrator._running else "unhealthy"
    
    # Check memory manager
    if hasattr(request.app.state, "memory_manager"):
        services["memory_manager"] = "healthy"
    
    # Check network manager
    if hasattr(request.app.state, "network_manager"):
        services["network_manager"] = "healthy"
    
    # Determine overall status
    overall_status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
    
    return HealthStatus(
        status=overall_status,
        environment=config.environment,
        uptime_seconds=uptime,
        services=services
    )


@router.get("/health/live")
async def liveness_probe() -> Dict[str, str]:
    """Kubernetes liveness probe endpoint.
    
    Returns:
        Simple liveness status
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe(request: Request) -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint.
    
    Returns:
        Readiness status with service checks
    """
    ready = True
    checks = {}
    
    # Check core services
    if hasattr(request.app.state, "message_bus"):
        mb_ready = request.app.state.message_bus._running
        checks["message_bus"] = mb_ready
        ready = ready and mb_ready
    
    if hasattr(request.app.state, "orchestrator"):
        orch_ready = request.app.state.orchestrator._running
        checks["orchestrator"] = orch_ready
        ready = ready and orch_ready
    
    return {
        "ready": ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metrics/system", response_model=SystemMetrics)
async def system_metrics() -> SystemMetrics:
    """Get system metrics.
    
    Returns:
        Current system metrics
    """
    try:
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        return SystemMetrics(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            active_connections=len(psutil.net_connections()),
        )
    except ImportError:
        # Return dummy metrics if psutil not available
        return SystemMetrics(
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
        )


@router.get("/metrics/modules")
async def module_metrics(request: Request) -> Dict[str, Any]:
    """Get metrics for all AI modules.
    
    Returns:
        Metrics for each registered module
    """
    if not hasattr(request.app.state, "registry"):
        return {"error": "Module registry not initialized"}
    
    registry = request.app.state.registry
    metrics = await registry.health_check_all()
    
    return {
        "modules": metrics,
        "total_modules": len(metrics),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metrics/memory")
async def memory_metrics(request: Request) -> Dict[str, Any]:
    """Get memory system metrics.
    
    Returns:
        Memory system statistics
    """
    if not hasattr(request.app.state, "memory_manager"):
        return {"error": "Memory manager not initialized"}
    
    memory_manager = request.app.state.memory_manager
    return memory_manager.get_metrics()


@router.get("/metrics/network")
async def network_metrics(request: Request) -> Dict[str, Any]:
    """Get network metrics including 6G readiness.
    
    Returns:
        Network statistics and 6G readiness indicators
    """
    if not hasattr(request.app.state, "network_manager"):
        return {"error": "Network manager not initialized"}
    
    network_manager = request.app.state.network_manager
    return network_manager.get_network_stats()