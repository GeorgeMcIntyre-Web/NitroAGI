"""Module management endpoints."""

from typing import Any, Dict, List
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from nitroagi.core.base import ModuleConfig, ModuleCapability
from nitroagi.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ModuleRegistrationRequest(BaseModel):
    """Module registration request model."""
    module_type: str = Field(..., description="Type of module to register")
    config: Dict[str, Any] = Field(..., description="Module configuration")


@router.get("/")
async def list_modules(request: Request) -> Dict[str, Any]:
    """List all registered modules.
    
    Returns:
        List of registered modules
    """
    if not hasattr(request.app.state, "registry"):
        raise HTTPException(status_code=503, detail="Module registry not initialized")
    
    registry = request.app.state.registry
    modules = []
    
    for name, module in registry._modules.items():
        modules.append({
            "name": name,
            "type": module.__class__.__name__,
            "status": module.status.value,
            "capabilities": [cap.value for cap in module.config.capabilities],
            "version": module.config.version
        })
    
    return {
        "modules": modules,
        "total": len(modules)
    }


@router.get("/{module_name}")
async def get_module_info(
    module_name: str,
    request: Request
) -> Dict[str, Any]:
    """Get information about a specific module.
    
    Args:
        module_name: Name of the module
        request: FastAPI request
        
    Returns:
        Module information
    """
    if not hasattr(request.app.state, "registry"):
        raise HTTPException(status_code=503, detail="Module registry not initialized")
    
    registry = request.app.state.registry
    module = registry.get_module(module_name)
    
    if not module:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
    
    health = await module.health_check()
    
    return {
        "name": module_name,
        "type": module.__class__.__name__,
        "status": module.status.value,
        "config": module.config.dict(),
        "health": health
    }


@router.post("/register")
async def register_module(
    registration: ModuleRegistrationRequest,
    request: Request
) -> Dict[str, str]:
    """Register a new module.
    
    Args:
        registration: Module registration request
        request: FastAPI request
        
    Returns:
        Registration confirmation
    """
    if not hasattr(request.app.state, "registry"):
        raise HTTPException(status_code=503, detail="Module registry not initialized")
    
    # This is a placeholder - actual implementation would create and register module
    return {
        "status": "success",
        "message": f"Module registration for {registration.module_type} received",
        "note": "Dynamic module registration not fully implemented"
    }


@router.post("/{module_name}/health")
async def check_module_health(
    module_name: str,
    request: Request
) -> Dict[str, Any]:
    """Check health of a specific module.
    
    Args:
        module_name: Name of the module
        request: FastAPI request
        
    Returns:
        Module health status
    """
    if not hasattr(request.app.state, "registry"):
        raise HTTPException(status_code=503, detail="Module registry not initialized")
    
    registry = request.app.state.registry
    module = registry.get_module(module_name)
    
    if not module:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
    
    health = await module.health_check()
    return health


@router.get("/capabilities/search")
async def search_by_capability(
    capability: str,
    request: Request
) -> Dict[str, Any]:
    """Find modules by capability.
    
    Args:
        capability: Capability to search for
        request: FastAPI request
        
    Returns:
        Modules with the specified capability
    """
    if not hasattr(request.app.state, "registry"):
        raise HTTPException(status_code=503, detail="Module registry not initialized")
    
    try:
        cap = ModuleCapability(capability)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid capability: {capability}")
    
    registry = request.app.state.registry
    modules = registry.get_modules_by_capability(cap)
    
    return {
        "capability": capability,
        "modules": [
            {
                "name": m.config.name,
                "type": m.__class__.__name__,
                "status": m.status.value
            }
            for m in modules
        ],
        "count": len(modules)
    }