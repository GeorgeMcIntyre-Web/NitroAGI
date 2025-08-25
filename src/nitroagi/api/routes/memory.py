"""Memory management endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from nitroagi.core.memory import MemoryType, MemoryPriority
from nitroagi.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class MemoryStoreRequest(BaseModel):
    """Memory store request model."""
    key: str = Field(..., description="Memory key")
    value: Any = Field(..., description="Value to store")
    memory_type: str = Field(default="working")
    priority: str = Field(default="normal")
    ttl_seconds: Optional[int] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)


@router.post("/store")
async def store_memory(
    request: MemoryStoreRequest,
    req: Request
) -> Dict[str, Any]:
    """Store a memory entry.
    
    Args:
        request: Memory store request
        req: FastAPI request
        
    Returns:
        Storage confirmation
    """
    if not hasattr(req.app.state, "memory_manager"):
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    
    memory_manager = req.app.state.memory_manager
    
    memory_id = await memory_manager.store(
        key=request.key,
        value=request.value,
        memory_type=MemoryType(request.memory_type),
        priority=MemoryPriority(request.priority),
        ttl_seconds=request.ttl_seconds,
        metadata=request.metadata
    )
    
    return {
        "status": "success",
        "memory_id": str(memory_id),
        "key": request.key
    }


@router.get("/retrieve/{key}")
async def retrieve_memory(
    key: str,
    req: Request,
    memory_type: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve a memory entry.
    
    Args:
        key: Memory key
        req: FastAPI request
        memory_type: Optional memory type filter
        
    Returns:
        Memory value if found
    """
    if not hasattr(req.app.state, "memory_manager"):
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    
    memory_manager = req.app.state.memory_manager
    
    mem_type = MemoryType(memory_type) if memory_type else None
    value = await memory_manager.retrieve(key, mem_type)
    
    if value is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {
        "status": "success",
        "key": key,
        "value": value
    }


@router.delete("/delete/{key}")
async def delete_memory(
    key: str,
    req: Request
) -> Dict[str, str]:
    """Delete a memory entry.
    
    Args:
        key: Memory key to delete
        req: FastAPI request
        
    Returns:
        Deletion confirmation
    """
    if not hasattr(req.app.state, "memory_manager"):
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    
    memory_manager = req.app.state.memory_manager
    
    deleted = await memory_manager.delete(key)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {
        "status": "success",
        "message": f"Memory {key} deleted"
    }


@router.post("/search")
async def search_memory(
    query: str,
    req: Request,
    memory_type: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """Search memory entries.
    
    Args:
        query: Search query
        req: FastAPI request
        memory_type: Optional memory type filter
        limit: Maximum results
        
    Returns:
        Search results
    """
    if not hasattr(req.app.state, "memory_manager"):
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    
    memory_manager = req.app.state.memory_manager
    
    mem_type = MemoryType(memory_type) if memory_type else None
    results = await memory_manager.search(query, mem_type, limit)
    
    return {
        "status": "success",
        "query": query,
        "results": [r.to_dict() for r in results],
        "count": len(results)
    }


@router.post("/consolidate")
async def consolidate_memory(req: Request) -> Dict[str, str]:
    """Trigger memory consolidation.
    
    Args:
        req: FastAPI request
        
    Returns:
        Consolidation status
    """
    if not hasattr(req.app.state, "memory_manager"):
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    
    memory_manager = req.app.state.memory_manager
    await memory_manager.consolidate()
    
    return {
        "status": "success",
        "message": "Memory consolidation completed"
    }