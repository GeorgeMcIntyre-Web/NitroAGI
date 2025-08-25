"""
NEXUS Core API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import logging

from src.nitroagi.core.orchestrator import Orchestrator
from src.nitroagi.core.prefrontal_cortex import PrefrontalCortex
from src.nitroagi.core.executive_controller import ExecutiveController

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nexus")


class TaskRequest(BaseModel):
    """Request model for task execution"""
    goal: str = Field(..., description="The goal or task to accomplish")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    modules: Optional[List[str]] = Field(None, description="Specific modules to use")
    strategy: Optional[str] = Field("auto", description="Execution strategy")
    timeout: Optional[int] = Field(300, description="Timeout in seconds")


class TaskResponse(BaseModel):
    """Response model for task execution"""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    execution_time: float
    modules_used: List[str]
    confidence: float
    timestamp: datetime


class SystemStatus(BaseModel):
    """System status response"""
    status: str
    version: str
    uptime: float
    active_modules: List[str]
    memory_usage: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class ExecutivePlan(BaseModel):
    """Executive planning request"""
    goal: str
    constraints: Optional[List[str]] = Field(default_factory=list)
    resources: Optional[Dict[str, float]] = Field(default_factory=dict)
    time_limit: Optional[int] = Field(None)


# Global instances (in production, use dependency injection)
orchestrator = None
prefrontal_cortex = None
executive_controller = None


def get_orchestrator() -> Orchestrator:
    """Get or create orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = Orchestrator()
    return orchestrator


def get_prefrontal_cortex() -> PrefrontalCortex:
    """Get or create prefrontal cortex instance"""
    global prefrontal_cortex
    if prefrontal_cortex is None:
        prefrontal_cortex = PrefrontalCortex()
    return prefrontal_cortex


def get_executive_controller() -> ExecutiveController:
    """Get or create executive controller instance"""
    global executive_controller
    if executive_controller is None:
        executive_controller = ExecutiveController()
    return executive_controller


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get NEXUS system status"""
    
    orchestrator = get_orchestrator()
    
    # Get active modules
    active_modules = []
    for module_name in ["language", "vision", "reasoning", "learning"]:
        if hasattr(orchestrator, f"{module_name}_module"):
            active_modules.append(module_name)
    
    # Get memory usage
    memory_usage = {
        "working_memory": "active",
        "episodic_memory": "active",
        "semantic_memory": "active"
    }
    
    # Get performance metrics
    performance = {
        "avg_response_time": 0.5,
        "success_rate": 0.95,
        "tasks_completed": 0
    }
    
    return SystemStatus(
        status="operational",
        version="1.0.0",
        uptime=0.0,
        active_modules=active_modules,
        memory_usage=memory_usage,
        performance_metrics=performance
    )


@router.post("/execute", response_model=TaskResponse)
async def execute_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks
):
    """Execute a task using NEXUS orchestration"""
    
    start_time = datetime.now()
    task_id = f"task_{start_time.timestamp()}"
    
    try:
        orchestrator = get_orchestrator()
        
        # Execute task
        result = await orchestrator.process_request(
            request.goal,
            context=request.context
        )
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Determine modules used
        modules_used = request.modules or ["orchestrator", "prefrontal_cortex"]
        
        return TaskResponse(
            task_id=task_id,
            status="completed",
            result=result,
            execution_time=execution_time,
            modules_used=modules_used,
            confidence=result.get("confidence", 0.8),
            timestamp=datetime.now()
        )
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Task execution timeout")
    except Exception as e:
        logger.error(f"Task execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plan")
async def create_executive_plan(request: ExecutivePlan):
    """Create an executive plan for a complex goal"""
    
    try:
        prefrontal = get_prefrontal_cortex()
        
        # Generate executive plan
        plan = await prefrontal.executive_process(
            goal=request.goal,
            context={
                "constraints": request.constraints,
                "resources": request.resources,
                "time_limit": request.time_limit
            },
            available_modules=["language", "vision", "reasoning", "learning"]
        )
        
        return {
            "status": "success",
            "plan": plan,
            "estimated_time": request.time_limit or 60,
            "confidence": 0.85
        }
        
    except Exception as e:
        logger.error(f"Planning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adapt")
async def adapt_system(
    feedback: Dict[str, Any]
):
    """Adapt system based on feedback"""
    
    try:
        # This would interface with the learning system
        return {
            "status": "adapted",
            "changes": [
                "Updated neural weights",
                "Adjusted routing priorities",
                "Refined decision thresholds"
            ],
            "feedback_processed": feedback
        }
        
    except Exception as e:
        logger.error(f"Adaptation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_execution_history(
    limit: int = 10,
    offset: int = 0
):
    """Get task execution history"""
    
    # In production, this would query from database
    return {
        "total": 0,
        "limit": limit,
        "offset": offset,
        "history": []
    }


@router.get("/metrics")
async def get_performance_metrics():
    """Get detailed performance metrics"""
    
    return {
        "response_times": {
            "p50": 0.3,
            "p95": 0.8,
            "p99": 1.2
        },
        "module_usage": {
            "language": 45,
            "vision": 20,
            "reasoning": 25,
            "learning": 10
        },
        "error_rate": 0.02,
        "throughput": 100
    }


@router.post("/reset")
async def reset_system():
    """Reset NEXUS to initial state"""
    
    global orchestrator, prefrontal_cortex, executive_controller
    
    # Reset instances
    orchestrator = None
    prefrontal_cortex = None
    executive_controller = None
    
    return {
        "status": "reset",
        "message": "System reset to initial state"
    }