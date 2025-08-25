"""
Learning System API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from src.nitroagi.core.learning import (
    LearningOrchestrator,
    State,
    FeedbackLoop
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/learning")


class LearningInteraction(BaseModel):
    """Learning interaction data"""
    state: Dict[str, Any] = Field(..., description="Current state")
    action: str = Field(..., description="Action taken")
    reward: float = Field(..., description="Reward received")
    next_state: Dict[str, Any] = Field(..., description="Next state")
    done: bool = Field(False, description="Episode complete")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    """Performance feedback"""
    feedback_type: str = Field(..., description="Type of feedback")
    value: float = Field(..., description="Feedback value")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AdaptationRequest(BaseModel):
    """Task adaptation request"""
    task_type: str = Field(..., description="Type of task")
    context: Dict[str, Any] = Field(..., description="Task context")


# Global instances
learning_orchestrator = None
feedback_loop = None


def get_learning_orchestrator() -> LearningOrchestrator:
    """Get or create learning orchestrator"""
    global learning_orchestrator
    if learning_orchestrator is None:
        learning_orchestrator = LearningOrchestrator()
    return learning_orchestrator


def get_feedback_loop() -> FeedbackLoop:
    """Get or create feedback loop"""
    global feedback_loop
    if feedback_loop is None:
        orchestrator = get_learning_orchestrator()
        feedback_loop = FeedbackLoop(orchestrator)
    return feedback_loop


@router.post("/learn")
async def learn_from_interaction(interaction: LearningInteraction):
    """Learn from a single interaction"""
    
    try:
        orchestrator = get_learning_orchestrator()
        
        # Convert dict to State objects
        state = State(
            task_complexity=interaction.state.get("complexity", 0.5),
            available_resources=interaction.state.get("resources", {}),
            module_performance=interaction.state.get("performance", {}),
            context_embedding=None,
            temporal_features={},
            goal_alignment=interaction.state.get("goal_alignment", 0.5),
            uncertainty=interaction.state.get("uncertainty", 0.5)
        )
        
        next_state = State(
            task_complexity=interaction.next_state.get("complexity", 0.5),
            available_resources=interaction.next_state.get("resources", {}),
            module_performance=interaction.next_state.get("performance", {}),
            context_embedding=None,
            temporal_features={},
            goal_alignment=interaction.next_state.get("goal_alignment", 0.5),
            uncertainty=interaction.next_state.get("uncertainty", 0.5)
        )
        
        await orchestrator.learn_from_interaction(
            state=state,
            action=interaction.action,
            reward=interaction.reward,
            next_state=next_state,
            done=interaction.done,
            context=interaction.context
        )
        
        return {
            "status": "learned",
            "buffer_size": len(orchestrator.rl_agent.replay_buffer.buffer),
            "epsilon": orchestrator.rl_agent.epsilon
        }
        
    except Exception as e:
        logger.error(f"Learning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-learn")
async def batch_learning(
    batch_size: int = 128
):
    """Perform batch learning"""
    
    try:
        orchestrator = get_learning_orchestrator()
        
        await orchestrator.batch_learning(batch_size)
        
        return {
            "status": "batch_learned",
            "batch_size": batch_size,
            "learning_status": orchestrator.get_learning_status()
        }
        
    except Exception as e:
        logger.error(f"Batch learning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def process_feedback(feedback: FeedbackRequest):
    """Process performance feedback"""
    
    try:
        loop = get_feedback_loop()
        
        adjustments = await loop.process_feedback(
            feedback.feedback_type,
            feedback.value,
            feedback.context
        )
        
        return {
            "status": "feedback_processed",
            "adjustments": adjustments,
            "feedback_history_size": len(loop.feedback_history)
        }
        
    except Exception as e:
        logger.error(f"Feedback processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adapt")
async def adapt_to_task(request: AdaptationRequest):
    """Adapt to new task type"""
    
    try:
        orchestrator = get_learning_orchestrator()
        
        adapted_params = await orchestrator.meta_learner.adapt_to_task(
            request.task_type,
            request.context
        )
        
        return {
            "status": "adapted",
            "adapted_parameters": adapted_params,
            "adaptations_count": len(orchestrator.meta_learner.adaptation_history)
        }
        
    except Exception as e:
        logger.error(f"Adaptation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_learning_status():
    """Get comprehensive learning status"""
    
    try:
        orchestrator = get_learning_orchestrator()
        
        status = orchestrator.get_learning_status()
        
        return {
            "status": "operational",
            "learning_status": status
        }
        
    except Exception as e:
        logger.error(f"Status retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_metrics():
    """Get performance metrics"""
    
    try:
        orchestrator = get_learning_orchestrator()
        
        metrics = orchestrator.performance_monitor.get_recent_performance()
        summary = orchestrator.performance_monitor.get_summary()
        
        return {
            "recent_performance": metrics,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Performance retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skills")
async def get_skill_levels():
    """Get current skill levels"""
    
    try:
        orchestrator = get_learning_orchestrator()
        
        skills = orchestrator.continuous_learner.skill_registry
        
        return {
            "skills": skills,
            "total_skills": len(skills),
            "average_level": sum(s["level"] for s in skills.values()) / len(skills) if skills else 0
        }
        
    except Exception as e:
        logger.error(f"Skills retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable")
async def enable_learning(
    mode: str = "online"
):
    """Enable learning with specified mode"""
    
    try:
        orchestrator = get_learning_orchestrator()
        
        orchestrator.learning_enabled = True
        orchestrator.adaptation_mode = mode
        
        return {
            "status": "enabled",
            "mode": mode
        }
        
    except Exception as e:
        logger.error(f"Enable learning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable")
async def disable_learning():
    """Disable learning"""
    
    try:
        orchestrator = get_learning_orchestrator()
        
        orchestrator.learning_enabled = False
        
        return {
            "status": "disabled"
        }
        
    except Exception as e:
        logger.error(f"Disable learning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save")
async def save_learning_state(
    path: str = "./checkpoints"
):
    """Save learning system state"""
    
    try:
        orchestrator = get_learning_orchestrator()
        
        orchestrator.save_state(path)
        
        return {
            "status": "saved",
            "path": path
        }
        
    except Exception as e:
        logger.error(f"Save state failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_learning_state(
    path: str = "./checkpoints"
):
    """Load learning system state"""
    
    try:
        orchestrator = get_learning_orchestrator()
        
        orchestrator.load_state(path)
        
        return {
            "status": "loaded",
            "path": path
        }
        
    except Exception as e:
        logger.error(f"Load state failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge")
async def get_knowledge_base():
    """Get current knowledge base"""
    
    try:
        orchestrator = get_learning_orchestrator()
        
        knowledge = orchestrator.continuous_learner.knowledge_base
        
        return {
            "knowledge_items": len(knowledge),
            "knowledge_sample": dict(list(knowledge.items())[:10])  # First 10 items
        }
        
    except Exception as e:
        logger.error(f"Knowledge retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_learning():
    """Reset learning system"""
    
    global learning_orchestrator, feedback_loop
    
    learning_orchestrator = None
    feedback_loop = None
    
    return {
        "status": "reset",
        "message": "Learning system reset to initial state"
    }