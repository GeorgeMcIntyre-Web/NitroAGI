"""
Reasoning API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import asyncio
import logging

from src.nitroagi.modules.advanced_reasoning.abstract_reasoning import AbstractReasoner, ReasoningType
from src.nitroagi.modules.advanced_reasoning.mathematical_solver import MathematicalSolver
from src.nitroagi.modules.advanced_reasoning.scientific_reasoning import ScientificReasoner
from src.nitroagi.modules.advanced_reasoning.creative_thinking import CreativeThinkingEngine, CreativeStrategy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reasoning")


class AbstractReasoningRequest(BaseModel):
    """Abstract reasoning request"""
    input_data: Dict[str, Any]
    reasoning_type: Optional[str] = Field(None, description="Type of reasoning to apply")


class MathProblemRequest(BaseModel):
    """Mathematical problem request"""
    problem: str = Field(..., description="Mathematical problem to solve")
    problem_type: Optional[str] = Field(None, description="Type of math problem")
    show_steps: bool = Field(True, description="Show solution steps")


class ScientificResearchRequest(BaseModel):
    """Scientific research request"""
    observations: List[str] = Field(..., description="Scientific observations")
    domain: str = Field("general", description="Scientific domain")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Research constraints")


class CreativeRequest(BaseModel):
    """Creative thinking request"""
    problem_description: str = Field(..., description="Problem requiring creative solution")
    constraints: Optional[List[str]] = Field(None, description="Constraints")
    goals: Optional[List[str]] = Field(None, description="Goals to achieve")
    strategy: Optional[str] = Field(None, description="Creative strategy to use")


# Global instances
abstract_reasoner = None
math_solver = None
scientific_reasoner = None
creative_engine = None


def get_abstract_reasoner() -> AbstractReasoner:
    """Get or create abstract reasoner"""
    global abstract_reasoner
    if abstract_reasoner is None:
        abstract_reasoner = AbstractReasoner()
    return abstract_reasoner


def get_math_solver() -> MathematicalSolver:
    """Get or create mathematical solver"""
    global math_solver
    if math_solver is None:
        math_solver = MathematicalSolver()
    return math_solver


def get_scientific_reasoner() -> ScientificReasoner:
    """Get or create scientific reasoner"""
    global scientific_reasoner
    if scientific_reasoner is None:
        scientific_reasoner = ScientificReasoner()
    return scientific_reasoner


def get_creative_engine() -> CreativeThinkingEngine:
    """Get or create creative thinking engine"""
    global creative_engine
    if creative_engine is None:
        creative_engine = CreativeThinkingEngine()
    return creative_engine


@router.post("/abstract")
async def abstract_reasoning(request: AbstractReasoningRequest):
    """Perform abstract reasoning"""
    
    try:
        reasoner = get_abstract_reasoner()
        
        # Convert string type to enum if provided
        reasoning_type = None
        if request.reasoning_type:
            try:
                reasoning_type = ReasoningType(request.reasoning_type)
            except ValueError:
                pass
        
        result = await reasoner.reason(
            request.input_data,
            reasoning_type
        )
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Abstract reasoning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/solve-math")
async def solve_mathematical_problem(request: MathProblemRequest):
    """Solve mathematical problem"""
    
    try:
        solver = get_math_solver()
        
        result = await solver.solve(
            request.problem,
            request.problem_type,
            request.show_steps
        )
        
        return {
            "status": "success",
            "solution": result
        }
        
    except Exception as e:
        logger.error(f"Math solving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-math")
async def solve_multiple_problems(problems: List[str]):
    """Solve multiple mathematical problems"""
    
    try:
        solver = get_math_solver()
        
        solutions = await solver.batch_solve(problems)
        
        return {
            "status": "success",
            "solutions": solutions,
            "total": len(solutions)
        }
        
    except Exception as e:
        logger.error(f"Batch math solving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scientific-research")
async def conduct_scientific_research(request: ScientificResearchRequest):
    """Conduct scientific research process"""
    
    try:
        reasoner = get_scientific_reasoner()
        
        result = await reasoner.conduct_research(
            request.observations,
            request.domain,
            request.constraints
        )
        
        return {
            "status": "success",
            "research": result
        }
        
    except Exception as e:
        logger.error(f"Scientific research failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/literature-review")
async def review_literature(
    topic: str,
    domain: str = "general"
):
    """Review existing knowledge on topic"""
    
    try:
        reasoner = get_scientific_reasoner()
        
        result = await reasoner.review_literature(topic, domain)
        
        return {
            "status": "success",
            "review": result
        }
        
    except Exception as e:
        logger.error(f"Literature review failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/creative-solve")
async def solve_creatively(request: CreativeRequest):
    """Solve problem using creative thinking"""
    
    try:
        engine = get_creative_engine()
        
        # Convert string strategy to enum if provided
        strategy = None
        if request.strategy:
            try:
                strategy = CreativeStrategy(request.strategy)
            except ValueError:
                pass
        
        solution = await engine.solve_creatively(
            request.problem_description,
            request.constraints,
            request.goals,
            strategy
        )
        
        return {
            "status": "success",
            "solution": {
                "selected_idea": solution.selected_idea.content if solution.selected_idea else None,
                "all_ideas": [idea.content for idea in solution.ideas],
                "implementation_plan": solution.implementation_plan,
                "innovation_level": solution.innovation_level,
                "confidence": solution.confidence
            }
        }
        
    except Exception as e:
        logger.error(f"Creative solving failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/design-thinking")
async def apply_design_thinking(
    problem_description: str,
    context: Optional[Dict[str, Any]] = None
):
    """Apply design thinking process"""
    
    try:
        engine = get_creative_engine()
        
        result = await engine.design_thinking_process(
            problem_description,
            context
        )
        
        return {
            "status": "success",
            "design_process": result
        }
        
    except Exception as e:
        logger.error(f"Design thinking failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities")
async def get_reasoning_capabilities():
    """Get available reasoning capabilities"""
    
    abstract_reasoner = get_abstract_reasoner()
    math_solver = get_math_solver()
    scientific_reasoner = get_scientific_reasoner()
    creative_engine = get_creative_engine()
    
    return {
        "abstract": abstract_reasoner.get_reasoning_capabilities(),
        "mathematical": math_solver.get_capabilities(),
        "scientific": scientific_reasoner.get_research_capabilities(),
        "creative": creative_engine.get_creative_capabilities()
    }


@router.post("/pattern-recognition")
async def recognize_patterns(data: List[Any]):
    """Recognize patterns in data"""
    
    try:
        reasoner = get_abstract_reasoner()
        
        pattern = await reasoner.pattern_recognizer.recognize_pattern(data)
        
        if pattern:
            next_elements = reasoner.pattern_recognizer.extend_pattern(pattern, 3)
            
            return {
                "status": "success",
                "pattern": {
                    "type": pattern.pattern_type,
                    "rules": pattern.rules,
                    "confidence": pattern.confidence,
                    "next_elements": next_elements
                }
            }
        else:
            return {
                "status": "success",
                "pattern": None,
                "message": "No clear pattern detected"
            }
            
    except Exception as e:
        logger.error(f"Pattern recognition failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analogy")
async def find_analogies(
    source: Dict[str, Any],
    target_domain: str
):
    """Find analogies between domains"""
    
    try:
        reasoner = get_abstract_reasoner()
        
        analogy = await reasoner.analogy_engine.find_analogy(
            source,
            target_domain
        )
        
        if analogy:
            return {
                "status": "success",
                "analogy": {
                    "source_domain": analogy.source_domain,
                    "target_domain": analogy.target_domain,
                    "mappings": analogy.mappings,
                    "similarity_score": analogy.similarity_score
                }
            }
        else:
            return {
                "status": "success",
                "analogy": None,
                "message": "No suitable analogy found"
            }
            
    except Exception as e:
        logger.error(f"Analogy finding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))