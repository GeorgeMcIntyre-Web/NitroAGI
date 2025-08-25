"""
Advanced Reasoning Module for NitroAGI NEXUS
Provides abstract reasoning, mathematical problem solving, and creative thinking
"""

from .abstract_reasoning import AbstractReasoner
from .mathematical_solver import MathematicalSolver
from .scientific_reasoning import ScientificReasoner
from .creative_thinking import CreativeThinkingEngine

__all__ = [
    "AbstractReasoner",
    "MathematicalSolver", 
    "ScientificReasoner",
    "CreativeThinkingEngine"
]