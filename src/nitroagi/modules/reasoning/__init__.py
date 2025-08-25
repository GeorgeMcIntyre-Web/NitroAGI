"""
Reasoning Module for NitroAGI NEXUS
Provides symbolic AI, logic programming, and causal reasoning capabilities
"""

from .reasoning_module import ReasoningModule
from .engines import (
    LogicEngine,
    CausalReasoner,
    KnowledgeGraph,
    ProblemSolver,
    InferenceEngine
)

__all__ = [
    "ReasoningModule",
    "LogicEngine",
    "CausalReasoner",
    "KnowledgeGraph",
    "ProblemSolver",
    "InferenceEngine"
]