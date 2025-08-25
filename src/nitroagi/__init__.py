"""
NitroAGI: Multi-Modal AI System

A brain-inspired artificial intelligence architecture combining specialized AI models
for enhanced cognitive capabilities.
"""

__version__ = "0.1.0"
__author__ = "George McIntyre"
__email__ = "george@nitroagi.dev"

from nitroagi.core.base import AIModule, ModuleConfig
from nitroagi.core.orchestrator import Orchestrator
from nitroagi.core.message_bus import MessageBus

__all__ = [
    "AIModule",
    "ModuleConfig",
    "Orchestrator",
    "MessageBus",
    "__version__",
]