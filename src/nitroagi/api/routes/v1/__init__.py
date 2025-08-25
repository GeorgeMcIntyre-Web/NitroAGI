"""
API v1 Routes for NitroAGI NEXUS
"""

from fastapi import APIRouter
from .nexus import router as nexus_router
from .reasoning import router as reasoning_router
from .learning import router as learning_router
from .multimodal import router as multimodal_router
from .modules import router as modules_router

router = APIRouter(prefix="/v1")

# Include all routers
router.include_router(nexus_router, tags=["NEXUS Core"])
router.include_router(reasoning_router, tags=["Reasoning"])
router.include_router(learning_router, tags=["Learning"])
router.include_router(multimodal_router, tags=["Multi-Modal"])
router.include_router(modules_router, tags=["Modules"])

__all__ = ["router"]