"""FastAPI application factory for NitroAGI."""

from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from nitroagi.api.routes import health, chat, vision, memory, modules, system
from nitroagi.api.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)
from nitroagi.core import MessageBus, Orchestrator, ModuleRegistry
from nitroagi.core.memory import MemoryManager
from nitroagi.core.network import NetworkManager
from nitroagi.utils.config import get_config
from nitroagi.utils.logging import get_logger
from nitroagi.core.exceptions import NitroAGIException


logger = get_logger(__name__)
config = get_config()

# Global instances
message_bus: MessageBus = None
orchestrator: Orchestrator = None
registry: ModuleRegistry = None
memory_manager: MemoryManager = None
network_manager: NetworkManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global message_bus, orchestrator, registry, memory_manager, network_manager
    
    logger.info("Starting NitroAGI API...")
    
    # Initialize core components
    message_bus = MessageBus()
    registry = ModuleRegistry()
    orchestrator = Orchestrator(registry, message_bus)
    memory_manager = MemoryManager()
    network_manager = NetworkManager()
    
    # Start components
    await message_bus.start()
    await orchestrator.start()
    await memory_manager.initialize(config.database.redis_url)
    await network_manager.start()
    
    # Store in app state
    app.state.message_bus = message_bus
    app.state.orchestrator = orchestrator
    app.state.registry = registry
    app.state.memory_manager = memory_manager
    app.state.network_manager = network_manager
    
    logger.info("NitroAGI API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NitroAGI API...")
    
    await network_manager.stop()
    await memory_manager.shutdown()
    await orchestrator.stop()
    await message_bus.stop()
    await registry.shutdown_all()
    
    logger.info("NitroAGI API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Create app with lifespan management
    app = FastAPI(
        title="NitroAGI",
        description="Multi-Modal AI System with Brain-Inspired Architecture",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
        openapi_url="/openapi.json" if config.debug else None,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=config.api.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add trusted host validation in production
    if config.is_production():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*.nitroagi.dev", "nitroagi.dev"]
        )
    
    # Add request logging
    if config.debug:
        app.add_middleware(RequestLoggingMiddleware)
    
    # Add rate limiting
    if config.api.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=config.api.rate_limit_per_minute,
            requests_per_hour=config.api.rate_limit_per_hour,
        )
    
    # Add Prometheus metrics
    if config.monitoring.prometheus_enabled:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
    
    # Exception handlers
    @app.exception_handler(NitroAGIException)
    async def nitroagi_exception_handler(request: Request, exc: NitroAGIException):
        """Handle NitroAGI exceptions."""
        return JSONResponse(
            status_code=400,
            content=exc.to_dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        if config.debug:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": str(exc),
                    "type": type(exc).__name__
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Internal Server Error"}
            )
    
    # Include routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
    app.include_router(vision.router, prefix="/api/v1/vision", tags=["vision"])
    app.include_router(memory.router, prefix="/api/v1/memory", tags=["memory"])
    app.include_router(modules.router, prefix="/api/v1/modules", tags=["modules"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "NitroAGI",
            "version": "0.1.0",
            "status": "running",
            "environment": config.environment,
            "features": {
                "vision": config.features.vision_module,
                "reasoning": config.features.reasoning_engine,
                "learning": config.features.learning_agent,
                "multi_modal": config.features.multi_modal,
                "6g_ready": True,
            }
        }
    
    return app