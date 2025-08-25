"""Request logging middleware."""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from nitroagi.utils.logging import get_logger, audit_log


logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response details.
        
        Args:
            request: Incoming request
            call_next: Next middleware in chain
            
        Returns:
            Response
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request received",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
            query_params=dict(request.query_params),
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        
        # Audit log for important endpoints
        if request.url.path.startswith("/api/v1/chat") or \
           request.url.path.startswith("/api/v1/modules"):
            audit_log(
                "api_request",
                user=request.headers.get("X-User-ID", "anonymous"),
                resource=request.url.path,
                action=request.method,
                result=f"status_{response.status_code}",
                request_id=request_id,
                duration_ms=duration_ms,
            )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(duration_ms, 2))
        
        return response