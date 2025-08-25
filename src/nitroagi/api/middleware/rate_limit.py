"""Rate limiting middleware for API endpoints."""

import time
from collections import defaultdict, deque
from typing import Dict, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from nitroagi.utils.logging import get_logger


logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        """Initialize rate limiter.
        
        Args:
            app: FastAPI application
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Track requests per IP
        self.minute_requests: Dict[str, deque] = defaultdict(deque)
        self.hour_requests: Dict[str, deque] = defaultdict(deque)
    
    async def dispatch(self, request: Request, call_next):
        """Process the request with rate limiting.
        
        Args:
            request: Incoming request
            call_next: Next middleware in chain
            
        Returns:
            Response or rate limit error
        """
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        current_time = time.time()
        
        # Check minute limit
        minute_queue = self.minute_requests[client_ip]
        self._clean_queue(minute_queue, current_time - 60)
        
        if len(minute_queue) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded (minute) for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute"
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + 60))
                }
            )
        
        # Check hour limit
        hour_queue = self.hour_requests[client_ip]
        self._clean_queue(hour_queue, current_time - 3600)
        
        if len(hour_queue) >= self.requests_per_hour:
            logger.warning(f"Rate limit exceeded (hour) for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_hour} requests per hour"
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_hour),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + 3600))
                }
            )
        
        # Add current request to queues
        minute_queue.append(current_time)
        hour_queue.append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(minute_queue)
        )
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response
    
    def _clean_queue(self, queue: deque, cutoff_time: float) -> None:
        """Remove old entries from queue.
        
        Args:
            queue: Request queue
            cutoff_time: Time before which to remove entries
        """
        while queue and queue[0] < cutoff_time:
            queue.popleft()