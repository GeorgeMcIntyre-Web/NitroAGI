"""API dependencies for dependency injection."""

from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from nitroagi.utils.config import get_config
from nitroagi.utils.logging import get_logger
from nitroagi.api.middleware.auth import verify_token


logger = get_logger(__name__)
config = get_config()

# Security
security = HTTPBearer()

# Database engine (lazy initialization)
_engine = None
_async_session = None


def get_engine():
    """Get or create database engine."""
    global _engine, _async_session
    
    if _engine is None:
        database_url = config.database.database_url
        # Convert to async URL
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        
        _engine = create_async_engine(
            database_url,
            echo=config.debug,
            pool_pre_ping=True,
            pool_size=20,
            max_overflow=40,
        )
        _async_session = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    return _engine, _async_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session.
    
    Yields:
        Database session
    """
    _, async_session = get_engine()
    
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Redis client (lazy initialization)
_redis_client = None


async def get_redis() -> redis.Redis:
    """Get Redis client.
    
    Returns:
        Redis client instance
    """
    global _redis_client
    
    if _redis_client is None:
        _redis_client = await redis.from_url(
            config.database.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
    
    return _redis_client


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Get current authenticated user.
    
    Args:
        credentials: Bearer token credentials
        
    Returns:
        User information from token
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {
        "user_id": payload.get("sub"),
        "roles": payload.get("roles", []),
        "exp": payload.get("exp")
    }


async def require_admin(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """Require admin role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


class RateLimiter:
    """Rate limiter dependency."""
    
    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Request limit per minute
        """
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    async def __call__(self, request) -> bool:
        """Check if request should be rate limited.
        
        Args:
            request: FastAPI request
            
        Returns:
            True if request is allowed
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        import time
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old entries
        self.requests = {
            ip: times for ip, times in self.requests.items()
            if any(t > current_time - 60 for t in times)
        }
        
        # Check rate limit
        if client_ip in self.requests:
            recent_requests = [
                t for t in self.requests[client_ip]
                if t > current_time - 60
            ]
            
            if len(recent_requests) >= self.requests_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            self.requests[client_ip] = recent_requests + [current_time]
        else:
            self.requests[client_ip] = [current_time]
        
        return True


# Create rate limiter instances
rate_limiter_strict = RateLimiter(requests_per_minute=10)
rate_limiter_normal = RateLimiter(requests_per_minute=60)
rate_limiter_relaxed = RateLimiter(requests_per_minute=600)