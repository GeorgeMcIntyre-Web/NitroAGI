"""Authentication middleware."""

from typing import Optional

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware

from nitroagi.utils.config import get_config
from nitroagi.utils.logging import get_logger


logger = get_logger(__name__)
config = get_config()

security = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """JWT authentication middleware."""
    
    def __init__(self, app, protected_paths: list = None):
        """Initialize auth middleware.
        
        Args:
            app: FastAPI application
            protected_paths: List of paths requiring authentication
        """
        super().__init__(app)
        self.protected_paths = protected_paths or ["/api/v1/"]
    
    async def dispatch(self, request: Request, call_next):
        """Check authentication for protected paths.
        
        Args:
            request: Incoming request
            call_next: Next middleware in chain
            
        Returns:
            Response or authentication error
        """
        # Check if path requires authentication
        path = request.url.path
        requires_auth = any(path.startswith(p) for p in self.protected_paths)
        
        # Skip auth for public endpoints
        if not requires_auth or path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            logger.warning(f"Missing authorization header for {path}")
            return HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Extract and verify token
        token = auth_header.split(" ")[1]
        
        try:
            payload = jwt.decode(
                token,
                config.security.jwt_secret_key,
                algorithms=[config.security.jwt_algorithm]
            )
            
            # Add user info to request state
            request.state.user_id = payload.get("sub")
            request.state.user_roles = payload.get("roles", [])
            
        except JWTError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Process request
        response = await call_next(request)
        return response


def create_access_token(user_id: str, roles: list = None) -> str:
    """Create a JWT access token.
    
    Args:
        user_id: User identifier
        roles: User roles
        
    Returns:
        JWT token string
    """
    import time
    
    payload = {
        "sub": user_id,
        "roles": roles or [],
        "exp": time.time() + config.security.jwt_expiration_delta_seconds,
        "iat": time.time(),
    }
    
    token = jwt.encode(
        payload,
        config.security.jwt_secret_key,
        algorithm=config.security.jwt_algorithm
    )
    
    return token


def verify_token(token: str) -> Optional[dict]:
    """Verify a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            config.security.jwt_secret_key,
            algorithms=[config.security.jwt_algorithm]
        )
        return payload
    except JWTError:
        return None