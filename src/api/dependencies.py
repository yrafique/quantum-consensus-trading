"""
API Dependencies
===============

FastAPI dependency providers for authentication, authorization, and common services.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt

from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from ..core.config import get_config
from ..core.exceptions import BaseRiverException

# Create specific exception classes for authentication
class AuthenticationException(BaseRiverException):
    """Authentication failed"""
    pass

class AuthorizationException(BaseRiverException):
    """Authorization failed"""
    pass

logger = structlog.get_logger(__name__)

# Security scheme
security_scheme = HTTPBearer(auto_error=False)


class User:
    """User model for authentication"""
    
    def __init__(
        self,
        user_id: str,
        username: str,
        email: str,
        roles: list = None,
        is_active: bool = True
    ):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.roles = roles or []
        self.is_active = is_active
        self.authenticated_at = datetime.utcnow()
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role"""
        return role in self.roles
    
    def is_admin(self) -> bool:
        """Check if user has admin privileges"""
        return "admin" in self.roles or "superuser" in self.roles
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "is_active": self.is_active,
            "authenticated_at": self.authenticated_at.isoformat()
        }


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)
) -> Optional[User]:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        Current user or None if not authenticated
        
    Raises:
        HTTPException: If authentication fails
    """
    config = get_config()
    
    # Skip authentication if disabled
    if not config.security.enable_authentication:
        # Return a default user for development
        return User(
            user_id="dev_user",
            username="developer",
            email="dev@rivertrading.ai",
            roles=["user", "admin"],
            is_active=True
        )
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            credentials.credentials,
            config.security.jwt_secret_key,
            algorithms=["HS256"]
        )
        
        # Extract user information
        user_id = payload.get("user_id")
        username = payload.get("username")
        email = payload.get("email")
        roles = payload.get("roles", [])
        
        if not user_id or not username:
            raise AuthenticationException("Invalid token payload")
        
        # Check token expiration
        exp = payload.get("exp")
        if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
            raise AuthenticationException("Token has expired")
        
        # Create user object
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            is_active=True
        )
        
        logger.info(
            "User authenticated successfully",
            user_id=user_id,
            username=username,
            roles=roles
        )
        
        return user
        
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired", token=credentials.credentials[:20] + "...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    except jwt.InvalidTokenError as exc:
        logger.warning(
            "Invalid JWT token",
            error=str(exc),
            token=credentials.credentials[:20] + "..."
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    except Exception as exc:
        logger.error(
            "Authentication error",
            error=str(exc),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_admin_access(
    current_user: User = Depends(get_current_user)
) -> bool:
    """
    Verify user has admin access.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        True if user has admin access
        
    Raises:
        HTTPException: If user lacks admin privileges
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    if not current_user.is_admin():
        logger.warning(
            "Admin access denied",
            user_id=current_user.user_id,
            username=current_user.username,
            roles=current_user.roles
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return True


async def verify_api_key(
    api_key: Optional[str] = None
) -> bool:
    """
    Verify API key for external service access.
    
    Args:
        api_key: API key to verify
        
    Returns:
        True if API key is valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    config = get_config()
    
    # Skip API key verification in development
    if config.environment.value == "development":
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    # In production, validate against stored API keys
    # This is a simplified implementation
    valid_api_keys = [
        # Load from database or configuration
    ]
    
    if api_key not in valid_api_keys:
        logger.warning("Invalid API key attempt", api_key=api_key[:8] + "...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return True


def require_role(required_role: str):
    """
    Dependency factory for role-based access control.
    
    Args:
        required_role: Role required to access endpoint
        
    Returns:
        Dependency function that checks user role
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        if not current_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive"
            )
        
        if not current_user.has_role(required_role):
            logger.warning(
                "Role access denied",
                user_id=current_user.user_id,
                required_role=required_role,
                user_roles=current_user.roles
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        
        return current_user
    
    return role_checker


def generate_jwt_token(user: User) -> str:
    """
    Generate JWT token for user.
    
    Args:
        user: User to generate token for
        
    Returns:
        JWT token string
    """
    config = get_config()
    
    # Token payload
    payload = {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "roles": user.roles,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=config.security.jwt_expiration_hours)
    }
    
    # Generate token
    token = jwt.encode(
        payload,
        config.security.jwt_secret_key,
        algorithm="HS256"
    )
    
    logger.info(
        "JWT token generated",
        user_id=user.user_id,
        username=user.username,
        expires_at=(payload["exp"]).isoformat()
    )
    
    return token


class RateLimiter:
    """Rate limiting dependency"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
    
    async def __call__(self, request) -> bool:
        """
        Check rate limit for request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if request is within rate limit
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        # Implementation would use Redis or similar for rate limiting
        # This is a placeholder
        return True


# Common dependency instances
admin_required = Depends(verify_admin_access)
trader_role_required = require_role("trader")
analyst_role_required = require_role("analyst")
rate_limit_60 = RateLimiter(60)
rate_limit_30 = RateLimiter(30)
rate_limit_10 = RateLimiter(10)