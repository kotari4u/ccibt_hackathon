"""
JWT authentication for API endpoints.
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from src.utils.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Exclude docs endpoints from authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    auto_error=False,  # Don't auto-raise errors, allows optional auth
)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


async def get_current_user(
    request: Optional[Request] = None,
    token: Optional[str] = None
) -> dict:
    """
    Get current user - Authentication disabled for all endpoints.
    
    Returns anonymous user for all requests.
    """
    # Authentication disabled - return anonymous user
    return {"username": "anonymous", "email": "anonymous@example.com"}


# Optional: Simple API key authentication for easier testing
async def verify_api_key(api_key: Optional[str] = None) -> bool:
    """
    Verify API key (simplified for demo).
    
    In production, validate against database.
    """
    # For demo purposes, accept any non-empty key
    # In production, validate against stored API keys
    return api_key is not None and len(api_key) > 0

