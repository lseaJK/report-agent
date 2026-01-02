"""Authentication and authorization management."""

import jwt
import bcrypt
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = "your-secret-key-here"  # Should be loaded from environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

security = HTTPBearer()


class AuthManager:
    """Authentication and authorization manager."""
    
    def __init__(self):
        """Initialize auth manager."""
        self._users: Dict[str, Dict[str, Any]] = {}
        self._refresh_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Create default admin user
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default users for testing."""
        # Create admin user
        admin_password = self._hash_password("admin123")
        self._users["admin"] = {
            "user_id": "admin",
            "username": "admin",
            "email": "admin@example.com",
            "full_name": "System Administrator",
            "password_hash": admin_password,
            "is_admin": True,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "last_login": None
        }
        
        # Create regular user
        user_password = self._hash_password("user123")
        self._users["user"] = {
            "user_id": "user",
            "username": "user",
            "email": "user@example.com",
            "full_name": "Regular User",
            "password_hash": user_password,
            "is_admin": False,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "last_login": None
        }
        
        logger.info("Created default users: admin/admin123, user/user123")
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    def _create_refresh_token(self, user_id: str) -> str:
        """Create refresh token."""
        token_data = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)  # Unique token ID
        }
        
        refresh_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        
        # Store refresh token
        self._refresh_tokens[token_data["jti"]] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": token_data["exp"]
        }
        
        return refresh_token
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return token data."""
        try:
            # Find user
            user = self._users.get(username)
            if not user:
                logger.warning(f"Authentication failed: user {username} not found")
                return None
            
            # Check if user is active
            if not user["is_active"]:
                logger.warning(f"Authentication failed: user {username} is inactive")
                return None
            
            # Verify password
            if not self._verify_password(password, user["password_hash"]):
                logger.warning(f"Authentication failed: invalid password for user {username}")
                return None
            
            # Update last login
            user["last_login"] = datetime.utcnow()
            
            # Create tokens
            access_token = self._create_access_token(
                data={"sub": user["user_id"], "username": username, "is_admin": user["is_admin"]}
            )
            refresh_token = self._create_refresh_token(user["user_id"])
            
            logger.info(f"User {username} authenticated successfully")
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
            }
            
        except Exception as e:
            logger.error(f"Error authenticating user {username}: {e}")
            return None
    
    async def refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Refresh access token using refresh token."""
        try:
            # Decode refresh token
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Verify token type
            if payload.get("type") != "refresh":
                logger.warning("Invalid token type for refresh")
                return None
            
            user_id = payload.get("user_id")
            token_id = payload.get("jti")
            
            if not user_id or not token_id:
                logger.warning("Invalid refresh token payload")
                return None
            
            # Check if refresh token exists and is valid
            if token_id not in self._refresh_tokens:
                logger.warning(f"Refresh token {token_id} not found")
                return None
            
            token_data = self._refresh_tokens[token_id]
            if token_data["user_id"] != user_id:
                logger.warning("Refresh token user mismatch")
                return None
            
            # Find user
            user = None
            for u in self._users.values():
                if u["user_id"] == user_id:
                    user = u
                    break
            
            if not user or not user["is_active"]:
                logger.warning(f"User {user_id} not found or inactive")
                return None
            
            # Create new access token
            access_token = self._create_access_token(
                data={"sub": user["user_id"], "username": user["username"], "is_admin": user["is_admin"]}
            )
            
            # Create new refresh token
            new_refresh_token = self._create_refresh_token(user["user_id"])
            
            # Remove old refresh token
            del self._refresh_tokens[token_id]
            
            logger.info(f"Token refreshed for user {user['username']}")
            
            return {
                "access_token": access_token,
                "refresh_token": new_refresh_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Refresh token expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Invalid refresh token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return None
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify access token and return user data."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Verify token type
            if payload.get("type") != "access":
                return None
            
            user_id = payload.get("sub")
            username = payload.get("username")
            
            if not user_id or not username:
                return None
            
            # Find user
            user = self._users.get(username)
            if not user or not user["is_active"]:
                return None
            
            return {
                "user_id": user_id,
                "username": username,
                "is_admin": user["is_admin"],
                "email": user["email"],
                "full_name": user["full_name"]
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Access token expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Invalid access token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None
    
    async def create_user(self, username: str, password: str, email: Optional[str] = None,
                         full_name: Optional[str] = None, is_admin: bool = False) -> bool:
        """Create a new user."""
        try:
            # Check if user already exists
            if username in self._users:
                logger.warning(f"User {username} already exists")
                return False
            
            # Create user
            user_id = hashlib.md5(username.encode()).hexdigest()[:12]
            password_hash = self._hash_password(password)
            
            self._users[username] = {
                "user_id": user_id,
                "username": username,
                "email": email,
                "full_name": full_name,
                "password_hash": password_hash,
                "is_admin": is_admin,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "last_login": None
            }
            
            logger.info(f"Created user: {username}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating user {username}: {e}")
            return False
    
    async def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        """Update user information."""
        try:
            if username not in self._users:
                return False
            
            user = self._users[username]
            
            # Apply updates
            for key, value in updates.items():
                if key in ["email", "full_name", "is_admin", "is_active"]:
                    user[key] = value
            
            logger.info(f"Updated user: {username}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating user {username}: {e}")
            return False
    
    async def change_password(self, username: str, current_password: str, new_password: str) -> bool:
        """Change user password."""
        try:
            user = self._users.get(username)
            if not user:
                return False
            
            # Verify current password
            if not self._verify_password(current_password, user["password_hash"]):
                logger.warning(f"Password change failed: invalid current password for user {username}")
                return False
            
            # Update password
            user["password_hash"] = self._hash_password(new_password)
            
            logger.info(f"Password changed for user: {username}")
            return True
            
        except Exception as e:
            logger.error(f"Error changing password for user {username}: {e}")
            return False
    
    async def delete_user(self, username: str) -> bool:
        """Delete a user."""
        try:
            if username not in self._users:
                return False
            
            # Don't allow deleting admin user
            if self._users[username]["is_admin"] and username == "admin":
                logger.warning("Cannot delete default admin user")
                return False
            
            del self._users[username]
            
            logger.info(f"Deleted user: {username}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting user {username}: {e}")
            return False
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information."""
        user = self._users.get(username)
        if user:
            # Return user data without password hash
            return {
                "user_id": user["user_id"],
                "username": user["username"],
                "email": user["email"],
                "full_name": user["full_name"],
                "is_admin": user["is_admin"],
                "is_active": user["is_active"],
                "created_at": user["created_at"],
                "last_login": user["last_login"]
            }
        return None
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users."""
        users = []
        for user in self._users.values():
            users.append({
                "user_id": user["user_id"],
                "username": user["username"],
                "email": user["email"],
                "full_name": user["full_name"],
                "is_admin": user["is_admin"],
                "is_active": user["is_active"],
                "created_at": user["created_at"],
                "last_login": user["last_login"]
            })
        return users
    
    async def revoke_refresh_token(self, token_id: str) -> bool:
        """Revoke a refresh token."""
        try:
            if token_id in self._refresh_tokens:
                del self._refresh_tokens[token_id]
                logger.info(f"Revoked refresh token: {token_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error revoking refresh token {token_id}: {e}")
            return False
    
    def cleanup_expired_tokens(self):
        """Clean up expired refresh tokens."""
        try:
            current_time = datetime.utcnow()
            expired_tokens = [
                token_id for token_id, data in self._refresh_tokens.items()
                if data["expires_at"] < current_time
            ]
            
            for token_id in expired_tokens:
                del self._refresh_tokens[token_id]
            
            if expired_tokens:
                logger.info(f"Cleaned up {len(expired_tokens)} expired refresh tokens")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired tokens: {e}")


# Global auth manager instance
auth_manager = AuthManager()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user."""
    try:
        token = credentials.credentials
        user_data = await auth_manager.verify_token(token)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get current user and verify admin privileges."""
    if not current_user.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return current_user


def require_permissions(*permissions: str):
    """Decorator to require specific permissions."""
    def decorator(func):
        async def wrapper(*args, current_user: Dict[str, Any] = Depends(get_current_user), **kwargs):
            # In a real implementation, this would check user permissions
            # For now, we'll just check if user is admin for any permission
            if permissions and not current_user.get("is_admin"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required permissions: {', '.join(permissions)}"
                )
            
            return await func(*args, current_user=current_user, **kwargs)
        
        return wrapper
    return decorator