"""
Authentication and Authorization System
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import jwt
import bcrypt
import secrets
import uuid

from app.core.config import settings
from app.core.database import get_redis

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    AGENT_MANAGER = "agent_manager"
    DEVELOPER = "developer"
    USER = "user"
    AGENT = "agent"  # For agent-to-agent authentication


class Permission(str, Enum):
    """System permissions"""
    # Agent management
    CREATE_AGENT = "create_agent"
    DELETE_AGENT = "delete_agent"
    MANAGE_AGENTS = "manage_agents"
    VIEW_AGENTS = "view_agents"
    
    # LLM operations
    USE_LLM = "use_llm"
    MANAGE_LLM_PROVIDERS = "manage_llm_providers"
    
    # Memory operations
    READ_MEMORY = "read_memory"
    WRITE_MEMORY = "write_memory"
    DELETE_MEMORY = "delete_memory"
    
    # System administration
    ADMIN_ACCESS = "admin_access"
    VIEW_LOGS = "view_logs"
    MANAGE_USERS = "manage_users"
    
    # API access
    API_ACCESS = "api_access"
    WEBHOOK_ACCESS = "webhook_access"


class User(BaseModel):
    """User model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: List[Permission] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class APIKey(BaseModel):
    """API key model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    key_hash: str
    user_id: str
    permissions: List[Permission] = Field(default_factory=list)
    is_active: bool = True
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """User session model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuthenticationSystem:
    """Authentication and authorization system"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.sessions: Dict[str, Session] = {}
        self.redis_client = None
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: [p for p in Permission],  # All permissions
            UserRole.AGENT_MANAGER: [
                Permission.CREATE_AGENT,
                Permission.DELETE_AGENT,
                Permission.MANAGE_AGENTS,
                Permission.VIEW_AGENTS,
                Permission.USE_LLM,
                Permission.READ_MEMORY,
                Permission.WRITE_MEMORY,
                Permission.API_ACCESS
            ],
            UserRole.DEVELOPER: [
                Permission.VIEW_AGENTS,
                Permission.USE_LLM,
                Permission.READ_MEMORY,
                Permission.WRITE_MEMORY,
                Permission.API_ACCESS,
                Permission.WEBHOOK_ACCESS
            ],
            UserRole.USER: [
                Permission.VIEW_AGENTS,
                Permission.USE_LLM,
                Permission.READ_MEMORY,
                Permission.API_ACCESS
            ],
            UserRole.AGENT: [
                Permission.USE_LLM,
                Permission.READ_MEMORY,
                Permission.WRITE_MEMORY,
                Permission.API_ACCESS
            ]
        }
        
        # JWT settings
        self.jwt_secret = settings.JWT_SECRET_KEY
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = 24
        
    async def initialize(self) -> None:
        """Initialize authentication system"""
        try:
            # Get Redis client
            self.redis_client = await get_redis()
            
            # Create default admin user if none exists
            await self._create_default_admin()
            
            logger.info("Authentication system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize authentication system: {e}")
            raise
    
    async def _create_default_admin(self) -> None:
        """Create default admin user"""
        try:
            # Check if any admin user exists
            admin_exists = any(user.role == UserRole.ADMIN for user in self.users.values())
            
            if not admin_exists:
                # Create default admin
                admin_user = User(
                    username="admin",
                    email="admin@ai-service.local",
                    password_hash=self._hash_password("admin123"),  # Change in production!
                    role=UserRole.ADMIN,
                    permissions=self.role_permissions[UserRole.ADMIN]
                )
                
                self.users[admin_user.id] = admin_user
                logger.warning("Created default admin user (username: admin, password: admin123)")
                
        except Exception as e:
            logger.error(f"Failed to create default admin: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _generate_jwt_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Generate JWT token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=self.jwt_expiry_hours)
        
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _verify_jwt_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user_id"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload.get("user_id")
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    async def create_user(self, username: str, email: str, password: str, role: UserRole) -> str:
        """Create a new user"""
        try:
            # Check if username or email already exists
            for user in self.users.values():
                if user.username == username:
                    raise ValueError("Username already exists")
                if user.email == email:
                    raise ValueError("Email already exists")
            
            # Create user
            user = User(
                username=username,
                email=email,
                password_hash=self._hash_password(password),
                role=role,
                permissions=self.role_permissions.get(role, [])
            )
            
            self.users[user.id] = user
            
            # Store in Redis
            await self._store_user_in_redis(user)
            
            logger.info(f"Created user: {username} ({role})")
            return user.id
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise
    
    async def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user with username/password"""
        try:
            # Find user by username
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user or not user.is_active:
                return None
            
            # Verify password
            if not self._verify_password(password, user.password_hash):
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            await self._store_user_in_redis(user)
            
            # Generate JWT token
            token = self._generate_jwt_token(user.id)
            
            # Create session
            session = Session(
                user_id=user.id,
                token=token,
                expires_at=datetime.utcnow() + timedelta(hours=self.jwt_expiry_hours)
            )
            
            self.sessions[session.id] = session
            await self._store_session_in_redis(session)
            
            logger.info(f"User authenticated: {username}")
            return token
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    async def authenticate_token(self, token: str) -> Optional[User]:
        """Authenticate user with JWT token"""
        try:
            # Verify JWT token
            user_id = self._verify_jwt_token(token)
            if not user_id:
                return None
            
            # Get user
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return None
            
            # Update session activity
            for session in self.sessions.values():
                if session.token == token and session.user_id == user_id:
                    session.last_activity = datetime.utcnow()
                    await self._store_session_in_redis(session)
                    break
            
            return user
            
        except Exception as e:
            logger.error(f"Token authentication failed: {e}")
            return None
    
    async def create_api_key(self, user_id: str, name: str, permissions: List[Permission], expires_at: Optional[datetime] = None) -> str:
        """Create API key for user"""
        try:
            # Verify user exists
            user = self.users.get(user_id)
            if not user:
                raise ValueError("User not found")
            
            # Generate API key
            api_key = secrets.token_urlsafe(32)
            key_hash = self._hash_password(api_key)
            
            # Create API key record
            api_key_record = APIKey(
                name=name,
                key_hash=key_hash,
                user_id=user_id,
                permissions=permissions,
                expires_at=expires_at
            )
            
            self.api_keys[api_key_record.id] = api_key_record
            
            # Store in Redis
            await self._store_api_key_in_redis(api_key_record)
            
            logger.info(f"Created API key: {name} for user {user.username}")
            return api_key  # Return the actual key (only time it's visible)
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise
    
    async def authenticate_api_key(self, api_key: str) -> Optional[Tuple[User, APIKey]]:
        """Authenticate API key"""
        try:
            # Find API key by hash
            api_key_record = None
            for key_record in self.api_keys.values():
                if self._verify_password(api_key, key_record.key_hash):
                    api_key_record = key_record
                    break
            
            if not api_key_record or not api_key_record.is_active:
                return None
            
            # Check expiration
            if api_key_record.expires_at and datetime.utcnow() > api_key_record.expires_at:
                return None
            
            # Get user
            user = self.users.get(api_key_record.user_id)
            if not user or not user.is_active:
                return None
            
            # Update usage statistics
            api_key_record.last_used = datetime.utcnow()
            api_key_record.usage_count += 1
            await self._store_api_key_in_redis(api_key_record)
            
            return user, api_key_record
            
        except Exception as e:
            logger.error(f"API key authentication failed: {e}")
            return None
    
    def check_permission(self, user: User, permission: Permission, api_key: Optional[APIKey] = None) -> bool:
        """Check if user has permission"""
        try:
            # Check user permissions
            if permission in user.permissions:
                return True
            
            # Check API key permissions if provided
            if api_key and permission in api_key.permissions:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    async def logout(self, token: str) -> bool:
        """Logout user (invalidate session)"""
        try:
            # Find and remove session
            session_to_remove = None
            for session_id, session in self.sessions.items():
                if session.token == token:
                    session_to_remove = session_id
                    break
            
            if session_to_remove:
                del self.sessions[session_to_remove]
                
                # Remove from Redis
                if self.redis_client:
                    await self.redis_client.delete(f"session:{session_to_remove}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    async def revoke_api_key(self, api_key_id: str) -> bool:
        """Revoke API key"""
        try:
            if api_key_id in self.api_keys:
                self.api_keys[api_key_id].is_active = False
                await self._store_api_key_in_redis(self.api_keys[api_key_id])
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False
    
    async def _store_user_in_redis(self, user: User) -> None:
        """Store user in Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"user:{user.id}"
            value = user.json()
            await self.redis_client.set(key, value, ex=86400)  # 24 hours
        except Exception as e:
            logger.error(f"Failed to store user in Redis: {e}")
    
    async def _store_session_in_redis(self, session: Session) -> None:
        """Store session in Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"session:{session.id}"
            value = session.json()
            ttl = int((session.expires_at - datetime.utcnow()).total_seconds())
            await self.redis_client.set(key, value, ex=max(ttl, 1))
        except Exception as e:
            logger.error(f"Failed to store session in Redis: {e}")
    
    async def _store_api_key_in_redis(self, api_key: APIKey) -> None:
        """Store API key in Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"api_key:{api_key.id}"
            value = api_key.json()
            await self.redis_client.set(key, value, ex=86400 * 30)  # 30 days
        except Exception as e:
            logger.error(f"Failed to store API key in Redis: {e}")
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def list_users(self) -> List[User]:
        """List all users"""
        return list(self.users.values())
    
    def list_api_keys(self, user_id: str) -> List[APIKey]:
        """List API keys for user"""
        return [key for key in self.api_keys.values() if key.user_id == user_id]
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        try:
            now = datetime.utcnow()
            expired_sessions = [
                session_id for session_id, session in self.sessions.items()
                if session.expires_at < now
            ]
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
                
                # Remove from Redis
                if self.redis_client:
                    await self.redis_client.delete(f"session:{session_id}")
            
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get authentication system statistics"""
        active_sessions = len([s for s in self.sessions.values() if s.expires_at > datetime.utcnow()])
        active_api_keys = len([k for k in self.api_keys.values() if k.is_active])
        
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "total_api_keys": len(self.api_keys),
            "active_api_keys": active_api_keys,
            "users_by_role": {
                role.value: len([u for u in self.users.values() if u.role == role])
                for role in UserRole
            }
        }


# Global authentication system instance
auth_system = AuthenticationSystem()
