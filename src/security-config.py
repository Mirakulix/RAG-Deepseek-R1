from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer
import secrets
from dataclasses import dataclass
import logging
from enum import Enum

# Security Konfiguration
class SecurityConfig:
    SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # Password Hashing
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # OAuth2
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    
    # Rate Limiting
    RATE_LIMIT_MINUTE = 100
    RATE_LIMIT_HOUR = 1000

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"

@dataclass
class UserData:
    username: str
    role: Role
    permissions: list[str]
    rate_limit: Optional[int] = None

class SecurityManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._token_blacklist = set()
        self._rate_limiters = {}

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Erstellt einen JWT Access Token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=SecurityConfig.ACCESS_TOKEN_EXPIRE_MINUTES
            )
            
        to_encode.update({"exp": expire})
        token = jwt.encode(
            to_encode,
            SecurityConfig.SECRET_KEY,
            algorithm=SecurityConfig.ALGORITHM
        )
        
        return token

    async def verify_token(self, token: str) -> UserData:
        """Verifiziert einen JWT Token."""
        try:
            if token in self._token_blacklist:
                raise HTTPException(
                    status_code=401,
                    detail="Token has been revoked"
                )
                
            payload = jwt.decode(
                token,
                SecurityConfig.SECRET_KEY,
                algorithms=[SecurityConfig.ALGORITHM]
            )
            
            username = payload.get("sub")
            role = payload.get("role")
            permissions = payload.get("permissions", [])
            
            if username is None:
                raise HTTPException(
                    status_code=401,
                    detail="Could not validate credentials"
                )
                
            return UserData(
                username=username,
                role=Role(role),
                permissions=permissions
            )
            
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )

    async def check_permission(
        self,
        token: str = Depends(SecurityConfig.oauth2_scheme),
        required_permissions: list[str] = []
    ) -> bool:
        """Überprüft Benutzerberechtigungen."""
        user_data = await self.verify_token(token)
        
        if user_data.role == Role.ADMIN:
            return True
            
        return all(perm in user_data.permissions for perm in required_permissions)

    async def rate_limit_check(self, user_data: UserData) -> bool:
        """Überprüft Rate Limiting für einen Benutzer."""
        if user_data.username not in self._rate_limiters:
            self._rate_limiters[user_data.username] = {
                "minute": {"count": 0, "reset": datetime.utcnow()},
                "hour": {"count": 0, "reset": datetime.utcnow()}
            }
            
        limiter = self._rate_limiters[user_data.username]
        now = datetime.utcnow()
        
        # Minute Reset
        if (now - limiter["minute"]["reset"]).total_seconds() > 60:
            limiter["minute"] = {"count": 0, "reset": now}
            
        # Hour Reset
        if (now - limiter["hour"]["reset"]).total_seconds() > 3600:
            limiter["hour"] = {"count": 0, "reset": now}
            
        # Check Limits
        if (limiter["minute"]["count"] >= SecurityConfig.RATE_LIMIT_MINUTE or
            limiter["hour"]["count"] >= SecurityConfig.RATE_LIMIT_HOUR):
            return False
            
        # Update Counters
        limiter["minute"]["count"] += 1
        limiter["hour"]["count"] += 1
        
        return True

    def revoke_token(self, token: str):
        """Widerruft einen Token."""
        self._token_blacklist.add(token)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verifiziert ein Passwort."""
        return SecurityConfig.pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Erstellt einen Passwort-Hash."""
        return SecurityConfig.pwd_context.hash(password)

# Middleware für Security
class SecurityMiddleware:
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.logger = logging.getLogger(__name__)

    async def __call__(self, request, call_next):
        try:
            # Token Extraktion
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            
            if token:
                # Token Verifizierung
                user_data = await self.security_manager.verify_token(token)
                
                # Rate Limiting
                if not await self.security_manager.rate_limit_check(user_data):
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded"
                    )
                
                # Request Ausführung
                response = await call_next(request)
                
                # Response Headers
                response.headers["X-Rate-Limit-Remaining"] = str(
                    SecurityConfig.RATE_LIMIT_MINUTE - 
                    self.security_manager._rate_limiters[user_data.username]["minute"]["count"]
                )
                
                return response
                
            return await call_next(request)
            
        except Exception as e:
            self.logger.error(f"Security middleware error: {str(e)}")
            raise

# Security Dependencies
security_manager = SecurityManager()

async def get_current_user(
    token: str = Depends(SecurityConfig.oauth2_scheme)
) -> UserData:
    return await security_manager.verify_token(token)

def require_permissions(permissions: list[str]):
    async def permission_checker(
        user: UserData = Depends(get_current_user)
    ) -> bool:
        if not await security_manager.check_permission(
            user.permissions,
            permissions
        ):
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions"
            )
        return True
    return permission_checker