"""
Security Configuration and Hardening for NitroAGI NEXUS
"""

import os
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    jwt_secret_key: str
    encryption_key: str
    password_salt: str
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    require_2fa: bool = False
    allowed_origins: List[str] = None
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600
    min_password_length: int = 12
    require_password_complexity: bool = True
    enable_audit_logging: bool = True
    secure_headers_enabled: bool = True
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:3000", "https://yourdomain.com"]


class SecurityManager:
    """Central security management system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.cipher_suite = Fernet(config.encryption_key.encode()[:32].ljust(32, b'0'))
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.active_sessions: Dict[str, datetime] = {}
        self.audit_log: List[Dict] = []
        
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def validate_password_strength(self, password: str) -> Dict[str, bool]:
        """Validate password meets security requirements"""
        checks = {
            'length': len(password) >= self.config.min_password_length,
            'uppercase': any(c.isupper() for c in password),
            'lowercase': any(c.islower() for c in password),
            'digit': any(c.isdigit() for c in password),
            'special': any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password),
            'no_common_patterns': not self._contains_common_patterns(password)
        }
        
        if not self.config.require_password_complexity:
            # Only check length if complexity not required
            return {'length': checks['length']}
        
        return checks
    
    def _contains_common_patterns(self, password: str) -> bool:
        """Check for common weak password patterns"""
        common_patterns = [
            'password', '123456', 'qwerty', 'admin', 'letmein',
            'welcome', 'monkey', 'dragon', 'master', 'shadow'
        ]
        
        password_lower = password.lower()
        return any(pattern in password_lower for pattern in common_patterns)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted = self.cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher_suite.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return ""
    
    def generate_jwt_token(
        self, 
        user_id: str, 
        permissions: List[str],
        expires_in: Optional[int] = None
    ) -> str:
        """Generate JWT token"""
        
        exp_time = expires_in or self.config.session_timeout
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=exp_time),
            'jti': self.generate_secure_token()  # Token ID for revocation
        }
        
        token = jwt.encode(payload, self.config.jwt_secret_key, algorithm='HS256')
        
        # Track active session
        self.active_sessions[payload['jti']] = datetime.utcnow()
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret_key, algorithms=['HS256'])
            
            # Check if session is still active
            jti = payload.get('jti')
            if jti not in self.active_sessions:
                return None
            
            # Check session timeout
            session_time = self.active_sessions[jti]
            if datetime.utcnow() - session_time > timedelta(seconds=self.config.session_timeout):
                self.revoke_session(jti)
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def revoke_session(self, jti: str):
        """Revoke a session by token ID"""
        if jti in self.active_sessions:
            del self.active_sessions[jti]
    
    def check_rate_limit(self, identifier: str, window_seconds: int = None) -> bool:
        """Check if identifier is within rate limits"""
        window = window_seconds or self.config.rate_limit_window
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=window)
        
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        # Clean old attempts
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > cutoff
        ]
        
        return len(self.failed_attempts[identifier]) < self.config.rate_limit_requests
    
    def record_failed_attempt(self, identifier: str):
        """Record a failed authentication attempt"""
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        self.failed_attempts[identifier].append(datetime.utcnow())
        
        self.audit_log.append({
            'event': 'failed_authentication',
            'identifier': identifier,
            'timestamp': datetime.utcnow().isoformat(),
            'ip_address': identifier  # Assuming identifier is IP
        })
    
    def is_account_locked(self, identifier: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if identifier not in self.failed_attempts:
            return False
        
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.config.lockout_duration)
        
        recent_failures = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > cutoff
        ]
        
        return len(recent_failures) >= self.config.max_login_attempts
    
    def validate_request_origin(self, origin: str) -> bool:
        """Validate request origin against allowed origins"""
        return origin in self.config.allowed_origins
    
    def sanitize_input(self, user_input: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
        sanitized = user_input
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        return sanitized[:1000]
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        message = f"{session_id}:{datetime.utcnow().isoformat()}"
        signature = hmac.new(
            self.config.jwt_secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{message}:{signature}"
    
    def verify_csrf_token(self, token: str, session_id: str) -> bool:
        """Verify CSRF token"""
        try:
            parts = token.split(':')
            if len(parts) != 3:
                return False
            
            received_session, timestamp, signature = parts
            
            if received_session != session_id:
                return False
            
            # Check if token is not too old (1 hour)
            token_time = datetime.fromisoformat(timestamp)
            if datetime.utcnow() - token_time > timedelta(hours=1):
                return False
            
            # Verify signature
            expected_message = f"{session_id}:{timestamp}"
            expected_signature = hmac.new(
                self.config.jwt_secret_key.encode(),
                expected_message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception:
            return False
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses"""
        if not self.config.secure_headers_enabled:
            return {}
        
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self' ws: wss:;"
            ),
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
    
    def audit_security_event(
        self, 
        event_type: str, 
        user_id: str = None, 
        ip_address: str = None,
        details: Dict = None
    ):
        """Log security event for audit"""
        if not self.config.enable_audit_logging:
            return
        
        audit_entry = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'ip_address': ip_address,
            'details': details or {},
            'severity': self._get_event_severity(event_type)
        }
        
        self.audit_log.append(audit_entry)
        
        # Log to file/database in production
        logger.info(f"Security audit: {audit_entry}")
    
    def _get_event_severity(self, event_type: str) -> str:
        """Determine severity level for security events"""
        high_severity = [
            'failed_authentication', 'account_locked', 'privilege_escalation',
            'data_access_violation', 'suspicious_activity'
        ]
        
        medium_severity = [
            'password_change', 'session_created', 'permission_denied',
            'rate_limit_exceeded'
        ]
        
        if event_type in high_severity:
            return 'HIGH'
        elif event_type in medium_severity:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_security_report(self) -> Dict:
        """Generate security status report"""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        
        recent_events = [
            event for event in self.audit_log
            if datetime.fromisoformat(event['timestamp']) > last_24h
        ]
        
        return {
            'active_sessions': len(self.active_sessions),
            'locked_accounts': sum(
                1 for attempts in self.failed_attempts.values()
                if len(attempts) >= self.config.max_login_attempts
            ),
            'recent_security_events': len(recent_events),
            'high_severity_events': sum(
                1 for event in recent_events
                if event.get('severity') == 'HIGH'
            ),
            'configuration': {
                'session_timeout': self.config.session_timeout,
                'max_login_attempts': self.config.max_login_attempts,
                'require_2fa': self.config.require_2fa,
                'secure_headers_enabled': self.config.secure_headers_enabled
            }
        }


class InputValidator:
    """Validates and sanitizes user inputs"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username format"""
        import re
        # Alphanumeric plus underscore and dash, 3-20 characters
        pattern = r'^[a-zA-Z0-9_-]{3,20}$'
        return re.match(pattern, username) is not None
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        import re
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        sanitized = sanitized.replace('..', '')
        return sanitized[:255]  # Limit length
    
    @staticmethod
    def validate_json_size(json_str: str, max_size_mb: int = 10) -> bool:
        """Validate JSON payload size"""
        size_bytes = len(json_str.encode('utf-8'))
        max_bytes = max_size_mb * 1024 * 1024
        return size_bytes <= max_bytes
    
    @staticmethod
    def sanitize_html(html: str) -> str:
        """Basic HTML sanitization"""
        import re
        # Remove script tags and their content
        html = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove on* event handlers
        html = re.sub(r'\s*on\w+\s*=\s*["\'][^"\']*["\']', '', html, flags=re.IGNORECASE)
        return html


def setup_security_config() -> SecurityConfig:
    """Setup security configuration from environment"""
    return SecurityConfig(
        jwt_secret_key=os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32)),
        encryption_key=os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode()),
        password_salt=os.getenv('PASSWORD_SALT', secrets.token_hex(16)),
        session_timeout=int(os.getenv('SESSION_TIMEOUT', '3600')),
        max_login_attempts=int(os.getenv('MAX_LOGIN_ATTEMPTS', '5')),
        lockout_duration=int(os.getenv('LOCKOUT_DURATION', '900')),
        require_2fa=os.getenv('REQUIRE_2FA', 'false').lower() == 'true',
        rate_limit_requests=int(os.getenv('RATE_LIMIT_REQUESTS', '1000')),
        rate_limit_window=int(os.getenv('RATE_LIMIT_WINDOW', '3600')),
        min_password_length=int(os.getenv('MIN_PASSWORD_LENGTH', '12')),
        require_password_complexity=os.getenv('REQUIRE_PASSWORD_COMPLEXITY', 'true').lower() == 'true',
        enable_audit_logging=os.getenv('ENABLE_AUDIT_LOGGING', 'true').lower() == 'true',
        secure_headers_enabled=os.getenv('SECURE_HEADERS_ENABLED', 'true').lower() == 'true'
    )


# Global security manager instance
_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    global _security_manager
    if _security_manager is None:
        config = setup_security_config()
        _security_manager = SecurityManager(config)
    return _security_manager