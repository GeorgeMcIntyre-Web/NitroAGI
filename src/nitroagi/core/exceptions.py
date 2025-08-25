"""Custom exceptions for NitroAGI."""

from typing import Any, Dict, Optional


class NitroAGIException(Exception):
    """Base exception for all NitroAGI errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            error_code: Optional error code for categorization
            details: Optional additional details about the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "NITROAGI_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class ModuleException(NitroAGIException):
    """Exception raised by AI modules."""
    
    def __init__(
        self,
        module_name: str,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize module exception.
        
        Args:
            module_name: Name of the module that raised the exception
            message: Error message
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message, error_code or "MODULE_ERROR", details)
        self.module_name = module_name
        self.details["module"] = module_name


class OrchestratorException(NitroAGIException):
    """Exception raised by the orchestrator."""
    
    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize orchestrator exception.
        
        Args:
            message: Error message
            task_id: Optional task ID related to the error
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message, error_code or "ORCHESTRATOR_ERROR", details)
        if task_id:
            self.details["task_id"] = task_id


class MessageBusException(NitroAGIException):
    """Exception raised by the message bus."""
    
    def __init__(
        self,
        message: str,
        topic: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize message bus exception.
        
        Args:
            message: Error message
            topic: Optional topic related to the error
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message, error_code or "MESSAGE_BUS_ERROR", details)
        if topic:
            self.details["topic"] = topic


class MemoryException(NitroAGIException):
    """Exception raised by the memory system."""
    
    def __init__(
        self,
        message: str,
        memory_type: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize memory exception.
        
        Args:
            message: Error message
            memory_type: Optional type of memory (working, episodic, semantic)
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message, error_code or "MEMORY_ERROR", details)
        if memory_type:
            self.details["memory_type"] = memory_type


class ConfigurationException(NitroAGIException):
    """Exception raised for configuration errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize configuration exception.
        
        Args:
            message: Error message
            config_key: Optional configuration key that caused the error
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message, error_code or "CONFIG_ERROR", details)
        if config_key:
            self.details["config_key"] = config_key


class ValidationException(NitroAGIException):
    """Exception raised for validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize validation exception.
        
        Args:
            message: Error message
            field: Optional field that failed validation
            value: Optional value that failed validation
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message, error_code or "VALIDATION_ERROR", details)
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = str(value)


class TimeoutException(NitroAGIException):
    """Exception raised when an operation times out."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize timeout exception.
        
        Args:
            message: Error message
            operation: Optional operation that timed out
            timeout_seconds: Optional timeout duration
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message, error_code or "TIMEOUT_ERROR", details)
        if operation:
            self.details["operation"] = operation
        if timeout_seconds is not None:
            self.details["timeout_seconds"] = timeout_seconds


class ResourceException(NitroAGIException):
    """Exception raised for resource-related errors."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize resource exception.
        
        Args:
            message: Error message
            resource_type: Optional type of resource
            resource_id: Optional resource identifier
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message, error_code or "RESOURCE_ERROR", details)
        if resource_type:
            self.details["resource_type"] = resource_type
        if resource_id:
            self.details["resource_id"] = resource_id


class AuthenticationException(NitroAGIException):
    """Exception raised for authentication errors."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize authentication exception.
        
        Args:
            message: Error message
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message, error_code or "AUTH_ERROR", details)


class AuthorizationException(NitroAGIException):
    """Exception raised for authorization errors."""
    
    def __init__(
        self,
        message: str = "Authorization failed",
        required_permission: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize authorization exception.
        
        Args:
            message: Error message
            required_permission: Optional permission that was required
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message, error_code or "AUTHZ_ERROR", details)
        if required_permission:
            self.details["required_permission"] = required_permission