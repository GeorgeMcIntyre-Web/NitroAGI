"""Logging utilities for NitroAGI with structured logging support."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import traceback
from functools import wraps
import asyncio
import time

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
import structlog
from pythonjsonlogger import jsonlogger

from nitroagi.utils.config import get_config


# Install rich traceback for better error display
install_rich_traceback(show_locals=True, suppress=[structlog])

# Rich console for pretty printing
console = Console()


class PerformanceFilter(logging.Filter):
    """Add performance metrics to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance data to the record."""
        # Add timestamp
        record.timestamp = datetime.utcnow().isoformat()
        
        # Add memory usage
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = process.memory_info().rss / 1024 / 1024
            record.cpu_percent = process.cpu_percent()
        except:
            record.memory_mb = 0
            record.cpu_percent = 0
        
        return True


class ContextFilter(logging.Filter):
    """Add context information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """Initialize with optional context."""
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to the record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class NitroAGIJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for NitroAGI logs."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to the log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record["timestamp"] = datetime.utcnow().isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno
        
        # Add performance metrics if available
        if hasattr(record, "memory_mb"):
            log_record["memory_mb"] = record.memory_mb
        if hasattr(record, "cpu_percent"):
            log_record["cpu_percent"] = record.cpu_percent
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            log_record["traceback"] = traceback.format_exception(*record.exc_info)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_logs: bool = False,
    rich_console: bool = True
) -> None:
    """Setup logging configuration for NitroAGI.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        json_logs: If True, use JSON formatting
        rich_console: If True, use rich console handler
    """
    config = get_config()
    
    # Determine log level
    level_str = log_level or config.log_level
    level = getattr(logging, level_str.upper(), logging.INFO)
    
    # Create log directory if needed
    if log_file or config.log_file:
        log_path = Path(log_file or config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.dict_tracebacks,
            structlog.processors.EventRenamer("message"),
            structlog.dev.ConsoleRenderer() if not json_logs else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Add performance filter
    perf_filter = PerformanceFilter()
    
    # Console handler
    if rich_console and not json_logs:
        # Rich console handler for pretty output
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=config.debug,
        )
        console_handler.setLevel(level)
        console_handler.addFilter(perf_filter)
        root_logger.addHandler(console_handler)
    else:
        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if json_logs:
            # JSON formatter for structured logs
            json_formatter = NitroAGIJSONFormatter()
            console_handler.setFormatter(json_formatter)
        else:
            # Standard formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
        
        console_handler.addFilter(perf_filter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file or config.log_file:
        file_handler = logging.FileHandler(log_file or config.log_file)
        file_handler.setLevel(level)
        file_handler.addFilter(perf_filter)
        
        if json_logs:
            json_formatter = NitroAGIJSONFormatter()
            file_handler.setFormatter(json_formatter)
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
        
        root_logger.addHandler(file_handler)
    
    # Set levels for third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger("nitroagi")
    logger.info(
        "NitroAGI logging initialized",
        extra={
            "environment": config.environment,
            "log_level": level_str,
            "debug": config.debug,
            "json_logs": json_logs,
        }
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a logger instance.
    
    Args:
        name: Logger name (defaults to caller's module)
        
    Returns:
        Structured logger instance
    """
    if name is None:
        # Get caller's module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "nitroagi")
        else:
            name = "nitroagi"
    
    return structlog.get_logger(name)


def log_execution_time(func):
    """Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    logger = get_logger(func.__module__)
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            logger.debug(
                f"Function executed: {func.__name__}",
                execution_time_ms=execution_time,
                function=func.__name__,
            )
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                f"Function failed: {func.__name__}",
                execution_time_ms=execution_time,
                function=func.__name__,
                error=str(e),
                exc_info=True,
            )
            raise
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            logger.debug(
                f"Async function executed: {func.__name__}",
                execution_time_ms=execution_time,
                function=func.__name__,
            )
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                f"Async function failed: {func.__name__}",
                execution_time_ms=execution_time,
                function=func.__name__,
                error=str(e),
                exc_info=True,
            )
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def log_method_calls(cls):
    """Class decorator to log all method calls.
    
    Args:
        cls: Class to decorate
        
    Returns:
        Decorated class
    """
    logger = get_logger(f"{cls.__module__}.{cls.__name__}")
    
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith("_"):
            setattr(cls, attr_name, log_execution_time(attr))
    
    # Log class instantiation
    original_init = cls.__init__
    
    def logged_init(self, *args, **kwargs):
        logger.debug(f"Creating instance of {cls.__name__}")
        original_init(self, *args, **kwargs)
    
    cls.__init__ = logged_init
    
    return cls


class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, **kwargs):
        """Initialize with context values."""
        self.context = kwargs
        self.logger = None
    
    def __enter__(self):
        """Enter the context."""
        self.logger = structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        if self.logger:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log an error with context.
    
    Args:
        error: The exception to log
        context: Optional context dictionary
    """
    logger = get_logger()
    
    error_dict = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
    
    if context:
        error_dict.update(context)
    
    logger.error("Error occurred", **error_dict)


def create_audit_logger(name: str = "audit") -> logging.Logger:
    """Create an audit logger for security events.
    
    Args:
        name: Logger name
        
    Returns:
        Audit logger instance
    """
    audit_logger = logging.getLogger(f"nitroagi.{name}")
    audit_logger.setLevel(logging.INFO)
    
    # Create audit log file
    audit_file = Path("logs") / f"{name}.log"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler with JSON formatting
    file_handler = logging.FileHandler(audit_file)
    json_formatter = NitroAGIJSONFormatter()
    file_handler.setFormatter(json_formatter)
    
    audit_logger.addHandler(file_handler)
    
    return audit_logger


# Create default audit logger
audit_logger = create_audit_logger()


def audit_log(
    event: str,
    user: Optional[str] = None,
    resource: Optional[str] = None,
    action: Optional[str] = None,
    result: Optional[str] = None,
    **kwargs
) -> None:
    """Log an audit event.
    
    Args:
        event: Event name
        user: User identifier
        resource: Resource being accessed
        action: Action being performed
        result: Result of the action
        **kwargs: Additional context
    """
    audit_logger.info(
        event,
        extra={
            "event": event,
            "user": user,
            "resource": resource,
            "action": action,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
    )