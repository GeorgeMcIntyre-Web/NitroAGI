"""Test utilities and helpers."""

import asyncio
import time
from typing import Any, Callable, Dict, Optional
from contextlib import asynccontextmanager
import json


async def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1
) -> bool:
    """Wait for a condition to become true.
    
    Args:
        condition: Function that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds
        
    Returns:
        True if condition was met, False if timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        if condition():
            return True
        await asyncio.sleep(interval)
    return False


async def async_retry(
    func: Callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Any:
    """Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
        
    Returns:
        Function result
        
    Raises:
        Last exception if all attempts fail
    """
    current_delay = delay
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
                current_delay *= backoff
    
    raise last_exception


@asynccontextmanager
async def measure_time():
    """Context manager to measure execution time.
    
    Yields:
        Dict with elapsed time in seconds
    """
    result = {"elapsed": 0.0}
    start = time.time()
    try:
        yield result
    finally:
        result["elapsed"] = time.time() - start


def assert_json_equal(actual: Any, expected: Any, ignore_keys: Optional[list] = None):
    """Assert JSON objects are equal, optionally ignoring certain keys.
    
    Args:
        actual: Actual JSON object
        expected: Expected JSON object
        ignore_keys: Keys to ignore in comparison
    """
    ignore_keys = ignore_keys or []
    
    def remove_keys(obj, keys):
        if isinstance(obj, dict):
            return {k: remove_keys(v, keys) for k, v in obj.items() if k not in keys}
        elif isinstance(obj, list):
            return [remove_keys(item, keys) for item in obj]
        return obj
    
    actual_cleaned = remove_keys(actual, ignore_keys)
    expected_cleaned = remove_keys(expected, ignore_keys)
    
    assert json.dumps(actual_cleaned, sort_keys=True) == json.dumps(expected_cleaned, sort_keys=True)


def create_mock_response(data: Any, status_code: int = 200) -> Dict[str, Any]:
    """Create mock HTTP response.
    
    Args:
        data: Response data
        status_code: HTTP status code
        
    Returns:
        Mock response dictionary
    """
    return {
        "status_code": status_code,
        "json": lambda: data,
        "text": json.dumps(data) if not isinstance(data, str) else data,
        "headers": {"content-type": "application/json"}
    }


class AsyncIteratorMock:
    """Mock for async iterators."""
    
    def __init__(self, items):
        self.items = items
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


def mock_async_generator(items):
    """Create mock async generator.
    
    Args:
        items: Items to yield
        
    Returns:
        Async generator
    """
    async def generator():
        for item in items:
            yield item
    return generator()


class TimeMocker:
    """Helper for mocking time in tests."""
    
    def __init__(self, start_time: float = None):
        self.current_time = start_time or time.time()
    
    def advance(self, seconds: float):
        """Advance mocked time by seconds."""
        self.current_time += seconds
        return self.current_time
    
    def __call__(self):
        """Return current mocked time."""
        return self.current_time