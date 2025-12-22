"""
Retry logic with exponential backoff for handling transient failures.
Production-grade implementation with jitter and observability.
"""

import asyncio
import time
import functools
from typing import Callable, Type, Tuple, TypeVar, Any, Optional
import logging
import random

from src.utils.logger import get_logger

# Prometheus metrics for observability
# In production, you MUST track how many retries are happening
try:
    from prometheus_client import Counter
    RETRY_COUNTER = Counter('ml_service_retries_total', 'Total retries', ['func_name', 'exception'])
    FALLBACK_COUNTER = Counter('ml_service_fallbacks_total', 'Total defaults returned', ['func_name'])
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    RETRY_COUNTER = None
    FALLBACK_COUNTER = None

logger = get_logger(__name__)

T = TypeVar('T')


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_instance: logging.Logger = None
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        exponential_base: Multiplier for delay on each retry
        exceptions: Tuple of exception types to catch and retry
        logger_instance: Optional logger for retry messages

    Returns:
        Decorated function with retry logic
    """
    log = logger_instance or logger

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        log.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )
                        raise

                    log.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    # delay *= exponential_base                    
                    delay = (initial_delay * (exponential_base ** attempt)) + random.uniform(0, 1)

            # This should never be reached, but satisfies type checker
            raise RuntimeError(f"{func.__name__} exhausted all retries")

        return wrapper

    return decorator


def async_retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_instance: logging.Logger = None
):
    """
    Decorator to retry an async function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        exponential_base: Multiplier for delay on each retry
        exceptions: Tuple of exception types to catch and retry
        logger_instance: Optional logger for retry messages

    Returns:
        Decorated async function with retry logic
    """
    log = logger_instance or logger

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            delay = initial_delay

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        log.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )
                        raise

                    log.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= exponential_base

            # This should never be reached, but satisfies type checker
            raise RuntimeError(f"{func.__name__} exhausted all retries")

        return wrapper

    return decorator


def async_retry_hardened(
    max_retries: int = 3,
    initial_delay: float = 0.5,
    exponential_base: float = 2.0,
    jitter_factor: float = 0.5,
    # NEVER default this to Exception. Force the engineer to choose.
    retryable_exceptions: Tuple[Type[Exception], ...] = (RuntimeError,),
    name: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None
):
    """
    Production-grade retry with jitter and observability.

    This decorator adds:
    - Exponential backoff with jitter to prevent thundering herd
    - Prometheus metrics tracking (if available)
    - Explicit exception type requirements
    - Enhanced logging with operation names

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        exponential_base: Multiplier for delay on each retry
        jitter_factor: Random jitter factor (0.0-1.0) to add to delay
        retryable_exceptions: Tuple of exception types to catch and retry
        name: Operation name for logging and metrics
        logger_instance: Optional logger for retry messages

    Returns:
        Decorated async function with production-grade retry logic
    """
    log = logger_instance or logger

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    # Track retry metrics if Prometheus is available
                    if METRICS_ENABLED:
                        RETRY_COUNTER.labels(func_name=op_name, exception=type(e).__name__).inc()

                    if attempt == max_retries - 1:
                        log.error(f"FATAL: {op_name} exhausted {max_retries} retries. Last error: {e}")
                        raise

                    # THE FIX: Add Jitter.
                    # This spreads the load so 1000 pods don't hit the DB at once.
                    sleep_time = delay + (random.random() * jitter_factor * delay)

                    log.warning(
                        f"RETRY: {op_name} attempt {attempt + 1} failed. "
                        f"Retrying in {sleep_time:.2f}s... Error: {e}"
                    )

                    await asyncio.sleep(sleep_time)
                    delay *= exponential_base

            raise last_exception  # Should be unreachable but kept for safety

        return wrapper
    return decorator


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Any = None,
    logger_instance: logging.Logger = None,
    **kwargs
) -> T:
    """
    Execute a function safely, catching exceptions and returning a default value.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        default: Default value to return on exception
        logger_instance: Optional logger for error messages
        **kwargs: Keyword arguments for func

    Returns:
        Function result or default value on exception
    """
    log = logger_instance or logger

    try:
        return func(*args, **kwargs)
    except Exception as e:
        log.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return default


async def safe_execute_observed(
    func: Callable[..., T],
    *args,
    default: Any = None,
    name: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
    **kwargs
) -> T:
    """
    Executes and returns default on failure, but ALERTS the system that it happened.

    This function wraps async operations with:
    - Exception handling with fallback to default value
    - Prometheus metrics tracking when fallback is used
    - Enhanced logging for observability

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        default: Default value to return on exception
        name: Operation name for logging and metrics
        logger_instance: Optional logger for error messages
        **kwargs: Keyword arguments for func

    Returns:
        Function result or default value on exception
    """
    log = logger_instance or logger
    op_name = name or getattr(func, '__name__', 'unnamed_op')

    try:
        return await func(*args, **kwargs)
    except Exception as e:
        # If this fires, we have 'silent' degradation that is now visible in Grafana
        if METRICS_ENABLED:
            FALLBACK_COUNTER.labels(func_name=op_name).inc()

        log.error(f"FALLBACK: {op_name} failed. Returning default. Error: {e}", exc_info=True)
        return default
