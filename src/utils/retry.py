"""
Retry logic with exponential backoff for handling transient failures.
"""

import time
import functools
from typing import Callable, Type, Tuple, TypeVar, Any
import logging

from src.utils.logger import get_logger

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
                    delay *= exponential_base

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
    import asyncio

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
