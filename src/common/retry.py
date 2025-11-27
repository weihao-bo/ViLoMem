"""Retry utilities for handling transient failures in model and API calls.

This module provides decorators and utilities for adding retry logic with exponential backoff
to handle network errors, rate limits, and other transient failures.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

# Type variable for generic function signatures
F = TypeVar("F", bound=Callable[..., Any])


class RetryableError(Exception):
    """Base exception for errors that should trigger a retry."""

    pass


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error should trigger a retry.

    Args:
        error: Exception that was raised

    Returns:
        True if the error is retryable, False otherwise
    """
    # Network-related errors from httpx
    if isinstance(  # noqa: UP038
        error,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.NetworkError,
        ),
    ):
        return True

    # HTTP status codes that should trigger retry
    if isinstance(error, httpx.HTTPStatusError):
        # Retry on 5xx server errors and 429 (rate limit)
        return error.response.status_code >= 500 or error.response.status_code == 429

    # LangChain/model-specific errors
    error_message = str(error).lower()
    retryable_patterns = [
        "timeout",
        "connection",
        "rate limit",
        "too many requests",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "network",
        "temporary failure",
    ]

    return any(pattern in error_message for pattern in retryable_patterns)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> Callable[[F], F]:
    """Retry a function with exponential backoff on retryable errors.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        jitter: Whether to add random jitter to delays (default: True)

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_retries=3)
        def api_call():
            # Make API call that might fail
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import random

            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if error is retryable
                    if not is_retryable_error(e):
                        logger.warning(
                            f"Non-retryable error in {func.__name__}: {type(e).__name__}: {e}"
                        )
                        raise

                    # Don't retry after last attempt
                    if attempt >= max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}. "
                            f"Last error: {type(e).__name__}: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (exponential_base**attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        f"Retryable error in {func.__name__} (attempt {attempt + 1}/{max_retries}): "
                        f"{type(e).__name__}: {e}. Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected retry logic failure in {func.__name__}")

        return wrapper  # type: ignore

    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> Callable[[F], F]:
    """Retry an async function with exponential backoff on retryable errors.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        jitter: Whether to add random jitter to delays (default: True)

    Returns:
        Decorated async function with retry logic

    Example:
        @async_retry_with_backoff(max_retries=3)
        async def api_call():
            # Make async API call that might fail
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            import random

            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if error is retryable
                    if not is_retryable_error(e):
                        logger.warning(
                            f"Non-retryable error in {func.__name__}: {type(e).__name__}: {e}"
                        )
                        raise

                    # Don't retry after last attempt
                    if attempt >= max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}. "
                            f"Last error: {type(e).__name__}: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (exponential_base**attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        f"Retryable error in {func.__name__} (attempt {attempt + 1}/{max_retries}): "
                        f"{type(e).__name__}: {e}. Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected retry logic failure in {func.__name__}")

        return wrapper  # type: ignore

    return decorator
