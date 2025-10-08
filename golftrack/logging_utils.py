from __future__ import annotations

import inspect
import logging
import time
from functools import wraps
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union, overload
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .config import Config

DEBUG_ENABLED = False
LOGGER_BASE_NAME = "golftrack"

F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Awaitable[Any]])


def configure_logging(config: "Config") -> None:
    """Configure logging for the application based on config.debug."""

    global DEBUG_ENABLED

    logger = logging.getLogger(LOGGER_BASE_NAME)
    env_level = os.getenv("GOLFTRACK_LOG_LEVEL") or os.getenv("LOG_LEVEL")
    level_name = (env_level or config.debug.log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    if not logger.handlers:
        if config.debug.log_file:
            handler: logging.Handler = logging.FileHandler(config.debug.log_file, encoding="utf-8")
        else:
            handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False

    DEBUG_ENABLED = config.debug.enabled or logger.isEnabledFor(logging.DEBUG)


def is_debug_enabled() -> bool:
    return DEBUG_ENABLED


def get_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_BASE_NAME}.{component}")


def _summarize_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool, type(None))):
        return repr(value)
    if hasattr(value, "shape"):
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        return f"{type(value).__name__}(shape={shape}, dtype={dtype})"
    if hasattr(value, "__len__") and not isinstance(value, dict):
        length = len(value)  # type: ignore[arg-type]
        return f"{type(value).__name__}(len={length})"
    if isinstance(value, dict):
        keys = list(value.keys())
        preview = keys[:3]
        suffix = "..." if len(keys) > 3 else ""
        return f"dict(keys={preview}{suffix})"
    return type(value).__name__


def _format_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    preview = []
    for arg in args[:3]:
        preview.append(_summarize_value(arg))
    if len(args) > 3:
        preview.append("...")
    for key, value in list(kwargs.items())[:3]:
        preview.append(f"{key}={_summarize_value(value)}")
    return ", ".join(preview)


@overload
def debug_trace(func: F) -> F: ...


@overload
def debug_trace(*, name: Optional[str] = None) -> Callable[[F], F]: ...


def debug_trace(func: Optional[F] = None, *, name: Optional[str] = None):
    """
    Decorator that logs function entry, exit, and exceptions when debug is enabled.

    Supports both sync and async callables.
    """

    def decorator(target: Callable[..., Any]):
        qualname = name or target.__qualname__
        logger = logging.getLogger(f"{LOGGER_BASE_NAME}.trace")

        if inspect.iscoroutinefunction(target):

            @wraps(target)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not is_debug_enabled():
                    return await target(*args, **kwargs)

                arg_preview = _format_args(args, kwargs)
                logger.debug("Entering %s(%s)", qualname, arg_preview)
                start = time.perf_counter()
                try:
                    result = await target(*args, **kwargs)
                    duration = (time.perf_counter() - start) * 1000.0
                    logger.debug("Exiting %s -> %s (%.2f ms)", qualname, _summarize_value(result), duration)
                    return result
                except Exception:
                    logger.exception("Exception in %s", qualname)
                    raise

            return async_wrapper  # type: ignore[return-value]

        @wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_debug_enabled():
                return target(*args, **kwargs)

            arg_preview = _format_args(args, kwargs)
            logger.debug("Entering %s(%s)", qualname, arg_preview)
            start = time.perf_counter()
            try:
                result = target(*args, **kwargs)
                duration = (time.perf_counter() - start) * 1000.0
                logger.debug("Exiting %s -> %s (%.2f ms)", qualname, _summarize_value(result), duration)
                return result
            except Exception:
                logger.exception("Exception in %s", qualname)
                raise

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


__all__ = ["configure_logging", "debug_trace", "get_logger", "is_debug_enabled"]
