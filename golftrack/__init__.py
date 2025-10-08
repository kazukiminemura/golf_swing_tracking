"""
Golf swing tracking application package.

This module exposes convenience entry points without importing heavy dependencies
until required (e.g., OpenVINO or FastAPI).
"""

from typing import Any


def create_app(*args: Any, **kwargs: Any):
    from .server import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
