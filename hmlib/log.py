"""Rich-based logging helpers for hmlib.

This module centralizes logging configuration so that:

- CLIs get colored, rich-formatted log output by default.
- Library code can call :func:`get_root_logger` or import :data:`logger`
  without worrying about handlers.
- When the rich progress UI is active, logging can be redirected into the
  scrolling log pane without duplicating output.
"""

from __future__ import annotations

import logging  # re-exported for callers that import from hmlib.log
from typing import Optional, Union

try:
    from rich.console import Console
    from rich.logging import RichHandler
except Exception:  # pragma: no cover - optional rich dependency
    Console = None
    RichHandler = None

_LOGGER_NAME = "hmlib"
_configured_logger: logging.Logger | None = None


def _normalize_level(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    try:
        return logging._nameToLevel.get(str(level).upper(), logging.INFO)
    except Exception:
        return logging.INFO


def _configure_logger(level: Union[int, str] = logging.INFO) -> logging.Logger:
    global _configured_logger

    lvl = _normalize_level(level)
    if _configured_logger is not None:
        _configured_logger.setLevel(lvl)
        return _configured_logger

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(lvl)

    # Only attach handlers if none are present so that host applications
    # can override logging configuration when embedding hmlib.
    if not logger.handlers:
        if RichHandler is not None and Console is not None:
            console = Console(stderr=True, force_terminal=True)
            handler = RichHandler(
                console=console,
                rich_tracebacks=True,
                markup=True,
                show_path=False,
            )
            handler.setLevel(lvl)
            logger.addHandler(handler)
        else:
            handler = logging.StreamHandler()
            handler.setLevel(lvl)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            logger.addHandler(handler)

    _configured_logger = logger
    return logger


# Default shared logger used across hmlib.
logger = _configure_logger()


def set_level(level: Union[int, str]) -> None:
    """Set the global logging level for the shared hmlib logger."""
    _configure_logger(level)


def get_root_logger(level: Union[int, str, None] = None) -> logging.Logger:
    """Return the shared root :class:`logging.Logger` used across hmlib."""
    if level is not None:
        return _configure_logger(level)
    return _configure_logger(logger.level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger that propagates to the shared hmlib logger."""
    base = _configure_logger()
    if not name or name == base.name:
        return base
    child = logging.getLogger(name)
    return child
