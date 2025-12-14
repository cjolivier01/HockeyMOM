"""Logging helpers that wrap mmengine's :class:`MMLogger` for hmlib."""

import logging  # noqa: F401
from typing import Union

from mmengine.logging import MMLogger

logger = MMLogger.get_current_instance()

def set_level(level: Union[int, str]) -> None:
    """Set the global logging level for the shared hmlib logger.

    @param level: Logging level (e.g. ``logging.INFO`` or ``\"INFO\"``).
    """
    logger.setLevel(level=level)


def get_root_logger(level: Union[int, str, None] = None) -> MMLogger:
    """Return the shared :class:`MMLogger` instance used across hmlib.

    @param level: Optional logging level override.
    @return: The configured :class:`MMLogger` instance.
    """
    if level is not None:
        logger.setLevel(level=level)
    return logger
