"""Logging helpers that wrap mmengine's :class:`MMLogger` for hmlib."""

import logging  # noqa: F401
from typing import Union

from mmengine.logging import MMLogger

logger = MMLogger.get_current_instance()

# def get_logger(name="root"):
#     formatter = logging.Formatter(
#         # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
#         fmt="%(asctime)s [%(levelname)s]: %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )

#     handler = logging.StreamHandler(sys.stderr)
#     # handler = logging.StreamHandler(sys.stdout)
#     handler.setFormatter(formatter)

#     logger = logging.getLogger(name)

#     logger.setLevel(logging.DEBUG)
#     # logger.setLevel(logging.INFO)


#     logger.addHandler(handler)
#     return logger
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
