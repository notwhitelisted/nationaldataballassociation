"""Centralized logging configuration using loguru.

Usage:
    from app.utils.logger import logger
    logger.info("Something happened")
    logger.debug("Debug details: {}", some_var)
"""

import sys

from loguru import logger as _logger

from app.config import settings

# Remove default handler and add our own
_logger.remove()
_logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# Export as `logger`
logger = _logger
