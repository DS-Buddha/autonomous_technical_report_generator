"""
Structured logging configuration for the hybrid agentic system.
Provides colored console output and optional file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

from src.config.settings import settings


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up a logger with optional colored console output and file logging.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    log_level = level or settings.log_level
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if HAS_COLORLOG and settings.enable_logging:
        # Colored formatter
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s[%(name)s]%(reset)s %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        # Standard formatter
        formatter = logging.Formatter(
            "%(levelname)-8s [%(name)s] %(message)s"
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default settings.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return setup_logger(name)
