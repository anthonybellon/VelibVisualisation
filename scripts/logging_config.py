"""
Logging configuration for VelibVisualisation.

Provides structured logging with configurable levels and formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL


def setup_logging(
    name: Optional[str] = None,
    level: str = LOG_LEVEL,
    log_file: Optional[Path] = None,
    format_str: str = LOG_FORMAT,
    date_format: str = LOG_DATE_FORMAT,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        name: Logger name. If None, configures the root logger.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        format_str: Log message format string.
        date_format: Date format string.

    Returns:
        Configured logger instance.
    """
    # Get or create logger
    logger = logging.getLogger(name)

    # Clear existing handlers
    logger.handlers = []

    # Set level
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Create formatter
    formatter = logging.Formatter(format_str, datefmt=date_format)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# Convenience function to set up basic logging for CLI scripts
def init_cli_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """
    Initialize logging for CLI scripts with verbose/quiet options.

    Args:
        verbose: If True, set log level to DEBUG.
        quiet: If True, set log level to WARNING.

    Returns:
        Configured root logger.
    """
    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "WARNING"
    else:
        level = "INFO"

    return setup_logging(level=level)
