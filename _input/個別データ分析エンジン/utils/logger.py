"""
Logging utilities for the AI Analysis Engine
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

# from ..config import config  # Avoid circular import


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        console: Whether to log to console

    Returns:
        Root logger instance
    """
    # Create logger
    logger = logging.getLogger("ai_analysis_engine")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        ensure_log_directory(log_file)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module

    Args:
        name: Module name

    Returns:
        Logger instance
    """
    return logging.getLogger(f"ai_analysis_engine.{name}")


def ensure_log_directory(log_file: str) -> None:
    """Ensure the directory for log file exists"""
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)


# Global logger setup
logger = setup_logging(
    log_file="./logs/ai_analysis_engine.log"
)
