"""
╔══════════════════════════════════════════════════════════════╗
║   Logging Configuration                                     ║
╚══════════════════════════════════════════════════════════════╝

Configures loguru-based logging with console and file sinks.
"""

import sys
from pathlib import Path
from loguru import logger

from configs.settings import Settings


def configure_logging():
    """Set up application-wide logging with loguru."""
    # Remove default handler
    logger.remove()

    # Console handler — colorized, human-readable
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        level=Settings.LOG_LEVEL,
        colorize=True,
    )

    # File handler — structured, rotated logs
    log_path = Path(Settings.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} — {message}",
        level=Settings.LOG_LEVEL,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,
    )

    return logger
