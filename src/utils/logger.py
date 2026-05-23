"""
╔══════════════════════════════════════════════════════════════╗
║   Logger — Custom Logging Setup                             ║
╚══════════════════════════════════════════════════════════════╝

Provides a consistent logger interface across all modules.
"""

from loguru import logger
from configs.logging_config import configure_logging

# Initialize logging on first import
_initialized = False


def setup_logger(name: str = None):
    """
    Get a configured logger instance.

    Args:
        name: Module name for context (typically __name__)

    Returns:
        Configured loguru logger
    """
    global _initialized
    if not _initialized:
        configure_logging()
        _initialized = True

    return logger.bind(name=name or "root")
