"""Centralized logging configuration"""
import sys
from pathlib import Path

from loguru import logger

from src.core.config import get_settings

settings = get_settings()


def setup_logging():
    """Configure application logging with loguru"""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with color formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )
    
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # File handler for errors only
    logger.add(
        "logs/errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )
    
    # File handler for all logs
    logger.add(
        "logs/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="50 MB",
        retention="7 days",
        compression="zip",
    )
    
    logger.info(f"Logging configured - Level: {settings.log_level}")
    
    return logger
