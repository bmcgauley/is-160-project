"""
Logging configuration for IS-160 CNN Employment Trends Analysis Project.

This module provides centralized logging configuration and utilities
for tracking data processing steps, validation results, and system events.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Constants
LOG_DIR = Path("logs")
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in a structured JSON format for better parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry['extra_data'] = record.extra_data

        return json.dumps(log_entry, ensure_ascii=False)

class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds ANSI color codes for console output.
    """

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format the message with color
        formatted_message = super().format(record)
        return f"{color}{formatted_message}{reset}"

def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    structured: bool = False,
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to files
        log_to_console: Whether to log to console
        structured: Whether to use structured (JSON) formatting
        log_dir: Directory for log files (defaults to LOG_DIR)

    Returns:
        logging.Logger: Root logger instance
    """
    if log_dir is None:
        log_dir = LOG_DIR

    # Create log directory if it doesn't exist
    if log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Create formatters
    if structured:
        formatter = StructuredFormatter()
        console_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handlers
    if log_to_file:
        # General log file
        general_log_file = log_dir / "is160_project.log"
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Error log file (WARNING and above)
        error_log_file = log_dir / "is160_project_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)

        # Data processing log file
        data_log_file = log_dir / "data_processing.log"
        data_handler = logging.handlers.RotatingFileHandler(
            data_log_file,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT
        )
        data_handler.setLevel(numeric_level)
        data_handler.setFormatter(formatter)
        data_handler.addFilter(DataProcessingFilter())
        root_logger.addHandler(data_handler)

    return root_logger

class DataProcessingFilter(logging.Filter):
    """
    Filter for data processing related log messages.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Include messages from data-related modules
        data_modules = ['data_download', 'data_acquisition', 'validation']
        return any(module in record.name for module in data_modules) or \
               any(keyword in record.getMessage().lower() for keyword in
                   ['data', 'download', 'process', 'validate', 'csv', 'qcew'])

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

def log_function_call(logger: logging.Logger, func_name: str, args: Optional[Dict[str, Any]] = None):
    """
    Log function entry with parameters.

    Args:
        logger: Logger instance
        func_name: Function name
        args: Function arguments (optional)
    """
    if args:
        logger.debug(f"Calling {func_name} with args: {args}")
    else:
        logger.debug(f"Calling {func_name}")

def log_performance(logger: logging.Logger, operation: str, duration: float, extra_data: Optional[Dict[str, Any]] = None):
    """
    Log performance metrics.

    Args:
        logger: Logger instance
        operation: Operation description
        duration: Duration in seconds
        extra_data: Additional data to log
    """
    log_data = {
        'operation': operation,
        'duration_seconds': duration,
        'duration_ms': duration * 1000
    }

    if extra_data:
        log_data.update(extra_data)

    # Create a log record with extra data
    logger.info(f"Performance: {operation} completed in {duration:.2f}s", extra={'extra_data': log_data})

def log_data_validation(logger: logging.Logger, dataset: str, validation_results: Dict[str, Any]):
    """
    Log data validation results.

    Args:
        logger: Logger instance
        dataset: Dataset name
        validation_results: Validation results dictionary
    """
    logger.info(f"Data validation completed for {dataset}", extra={'extra_data': validation_results})

def log_error_with_context(logger: logging.Logger, error: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Log an error with additional context information.

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
    """
    error_data = {
        'error_type': type(error).__name__,
        'error_message': str(error)
    }

    if context:
        error_data.update(context)

    logger.error(f"Error occurred: {error}", extra={'extra_data': error_data}, exc_info=True)

# Convenience functions for common logging patterns
def log_data_download(logger: logging.Logger, filename: str, success: bool, file_size: Optional[int] = None):
    """Log data download results."""
    extra_data = {'filename': filename, 'success': success}
    if file_size is not None:
        extra_data['file_size_bytes'] = file_size

    if success:
        logger.info(f"Downloaded {filename}", extra={'extra_data': extra_data})
    else:
        logger.error(f"Failed to download {filename}", extra={'extra_data': extra_data})

def log_data_processing(logger: logging.Logger, operation: str, records_processed: int, duration: float):
    """Log data processing results."""
    extra_data = {
        'operation': operation,
        'records_processed': records_processed,
        'processing_rate': records_processed / duration if duration > 0 else 0
    }

    logger.info(f"Data processing: {operation} - {records_processed} records in {duration:.2f}s",
                extra={'extra_data': extra_data})

def create_session_log(session_id: str, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Create a session-specific logger for tracking a processing session.

    Args:
        session_id: Unique session identifier
        config: Session configuration

    Returns:
        logging.Logger: Session logger
    """
    session_logger = logging.getLogger(f"session.{session_id}")

    # Add session context to all log records from this logger
    session_filter = SessionContextFilter(session_id, config)
    session_logger.addFilter(session_filter)

    return session_logger

class SessionContextFilter(logging.Filter):
    """
    Filter that adds session context to log records.
    """

    def __init__(self, session_id: str, config: Optional[Dict[str, Any]] = None):
        self.session_id = session_id
        self.config = config or {}

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = self.session_id
        record.session_config = self.config
        return True

# Default setup for the project
def setup_default_logging():
    """Set up default logging configuration for the project."""
    return setup_logging(
        log_level="INFO",
        log_to_file=True,
        log_to_console=True,
        structured=False
    )

# Initialize default logging when module is imported
_default_logger = setup_default_logging()