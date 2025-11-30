import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "rag_app.log"
DEBUG_LOG_FILE = LOG_DIR / "debug.log"
SECURITY_LOG_FILE = LOG_DIR / "security.log"

class StructuredFormatter(logging.Formatter):
    """Enhanced formatter for better debugging and LLM readability"""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": getattr(record, 'module', record.name.split('.')[-1]),
            "function": getattr(record, 'funcName', 'unknown'),
            "line": getattr(record, 'lineno', 0)
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields for debugging
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        # Format for readability
        if record.levelname in ['ERROR', 'CRITICAL']:
            return f"[{log_entry['timestamp']}] {log_entry['level']} | {log_entry['logger']} | {log_entry['message']}" + \
                   (f" | Exception: {log_entry.get('exception', '')}" if 'exception' in log_entry else "")
        elif record.levelname == 'WARNING':
            return f"[{log_entry['timestamp']}] {log_entry['level']} | {log_entry['logger']} | {log_entry['message']}"
        elif record.levelname == 'INFO':
            return f"[{log_entry['timestamp']}] {log_entry['level']} | {log_entry['logger']} | {log_entry['message']}"
        else:
            return f"[{log_entry['timestamp']}] {log_entry['level']} | {log_entry['logger']} | {log_entry['message']}"

class SecurityFormatter(logging.Formatter):
    """Special formatter for security-related logs"""
    
    def format(self, record):
        timestamp = datetime.utcnow().isoformat() + "Z"
        return f"[{timestamp}] SECURITY | {record.getMessage()}"

def setup_logging():
    """Setup enhanced logging with multiple handlers and formatters"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Structured formatter for main logs
    structured_formatter = StructuredFormatter()
    
    # Simple formatter for console (less verbose)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Security formatter
    security_formatter = SecurityFormatter()

    # Main application log (INFO and above)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10_000_000, backupCount=10
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(structured_formatter)

    # Debug log (ALL levels) - for detailed debugging
    debug_handler = RotatingFileHandler(
        DEBUG_LOG_FILE, maxBytes=20_000_000, backupCount=5
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(structured_formatter)

    # Security log (WARNING and above) - for security events
    security_handler = RotatingFileHandler(
        SECURITY_LOG_FILE, maxBytes=5_000_000, backupCount=10
    )
    security_handler.setLevel(logging.WARNING)
    security_handler.setFormatter(security_formatter)

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(debug_handler)
        logger.addHandler(security_handler)
        logger.addHandler(console_handler)
        
        # Log startup message
        logger.info("RAG Application logging initialized - Main: %s, Debug: %s, Security: %s",
                   LOG_FILE, DEBUG_LOG_FILE, SECURITY_LOG_FILE)

    return logger

# Utility functions for structured logging
def log_user_action(logger, action: str, user_id: str = None, **kwargs):
    """Log user actions with consistent format"""
    extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"USER_ACTION: {action} | user_id={user_id or 'anonymous'} | {extra_info}")

def log_security_event(logger, event: str, user_id: str = None, **kwargs):
    """Log security events with consistent format"""
    extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.warning(f"SECURITY_EVENT: {event} | user_id={user_id or 'anonymous'} | {extra_info}")

def log_performance_metric(logger, operation: str, duration_ms: float, **kwargs):
    """Log performance metrics"""
    extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"PERFORMANCE: {operation} | duration_ms={duration_ms:.2f} | {extra_info}")

def log_llm_interaction(logger, provider: str, prompt_tokens: int, response_tokens: int, **kwargs):
    """Log LLM interactions for debugging"""
    extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"LLM_INTERACTION: provider={provider} | prompt_tokens={prompt_tokens} | response_tokens={response_tokens} | {extra_info}")

def log_sensitive_debug(logger, message: str, **sensitive_data):
    """Log sensitive information for debugging (to be removed in production)"""
    # WARNING: This logs sensitive data - remove in production
    sensitive_info = " | ".join([f"{k}={v}" for k, v in sensitive_data.items()])
    logger.debug(f"SENSITIVE_DEBUG: {message} | {sensitive_info} | [REMOVE_IN_PRODUCTION]")
