"""Colored console logging utilities."""

from enum import Enum
from typing import Any


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _log(prefix: str, color: str, message: str, *args: Any) -> None:
    """Internal helper to print colored log messages."""
    formatted = message % args if args else message
    print(f"{color}{prefix}{Colors.RESET} {formatted}")


def log_header(message: str, *args: Any) -> None:
    """Log a header/section message."""
    _log(">>> ", Colors.BOLD + Colors.CYAN, message, *args)


def log_info(message: str, *args: Any) -> None:
    """Log an info message."""
    _log("[INFO] ", Colors.BLUE, message, *args)


def log_success(message: str, *args: Any) -> None:
    """Log a success message."""
    _log("[OK] ", Colors.GREEN, message, *args)


def log_warning(message: str, *args: Any) -> None:
    """Log a warning message."""
    _log("[WARN] ", Colors.YELLOW, message, *args)


def log_error(message: str, *args: Any) -> None:
    """Log an error message."""
    _log("[ERROR] ", Colors.RED, message, *args)
