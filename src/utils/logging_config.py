"""
Structured logging configuration using structlog.

Sets up JSON logging for production and coloured console logging for development.
"""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """
    Configure logging for the terminal.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, emit JSON lines (for log aggregators).
                     If False, emit human-readable coloured output.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    try:
        import structlog

        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
        ]

        if json_output:
            renderer = structlog.processors.JSONRenderer()
        else:
            renderer = structlog.dev.ConsoleRenderer(colors=True)

        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            cache_logger_on_first_use=True,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
            foreign_pre_chain=shared_processors,
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(log_level)

        # Silence noisy third-party loggers
        for noisy in ("aiohttp", "websockets", "web3", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    except ImportError:
        # Fallback to standard logging if structlog is not installed
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            stream=sys.stdout,
        )
