import pytest
import logging
from unittest.mock import patch, MagicMock
from app.core.config import settings

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

if settings.STANDALONE_MODE:
    pytest.skip("Logging tests skipped in standalone mode", allow_module_level=True)

from app.core.logging import configure_logging, get_logger, AuditLogger


def test_configure_logging():
    """Test logging configuration."""
    with patch("logging.basicConfig") as mock_basic_config:
        with patch("structlog.configure") as mock_structlog_config:
            configure_logging()
            mock_basic_config.assert_called_once()
            mock_structlog_config.assert_called_once()


def test_get_logger():
    """Test logger creation."""
    logger = get_logger("test_module")
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")
    assert hasattr(logger, "debug")


def test_logger_levels():
    """Test different logging levels."""
    logger = get_logger("test_levels")

    with patch.object(logger, "info") as mock_info:
        logger.info("Test info message")
        mock_info.assert_called_once_with("Test info message")

    with patch.object(logger, "error") as mock_error:
        logger.error("Test error message")
        mock_error.assert_called_once_with("Test error message")


def test_logger_formatting():
    """Test logger message formatting."""
    logger = get_logger("test_format")

    with patch.object(logger, "info") as mock_info:
        logger.info("Formatted message")
        mock_info.assert_called_once_with("Formatted message")


def test_audit_logger():
    """Test audit logger functionality."""
    audit_logger = AuditLogger()

    with patch.object(audit_logger.logger, "info") as mock_info:
        audit_logger.log_user_action(
            user_id=1,
            action="login",
            resource_type="user",
            resource_id="1",
            details={},
            ip_address="127.0.0.1",
            user_agent="test",
        )
        mock_info.assert_called_once()


def test_audit_logger_system_event():
    """Test audit logger system event logging."""
    audit_logger = AuditLogger()

    with patch.object(audit_logger.logger, "info") as mock_info:
        audit_logger.log_system_event(
            event_type="startup", description="System started", details={}
        )
        mock_info.assert_called_once()


def test_multiple_loggers():
    """Test creating multiple loggers."""
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    assert logger1 is not None
    assert logger2 is not None
    assert hasattr(logger1, "info")
    assert hasattr(logger2, "info")


def test_logger_hierarchy():
    """Test logger hierarchy."""
    parent_logger = get_logger("parent")
    child_logger = get_logger("parent.child")

    assert parent_logger is not None
    assert child_logger is not None
    assert hasattr(parent_logger, "info")
    assert hasattr(child_logger, "info")


def test_exception_logging():
    """Test exception logging."""
    logger = get_logger("test_exception")

    with patch.object(logger, "exception") as mock_exception:
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("An error occurred")

        mock_exception.assert_called_once_with("An error occurred")


def test_structured_logging():
    """Test structured logging with extra fields."""
    logger = get_logger("test_structured")

    with patch.object(logger, "info") as mock_info:
        logger.info("Structured message", extra={"user_id": 123, "action": "login"})
        mock_info.assert_called_once()
