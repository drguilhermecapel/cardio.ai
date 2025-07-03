import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from app.core.config import settings

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

if settings.STANDALONE_MODE:
    pytest.skip("Security tests skipped in standalone mode", allow_module_level=True)

from app.core.security import (
    create_access_token,
    verify_password,
    get_password_hash,
    decode_access_token,
)


def test_create_access_token():
    """Test JWT access token creation."""
    data = {"sub": "test@example.com"}

    with patch("app.core.security.jwt.encode") as mock_encode:
        mock_encode.return_value = "test_token"

        token = create_access_token(data)

        mock_encode.assert_called_once()
        assert token == "test_token"


def test_create_access_token_with_expiry():
    """Test JWT access token creation with custom expiry."""
    data = {"sub": "test@example.com"}
    expires_delta = timedelta(hours=2)

    with patch("app.core.security.jwt.encode") as mock_encode:
        mock_encode.return_value = "test_token_with_expiry"

        token = create_access_token(data, expires_delta)

        mock_encode.assert_called_once()
        assert token == "test_token_with_expiry"


def test_verify_password():
    """Test password verification."""
    plain_password = "test_password"
    hashed_password = "$2b$12$test_hash"

    with patch("app.core.security.pwd_context.verify") as mock_verify:
        mock_verify.return_value = True

        result = verify_password(plain_password, hashed_password)

        mock_verify.assert_called_once_with(plain_password, hashed_password)
        assert result is True


def test_verify_password_invalid():
    """Test password verification with invalid password."""
    plain_password = "wrong_password"
    hashed_password = "$2b$12$test_hash"

    with patch("app.core.security.pwd_context.verify") as mock_verify:
        mock_verify.return_value = False

        result = verify_password(plain_password, hashed_password)

        mock_verify.assert_called_once_with(plain_password, hashed_password)
        assert result is False


def test_get_password_hash():
    """Test password hashing."""
    password = "test_password"

    with patch("app.core.security.pwd_context.hash") as mock_hash:
        mock_hash.return_value = "$2b$12$hashed_password"

        hashed = get_password_hash(password)

        mock_hash.assert_called_once_with(password)
        assert hashed == "$2b$12$hashed_password"


def test_decode_access_token():
    """Test JWT access token decoding."""
    token = "valid_jwt_token"

    with patch("app.core.security.verify_token") as mock_verify:
        mock_verify.return_value = {
            "sub": "test@example.com",
            "exp": 1234567890,
            "type": "access",
        }

        payload = decode_access_token(token)

        mock_verify.assert_called_once_with(token, "access")
        assert payload is not None
        assert payload["sub"] == "test@example.com"


def test_decode_access_token_invalid():
    """Test JWT access token decoding with invalid token."""
    token = "invalid_jwt_token"

    with patch("app.core.security.verify_token") as mock_verify:
        from app.core.exceptions import AuthenticationException

        mock_verify.side_effect = AuthenticationException("Invalid token")

        payload = decode_access_token(token)

        mock_verify.assert_called_once_with(token, "access")
        assert payload is None


def test_token_expiry_validation():
    """Test token expiry validation."""
    with patch("app.core.security.datetime") as mock_datetime:
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.utcnow.return_value = mock_now

        data = {"sub": "test@example.com"}
        expires_delta = timedelta(minutes=30)

        with patch("app.core.security.jwt.encode") as mock_encode:
            create_access_token(data, expires_delta)

            call_args = mock_encode.call_args[0][0]
            assert "exp" in call_args


def test_password_context_configuration():
    """Test password context configuration."""
    with patch("app.core.security.CryptContext") as mock_context:
        mock_instance = MagicMock()
        mock_context.return_value = mock_instance

        from app.core.security import pwd_context

        assert pwd_context is not None


def test_jwt_algorithm_configuration():
    """Test JWT algorithm configuration."""
    from app.core.security import ALGORITHM

    assert ALGORITHM == "HS256"


def test_access_token_expire_minutes():
    """Test access token expiration configuration."""
    from app.core.security import ACCESS_TOKEN_EXPIRE_MINUTES

    assert isinstance(ACCESS_TOKEN_EXPIRE_MINUTES, int)
    assert ACCESS_TOKEN_EXPIRE_MINUTES > 0
