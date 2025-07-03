"""Focused ML model service tests for actual methods."""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
from unittest.mock import AsyncMock, Mock, patch
import numpy as np

from app.services.ml_model_service import MLModelService


@pytest.fixture
def ml_model_service():
    """Create ML model service instance."""
    with patch("app.services.ml_model_service.Path.exists", return_value=False):
        return MLModelService()


@pytest.fixture
def sample_ecg_data():
    """Sample ECG data as numpy array."""
    return np.random.randn(5000, 12).astype(np.float32)


@pytest.mark.asyncio
async def test_analyze_ecg(ml_model_service, sample_ecg_data):
    """Test ECG analysis with actual method."""
    sample_rate = 500
    leads_names = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]

    ml_model_service.models = {}

    result = await ml_model_service.analyze_ecg(
        sample_ecg_data, sample_rate, leads_names
    )

    assert "confidence" in result
    assert "predictions" in result
    assert "rhythm" in result
    assert "events" in result
    assert isinstance(result["confidence"], float)


@pytest.mark.asyncio
async def test_get_model_info(ml_model_service):
    """Test getting model information."""
    info = ml_model_service.get_model_info()

    assert "loaded_models" in info
    assert "model_metadata" in info
    assert "memory_usage" in info
    assert isinstance(info["loaded_models"], list)


@pytest.mark.asyncio
async def test_unload_model(ml_model_service):
    """Test unloading a model."""
    ml_model_service.models["test_model"] = Mock()
    ml_model_service.model_metadata["test_model"] = {}

    result = ml_model_service.unload_model("test_model")

    assert result is True
    assert "test_model" not in ml_model_service.models


@pytest.mark.asyncio
async def test_unload_nonexistent_model(ml_model_service):
    """Test unloading a model that doesn't exist."""
    result = ml_model_service.unload_model("nonexistent_model")

    assert result is False
