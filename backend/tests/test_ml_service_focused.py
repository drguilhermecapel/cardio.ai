"""Focused ML Model Service Tests."""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
import numpy as np
from unittest.mock import Mock, patch
from app.services.ml_model_service import MLModelService


@pytest.fixture
def ml_service():
    """Create ML service instance."""
    with patch("onnxruntime.InferenceSession"):
        return MLModelService()


@pytest.mark.asyncio
async def test_analyze_ecg_success(ml_service):
    """Test successful ECG analysis."""
    mock_ecg_model = Mock()
    mock_ecg_model.run.return_value = [
        np.array(
            [
                [
                    0.1,
                    0.9,
                    0.8,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                ]
            ]
        )
    ]
    mock_ecg_model.get_inputs.return_value = [Mock(name="input")]

    mock_rhythm_model = Mock()
    mock_rhythm_model.run.return_value = [
        np.array([[0.3, 0.7, 0.1, 0.2, 0.4, 0.5, 0.6]])
    ]
    mock_rhythm_model.get_inputs.return_value = [Mock(name="input")]

    mock_quality_model = Mock()
    mock_quality_model.run.return_value = [np.array([[[0.85]]])]
    mock_quality_model.get_inputs.return_value = [Mock(name="input")]

    ml_service.models = {
        "ecg_classifier": mock_ecg_model,
        "rhythm_detector": mock_rhythm_model,
        "quality_assessor": mock_quality_model,
    }

    ecg_data = np.random.randn(5000, 12).astype(np.float32)
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

    result = await ml_service.analyze_ecg(
        ecg_data=ecg_data, sample_rate=sample_rate, leads_names=leads_names
    )

    assert result is not None
    assert "predictions" in result
    assert "confidence" in result
    assert "rhythm" in result
    assert result["confidence"] > 0


@pytest.mark.asyncio
async def test_analyze_ecg_no_models(ml_service):
    """Test ECG analysis when no models are loaded."""
    ml_service.models = {}  # No models loaded

    ecg_data = np.random.randn(5000, 12).astype(np.float32)
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

    result = await ml_service.analyze_ecg(
        ecg_data=ecg_data, sample_rate=sample_rate, leads_names=leads_names
    )

    assert result is not None
    assert result["confidence"] == 0.0
    assert result["rhythm"] == "Unknown"


def test_get_model_info(ml_service):
    """Test getting model information."""
    ml_service.models = {"ecg_classifier": Mock(), "rhythm_detector": Mock()}
    ml_service.model_metadata = {
        "ecg_classifier": {"version": "1.0", "accuracy": 0.95},
        "rhythm_detector": {"version": "1.1", "accuracy": 0.92},
    }

    info = ml_service.get_model_info()

    assert info is not None
    assert "loaded_models" in info
    assert "ecg_classifier" in info["loaded_models"]
    assert "rhythm_detector" in info["loaded_models"]


def test_unload_model_success(ml_service):
    """Test successful model unloading."""
    ml_service.models = {"ecg_classifier": Mock(), "rhythm_detector": Mock()}
    ml_service.model_metadata = {
        "ecg_classifier": {"version": "1.0"},
        "rhythm_detector": {"version": "1.1"},
    }

    success = ml_service.unload_model("ecg_classifier")

    assert success is True
    assert "ecg_classifier" not in ml_service.models
    assert "ecg_classifier" not in ml_service.model_metadata


def test_unload_model_not_found(ml_service):
    """Test unloading non-existent model."""
    ml_service.models = {"rhythm_detector": Mock()}

    success = ml_service.unload_model("nonexistent_model")

    assert success is False
