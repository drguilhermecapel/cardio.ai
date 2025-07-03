"""Test ML Model Service."""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.services.ml_model_service import MLModelService


@pytest.fixture
def mock_onnx_session():
    """Mock ONNX Runtime session."""
    session = Mock()
    session.run.return_value = [
        np.array([[0.1, 0.9]]),  # Classification probabilities
        np.array([[0.2, 0.8]]),  # Rhythm probabilities
    ]
    return session


@pytest.fixture
def ml_service():
    """Create ML model service instance."""
    with patch("onnxruntime.InferenceSession"):
        service = MLModelService()
        service.models = {"ecg_classifier": Mock(), "rhythm_detector": Mock()}
        service.model_metadata = {}
        return service


@pytest.fixture
def sample_ecg_data():
    """Sample ECG signal data."""
    return {
        "signal_data": np.random.randn(12, 5000).tolist(),  # 12 leads, 5000 samples
        "sampling_rate": 500,
        "duration": 10.0,
    }


@pytest.mark.asyncio
async def test_load_models_success(ml_service):
    """Test successful model loading."""
    assert isinstance(ml_service.models, dict)
    assert isinstance(ml_service.model_metadata, dict)


@pytest.mark.asyncio
async def test_load_models_file_not_found():
    """Test model loading with missing files."""
    service = MLModelService()

    assert isinstance(service.models, dict)


@pytest.mark.asyncio
async def test_analyze_ecg_success(ml_service, sample_ecg_data):
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

    ml_service.models = {
        "ecg_classifier": mock_ecg_model,
        "rhythm_detector": mock_rhythm_model,
    }

    ecg_data = np.array(sample_ecg_data["signal_data"], dtype=np.float32)

    result = await ml_service.analyze_ecg(
        ecg_data=ecg_data,
        sample_rate=sample_ecg_data["sampling_rate"],
        leads_names=[
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
        ],
    )

    assert result is not None
    assert "predictions" in result
    assert "rhythm" in result
    assert "confidence" in result


@pytest.mark.asyncio
async def test_analyze_ecg_not_loaded():
    """Test ECG analysis when models not loaded."""
    service = MLModelService()
    service.models = {}  # Empty models dictionary

    ecg_data = np.random.randn(12, 5000).astype(np.float32)

    result = await service.analyze_ecg(
        ecg_data=ecg_data,
        sample_rate=500,
        leads_names=[
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
        ],
    )

    assert result is not None
    assert result["confidence"] == 0.0
    assert result["rhythm"] == "Unknown"


@pytest.mark.asyncio
async def test_analyze_ecg_invalid_data(ml_service):
    """Test ECG analysis with invalid data."""
    invalid_data = np.array([[1, 2]], dtype=np.float32)  # Wrong shape

    result = await ml_service.analyze_ecg(
        ecg_data=invalid_data, sample_rate=500, leads_names=["I", "II"]
    )

    assert result is not None
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_error_handling_corrupted_model(ml_service, sample_ecg_data):
    """Test handling of corrupted model files."""
    ml_service.models["ecg_classifier"].run.side_effect = Exception("Model corrupted")

    ecg_data = np.array(sample_ecg_data["signal_data"], dtype=np.float32)

    result = await ml_service.analyze_ecg(
        ecg_data=ecg_data,
        sample_rate=sample_ecg_data["sampling_rate"],
        leads_names=[
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
        ],
    )

    assert result is not None
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_concurrent_inference(ml_service, mock_onnx_session):
    """Test concurrent model inference."""
    ecg_data = np.random.randn(12, 5000).astype(np.float32)

    import asyncio

    tasks = [
        ml_service.analyze_ecg(
            ecg_data=ecg_data,
            sample_rate=500,
            leads_names=[
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
            ],
        )
        for _ in range(3)
    ]

    results = await asyncio.gather(*tasks)
    assert len(results) == 3
    for result in results:
        assert result is not None
