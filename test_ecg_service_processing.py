"""ECG service processing methods tests to boost coverage."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, mock_open
import numpy as np
from datetime import datetime
import asyncio
import os

os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test.db"

from app.services.ecg_service import ECGAnalysisService
from app.core.constants import AnalysisStatus, ClinicalUrgency, DiagnosisCategory


@pytest.fixture
def ecg_service():
    """Create ECG service instance."""
    mock_db = Mock()
    mock_ml = Mock()
    mock_validation = Mock()
    service = ECGAnalysisService(mock_db, mock_ml, mock_validation)
    service.repository = Mock()
    service.processor = Mock()
    service.quality_analyzer = Mock()
    return service


@pytest.mark.asyncio
async def test_process_analysis_async_success(ecg_service):
    """Test successful ECG analysis processing - covers lines 107-184."""
    mock_analysis = Mock()
    mock_analysis.id = 1
    mock_analysis.file_path = "/tmp/test.txt"
    mock_analysis.sample_rate = 500
    mock_analysis.leads_names = ["I", "II"]
    mock_analysis.retry_count = 0

    ecg_service_instance.repository.get_analysis_by_id = AsyncMock(return_value=mock_analysis)
    ecg_service_instance.processor.load_ecg_file = AsyncMock(
        return_value=np.array([[1, 2], [3, 4]])
    )
    ecg_service_instance.processor.preprocess_signal = AsyncMock(
        return_value=np.array([[1, 2], [3, 4]])
    )
    ecg_service_instance.quality_analyzer.analyze_quality = AsyncMock(
        return_value={"overall_score": 0.9, "noise_level": 0.1, "baseline_wander": 0.05}
    )
    ecg_service_instance.ml_service.analyze_ecg = AsyncMock(
        return_value={
            "confidence": 0.85,
            "predictions": {"normal": 0.8},
            "interpretability": {},
            "rhythm": "sinus",
        }
    )
    ecg_service_instance._extract_measurements = Mock(
        return_value={
            "heart_rate": 75,
            "pr_interval": 160,
            "qrs_duration": 100,
            "qt_interval": 400,
            "qtc_interval": 420,
            "detailed_measurements": [],
        }
    )
    ecg_service_instance._generate_annotations = Mock(return_value=[])
    ecg_service_instance._assess_clinical_urgency = Mock(
        return_value={
            "urgency": ClinicalUrgency.LOW,
            "critical": False,
            "primary_diagnosis": "Normal ECG",
            "secondary_diagnoses": [],
            "category": DiagnosisCategory.NORMAL,
            "icd10_codes": [],
            "recommendations": [],
        }
    )
    ecg_service_instance.repository.update_analysis = AsyncMock()
    ecg_service_instance.repository.update_analysis_status = AsyncMock()
    ecg_service_instance.repository.create_measurement = AsyncMock()
    ecg_service_instance.repository.create_annotation = AsyncMock()
    ecg_service_instance.validation_service.create_urgent_validation = AsyncMock()

    with patch("app.services.ecg_service_instance.datetime") as mock_datetime:
        mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 12, 0, 0)

        await ecg_service_instance._process_analysis_async(1)

        ecg_service_instance.repository.update_analysis.assert_called()


@pytest.mark.asyncio
async def test_process_analysis_async_failure_with_retry(ecg_service):
    """Test ECG analysis processing failure with retry - covers lines 188-205."""
    mock_analysis = Mock()
    mock_analysis.retry_count = 1

    ecg_service_instance.repository.get_analysis_by_id = AsyncMock(return_value=mock_analysis)
    ecg_service_instance.processor.load_ecg_file = AsyncMock(side_effect=Exception("File error"))
    ecg_service_instance.repository.update_analysis = AsyncMock()
    ecg_service_instance.repository.update_analysis_status = AsyncMock()

    with patch("asyncio.sleep") as mock_sleep, patch(
        "asyncio.create_task"
    ) as mock_task:
        await ecg_service_instance._process_analysis_async(1)

        ecg_service_instance.repository.update_analysis.assert_called_with(
            1,
            {
                "status": AnalysisStatus.FAILED,
                "error_message": "File error",
                "retry_count": 2,
            },
        )


@pytest.mark.asyncio
async def test_process_analysis_async_critical_validation(ecg_service):
    """Test ECG analysis with critical validation - covers lines 181-182."""
    mock_analysis = Mock()
    mock_analysis.id = 1
    mock_analysis.file_path = "/tmp/test.txt"
    mock_analysis.sample_rate = 500
    mock_analysis.leads_names = ["I", "II"]
    mock_analysis.retry_count = 0

    ecg_service_instance.repository.get_analysis_by_id = AsyncMock(return_value=mock_analysis)
    ecg_service_instance.processor.load_ecg_file = AsyncMock(
        return_value=np.array([[1, 2], [3, 4]])
    )
    ecg_service_instance.processor.preprocess_signal = AsyncMock(
        return_value=np.array([[1, 2], [3, 4]])
    )
    ecg_service_instance.quality_analyzer.analyze_quality = AsyncMock(
        return_value={"overall_score": 0.9}
    )
    ecg_service_instance.ml_service.analyze_ecg = AsyncMock(
        return_value={"confidence": 0.85, "predictions": {}}
    )
    ecg_service_instance._extract_measurements = Mock(
        return_value={"heart_rate": 75, "detailed_measurements": []}
    )
    ecg_service_instance._generate_annotations = Mock(return_value=[])
    ecg_service_instance._assess_clinical_urgency = Mock(
        return_value={"urgency": ClinicalUrgency.CRITICAL, "critical": True}
    )
    ecg_service_instance.repository.update_analysis = AsyncMock()
    ecg_service_instance.repository.update_analysis_status = AsyncMock()
    ecg_service_instance.repository.create_measurement = AsyncMock()
    ecg_service_instance.repository.create_annotation = AsyncMock()
    ecg_service_instance.validation_service.create_urgent_validation = AsyncMock()

    with patch("app.services.ecg_service_instance.datetime") as mock_datetime:
        mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 12, 0, 0)

        await ecg_service_instance._process_analysis_async(1)

        ecg_service_instance.validation_service.create_urgent_validation.assert_called_once_with(
            1
        )


@pytest.mark.asyncio
async def test_calculate_file_info(ecg_service):
    """Test file info calculation - covers lines 207-221."""
    with patch("app.services.ecg_service_instance.Path") as mock_path, patch(
        "app.services.ecg_service_instance.hashlib.sha256"
    ) as mock_hash:
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_file.stat.return_value.st_size = 1024
        mock_path.return_value = mock_file

        mock_hash_obj = Mock()
        mock_hash_obj.hexdigest.return_value = "test_hash"
        mock_hash.return_value = mock_hash_obj

        with patch("app.services.ecg_service_instance.open", mock_open(read_data=b"test data")):
            file_hash, file_size = await ecg_service_instance._calculate_file_info(
                "/tmp/test.txt"
            )

            assert file_hash == "test_hash"
            assert file_size == 1024


@pytest.mark.asyncio
async def test_calculate_file_info_file_not_found(ecg_service):
    """Test file info calculation with missing file - covers lines 210-211."""
    with patch("pathlib.Path") as mock_path:
        mock_file = Mock()
        mock_file.exists.return_value = False
        mock_path.return_value = mock_file

        with pytest.raises(Exception):
            await ecg_service_instance._calculate_file_info("/nonexistent/file.txt")


@pytest.mark.asyncio
async def test_extract_measurements_success(ecg_service):
    """Test ECG measurements extraction - covers lines 223-272."""
    ecg_data = np.random.rand(5000, 12)
    sample_rate = 500

    with patch("neurokit2.ecg_process") as mock_process:
        mock_signals = np.random.rand(5000, 12)
        mock_info = {
            "ECG_Rate": np.array([75, 76, 74]),
            "ECG_R_Peaks": np.array([100, 600, 1100, 1600]),
        }
        mock_process.return_value = (mock_signals, mock_info)

        measurements = ecg_service_instance._extract_measurements(ecg_data, sample_rate)

        assert "heart_rate" in measurements
        assert "detailed_measurements" in measurements
        assert len(measurements["detailed_measurements"]) > 0


@pytest.mark.asyncio
async def test_extract_measurements_error_handling(ecg_service):
    """Test ECG measurements extraction error handling - covers lines 270-272."""
    ecg_data = np.random.rand(5000, 12)
    sample_rate = 500

    with patch("scipy.signal.find_peaks", side_effect=Exception("Processing error")):
        measurements = ecg_service_instance._extract_measurements(ecg_data, sample_rate)

        assert measurements["heart_rate"] is None
        assert measurements["detailed_measurements"] == []


@pytest.mark.asyncio
async def test_generate_annotations_success(ecg_service):
    """Test ECG annotations generation - covers lines 274-316."""
    ecg_data = np.random.rand(5000, 12)
    ai_results = {
        "events": [
            {
                "label": "arrhythmia",
                "time_ms": 1000,
                "confidence": 0.9,
                "properties": {},
            }
        ]
    }
    sample_rate = 500

    with patch("neurokit2.ecg_process") as mock_process:
        mock_signals = np.random.rand(5000, 12)
        mock_info = {"ECG_R_Peaks": np.array([100, 600, 1100])}
        mock_process.return_value = (mock_signals, mock_info)

        annotations = ecg_service_instance._generate_annotations(
            ecg_data, ai_results, sample_rate
        )

        assert len(annotations) > 0
        assert any(ann["annotation_type"] == "beat" for ann in annotations)
        assert any(ann["annotation_type"] == "event" for ann in annotations)


@pytest.mark.asyncio
async def test_generate_annotations_error_handling(ecg_service):
    """Test ECG annotations generation error handling - covers lines 314-316."""
    ecg_data = np.random.rand(5000, 12)
    ai_results = {}
    sample_rate = 500

    with patch("scipy.signal.find_peaks", side_effect=Exception("Processing error")):
        annotations = ecg_service_instance._generate_annotations(
            ecg_data, ai_results, sample_rate
        )

        assert annotations == []


@pytest.mark.asyncio
async def test_assess_clinical_urgency_critical(ecg_service):
    """Test clinical urgency assessment critical - covers lines 318-381."""
    ai_results = {
        "predictions": {"ventricular_fibrillation": 0.8, "normal": 0.1},
        "confidence": 0.9,
    }

    assessment = ecg_service_instance._assess_clinical_urgency(ai_results)

    assert assessment["urgency"] == ClinicalUrgency.CRITICAL
    assert assessment["critical"] is True
    assert "Immediate medical attention required" in assessment["recommendations"]


@pytest.mark.asyncio
async def test_assess_clinical_urgency_high_priority(ecg_service):
    """Test clinical urgency assessment high priority - covers lines 355-370."""
    ai_results = {
        "predictions": {"atrial_fibrillation": 0.7, "normal": 0.2},
        "confidence": 0.8,
    }

    assessment = ecg_service_instance._assess_clinical_urgency(ai_results)

    assert assessment["urgency"] == ClinicalUrgency.HIGH
    assert assessment["critical"] is False
    assert "Cardiology consultation recommended" in assessment["recommendations"]


@pytest.mark.asyncio
async def test_assess_clinical_urgency_low_confidence(ecg_service):
    """Test clinical urgency with low AI confidence - covers lines 372-375."""
    ai_results = {"predictions": {"normal": 0.6}, "confidence": 0.5}

    assessment = ecg_service_instance._assess_clinical_urgency(ai_results)

    assert (
        "Manual review recommended due to low AI confidence"
        in assessment["recommendations"]
    )


@pytest.mark.asyncio
async def test_assess_clinical_urgency_error_handling(ecg_service):
    """Test clinical urgency assessment error handling - covers lines 379-381."""
    ai_results = None

    assessment = ecg_service_instance._assess_clinical_urgency(ai_results)

    assert assessment["urgency"] == ClinicalUrgency.LOW
    assert assessment["critical"] is False
    assert assessment["primary_diagnosis"] == "Normal ECG"


@pytest.mark.asyncio
async def test_get_analysis_by_id(ecg_service):
    """Test get analysis by ID - covers lines 383-385."""
    mock_analysis = Mock()
    ecg_service_instance.repository.get_analysis_by_id = AsyncMock(return_value=mock_analysis)

    result = await ecg_service_instance.get_analysis_by_id(1)

    assert result == mock_analysis
    ecg_service_instance.repository.get_analysis_by_id.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_get_analyses_by_patient(ecg_service):
    """Test get analyses by patient - covers lines 387-391."""
    mock_analyses = [Mock(), Mock()]
    ecg_service_instance.repository.get_analyses_by_patient = AsyncMock(
        return_value=mock_analyses
    )

    result = await ecg_service_instance.get_analyses_by_patient(1, limit=10, offset=0)

    assert result == mock_analyses
    ecg_service_instance.repository.get_analyses_by_patient.assert_called_once_with(1, 10, 0)


@pytest.mark.asyncio
async def test_search_analyses(ecg_service):
    """Test search analyses - covers lines 393-400."""
    mock_analyses = [Mock(), Mock()]
    ecg_service_instance.repository.search_analyses = AsyncMock(return_value=(mock_analyses, 2))

    filters = {"patient_id": 1, "status": "completed"}
    result = await ecg_service_instance.search_analyses(filters, limit=10, offset=0)

    assert result == (mock_analyses, 2)
    ecg_service_instance.repository.search_analyses.assert_called_once_with(filters, 10, 0)


@pytest.mark.asyncio
async def test_delete_analysis(ecg_service):
    """Test delete analysis - covers lines 402-404."""
    ecg_service_instance.repository.delete_analysis = AsyncMock(return_value=True)

    result = await ecg_service_instance.delete_analysis(1)

    assert result is True
    ecg_service_instance.repository.delete_analysis.assert_called_once_with(1)
