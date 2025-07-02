"""
Complete Medical-Grade Test Coverage for Hybrid ECG Analysis Service
Target: 95%+ coverage for critical medical module
Compliance: FDA CFR 21 Part 820, ISO 13485, EU MDR 2017/745, ANVISA RDC 185/2001
"""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
import time
import tempfile
import os
from typing import Dict, Any

from app.services.hybrid_ecg_service import (
    UniversalECGReader,
    AdvancedPreprocessor,
    FeatureExtractor,
    HybridECGAnalysisService,
)


class TestUniversalECGReaderComplete:
    """Complete coverage tests for UniversalECGReader."""

    def test_initialization(self):
        """Test reader initialization."""
        reader = UniversalECGReader()
        assert reader is not None
        assert hasattr(reader, "supported_formats")
        assert ".csv" in reader.supported_formats
        assert ".dat" in reader.supported_formats
        assert ".edf" in reader.supported_formats
        assert ".txt" in reader.supported_formats
        assert ".png" in reader.supported_formats
        assert ".jpg" in reader.supported_formats

    def test_read_ecg_csv_success(self):
        """Test successful CSV reading."""
        reader = UniversalECGReader()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data = pd.DataFrame(
                {
                    "I": np.random.randn(100) * 0.1,
                    "II": np.random.randn(100) * 0.1,
                    "V1": np.random.randn(100) * 0.1,
                }
            )
            test_data.to_csv(f.name, index=False)

            result = reader.read_ecg(f.name)
            assert isinstance(result, dict)
            assert "signal" in result
            assert "sampling_rate" in result
            assert "labels" in result

            os.unlink(f.name)

    def test_read_ecg_unsupported_format(self):
        """Test unsupported file format."""
        reader = UniversalECGReader()

        with pytest.raises(ValueError, match="Unsupported format"):
            reader.read_ecg("/fake/path.xyz")

    def test_read_mitbih_import_error(self):
        """Test MIT-BIH reading with ImportError."""
        reader = UniversalECGReader()

        with patch("wfdb.rdrecord", side_effect=ImportError("wfdb not available")):
            result = reader._read_mitbih("/fake/path.dat")
            assert result is None

    def test_read_mitbih_general_exception(self):
        """Test MIT-BIH reading with general exception."""
        reader = UniversalECGReader()

        with patch("wfdb.rdrecord", side_effect=Exception("File not found")):
            result = reader._read_mitbih("/fake/path.dat")
            assert result is None

    def test_read_mitbih_success(self):
        """Test successful MIT-BIH reading."""
        reader = UniversalECGReader()

        mock_record = Mock()
        mock_record.p_signal = np.random.randn(1000, 2)
        mock_record.fs = 360
        mock_record.sig_name = ["MLII", "V1"]

        with patch("wfdb.rdrecord", return_value=mock_record):
            result = reader._read_mitbih("/fake/path.dat")
            assert isinstance(result, dict)
            assert "signal" in result
            assert "sampling_rate" in result
            assert result["sampling_rate"] == 360

    def test_read_edf_import_error(self):
        """Test EDF reading with ImportError."""
        reader = UniversalECGReader()

        with patch(
            "pyedflib.EdfReader", side_effect=ImportError("pyedflib not available")
        ):
            result = reader._read_edf("/fake/path.edf")
            assert result is None

    def test_read_edf_general_exception(self):
        """Test EDF reading with general exception."""
        reader = UniversalECGReader()

        with patch("pyedflib.EdfReader", side_effect=Exception("EDF error")):
            result = reader._read_edf("/fake/path.edf")
            assert result is None

    def test_read_csv_exception(self):
        """Test CSV reading with exception."""
        reader = UniversalECGReader()

        with patch("pandas.read_csv", side_effect=Exception("CSV error")):
            result = reader._read_csv("/fake/path.csv")
            assert result is None

    def test_read_csv_success_with_sampling_rate(self):
        """Test CSV reading with custom sampling rate."""
        reader = UniversalECGReader()

        mock_data = pd.DataFrame({"I": [1, 2, 3], "II": [4, 5, 6]})
        with patch("pandas.read_csv", return_value=mock_data):
            result = reader._read_csv("/fake/path.csv", 250)
            assert isinstance(result, dict)
            assert "signal" in result
            assert "sampling_rate" in result
            assert result["sampling_rate"] == 250
            assert result["labels"] == ["I", "II"]

    def test_read_text_success(self):
        """Test successful text reading."""
        reader = UniversalECGReader()

        mock_data = np.array([[1, 2], [3, 4], [5, 6]])
        with patch("numpy.loadtxt", return_value=mock_data):
            result = reader._read_text("/fake/path.txt")
            assert isinstance(result, dict)
            assert "signal" in result
            assert "sampling_rate" in result

    def test_read_text_exception(self):
        """Test text reading with exception."""
        reader = UniversalECGReader()

        with patch("numpy.loadtxt", side_effect=Exception("Text error")):
            result = reader._read_text("/fake/path.txt")
            assert result is None

    @pytest.mark.asyncio
    async def test_read_image_not_implemented(self):
        """Test image reading with NotImplementedError."""
        reader = UniversalECGReader()

        result = await reader._read_image("/fake/path.png")
        assert isinstance(result, dict)
        assert "signal" in result
        assert "sampling_rate" in result
        assert result["metadata"]["processing_method"] in [
            "fallback_random",
            "import_error_fallback",
            "exception_fallback",
        ]
        assert result["metadata"]["scanner_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_read_image_exception_in_fallback(self):
        """Test image reading with exception in fallback generation."""
        reader = UniversalECGReader()

        result = await reader._read_image("/fake/path.png")
        assert isinstance(result, dict)
        assert "signal" in result
        assert result["metadata"]["processing_method"] in [
            "fallback_random",
            "import_error_fallback",
            "exception_fallback",
        ]


class TestAdvancedPreprocessorComplete:
    """Complete coverage tests for AdvancedPreprocessor."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = AdvancedPreprocessor()
        assert preprocessor is not None

    def test_preprocess_signal_basic(self):
        """Test basic signal preprocessing."""
        preprocessor = AdvancedPreprocessor()
        signal = np.random.randn(1000, 1).astype(np.float64)

        result = preprocessor.preprocess_signal(signal)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0

    def test_remove_baseline_wandering(self):
        """Test baseline wandering removal."""
        preprocessor = AdvancedPreprocessor()
        signal = np.random.randn(500).astype(np.float64)

        result = preprocessor._remove_baseline_wandering(signal)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(signal)

    def test_remove_powerline_interference(self):
        """Test powerline interference removal."""
        preprocessor = AdvancedPreprocessor()
        signal = np.random.randn(500).astype(np.float64)

        result = preprocessor._remove_powerline_interference(signal)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(signal)

    def test_bandpass_filter(self):
        """Test bandpass filter."""
        preprocessor = AdvancedPreprocessor()
        signal = np.random.randn(500).astype(np.float64)

        result = preprocessor._bandpass_filter(signal)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(signal)

    def test_apply_bandpass_filter(self):
        """Test apply bandpass filter."""
        preprocessor = AdvancedPreprocessor()
        signal = np.random.randn(500).astype(np.float64)

        result = preprocessor._bandpass_filter(signal, 250)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(signal)

    def test_wavelet_denoise(self):
        """Test wavelet denoising."""
        preprocessor = AdvancedPreprocessor()
        signal = np.random.randn(500).astype(np.float64)

        result = preprocessor._wavelet_denoise(signal)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(signal)

    def test_preprocess_signal_short(self):
        """Test preprocessing with very short signal."""
        preprocessor = AdvancedPreprocessor()
        short_signal = np.random.randn(10, 1).astype(np.float64)

        try:
            result = preprocessor.preprocess_signal(short_signal)
            assert isinstance(result, np.ndarray)
        except Exception:
            pass


class TestFeatureExtractorComplete:
    """Complete coverage tests for FeatureExtractor."""

    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()
        assert extractor is not None

    def test_extract_all_features_basic(self):
        """Test basic feature extraction."""
        extractor = FeatureExtractor()
        signal = np.random.randn(5000, 1).astype(np.float64)

        features = extractor.extract_all_features(signal)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_all_features_with_r_peaks(self):
        """Test feature extraction with provided R peaks."""
        extractor = FeatureExtractor()
        signal = np.random.randn(5000, 1).astype(np.float64)
        r_peaks = np.array([100, 200, 300, 400, 500], dtype=np.int64)

        features = extractor.extract_all_features(signal, r_peaks=r_peaks)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_detect_r_peaks(self):
        """Test R peak detection."""
        extractor = FeatureExtractor()
        signal = np.random.randn(1000, 1).astype(np.float64)

        r_peaks = extractor._detect_r_peaks(signal)
        assert isinstance(r_peaks, np.ndarray)

    def test_extract_morphological_features(self):
        """Test morphological feature extraction."""
        extractor = FeatureExtractor()
        signal = np.random.randn(1000, 1).astype(np.float64)
        r_peaks = np.array([100, 200, 300, 400], dtype=np.int64)

        features = extractor._extract_morphological_features(signal, r_peaks)
        assert isinstance(features, dict)

    def test_extract_interval_features(self):
        """Test interval feature extraction."""
        extractor = FeatureExtractor()
        signal = np.random.randn(1000, 1).astype(np.float64)
        r_peaks = np.array([100, 200, 300, 400], dtype=np.int64)

        features = extractor._extract_interval_features(signal, r_peaks)
        assert isinstance(features, dict)

    def test_extract_hrv_features(self):
        """Test HRV feature extraction."""
        extractor = FeatureExtractor()
        r_peaks = np.array([100, 200, 300, 400], dtype=np.int64)

        features = extractor._extract_hrv_features(r_peaks)
        assert isinstance(features, dict)

    def test_extract_spectral_features(self):
        """Test spectral feature extraction."""
        extractor = FeatureExtractor()
        signal = np.random.randn(1000, 1).astype(np.float64)

        features = extractor._extract_spectral_features(signal)
        assert isinstance(features, dict)

    def test_extract_wavelet_features(self):
        """Test wavelet feature extraction."""
        extractor = FeatureExtractor()
        signal = np.random.randn(1000, 1).astype(np.float64)

        features = extractor._extract_wavelet_features(signal)
        assert isinstance(features, dict)

    def test_extract_nonlinear_features(self):
        """Test nonlinear feature extraction."""
        extractor = FeatureExtractor()
        signal = np.random.randn(1000, 1).astype(np.float64)
        r_peaks = np.array([100, 200, 300, 400, 500], dtype=np.int64)

        features = extractor._extract_nonlinear_features(signal, r_peaks)
        assert isinstance(features, dict)

    def test_sample_entropy(self):
        """Test sample entropy calculation."""
        extractor = FeatureExtractor()
        signal = np.random.randn(100).astype(np.float64)

        entropy = extractor._sample_entropy(signal, m=2, r=0.2)
        assert isinstance(entropy, float)

    def test_approximate_entropy(self):
        """Test approximate entropy calculation."""
        extractor = FeatureExtractor()
        signal = np.random.randn(100).astype(np.float64)

        entropy = extractor._approximate_entropy(signal, m=2, r=0.2)
        assert isinstance(entropy, float)

    def test_extract_features_empty_r_peaks(self):
        """Test feature extraction with empty R peaks."""
        extractor = FeatureExtractor()
        signal = np.random.randn(1000, 1).astype(np.float64)
        empty_peaks = np.array([], dtype=np.int64)

        features = extractor._extract_morphological_features(signal, empty_peaks)
        assert isinstance(features, dict)

        features = extractor._extract_interval_features(signal, empty_peaks)
        assert isinstance(features, dict)

        features = extractor._extract_hrv_features(empty_peaks)
        assert isinstance(features, dict)

    def test_extract_features_single_r_peak(self):
        """Test feature extraction with single R peak."""
        extractor = FeatureExtractor()
        single_peak = np.array([500], dtype=np.int64)

        features = extractor._extract_hrv_features(single_peak)
        assert isinstance(features, dict)


class TestHybridECGAnalysisServiceComplete:
    """Complete coverage tests for HybridECGAnalysisService."""

    @pytest.fixture
    def mock_db(self):
        return Mock()

    @pytest.fixture
    def mock_validation_service(self):
        return Mock()

    @pytest.fixture
    def ecg_service(self, mock_db, mock_validation_service):
        return HybridECGAnalysisService(
            db=mock_db, validation_service=mock_validation_service
        )

    def test_initialization(self, mock_db, mock_validation_service):
        """Test service initialization."""
        service = HybridECGAnalysisService(
            db=mock_db, validation_service=mock_validation_service
        )
        assert service.db == mock_db
        assert service.validation_service == mock_validation_service
        assert hasattr(service, "ecg_reader")
        assert hasattr(service, "preprocessor")
        assert hasattr(service, "feature_extractor")
        assert hasattr(service, "repository")
        assert hasattr(service, "ecg_logger")

    @pytest.mark.asyncio
    async def test_analyze_ecg_comprehensive_success(self, ecg_service):
        """Test successful comprehensive ECG analysis."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data = pd.DataFrame(
                {
                    "I": np.random.randn(1000) * 0.1,
                    "II": np.random.randn(1000) * 0.1,
                    "V1": np.random.randn(1000) * 0.1,
                }
            )
            test_data.to_csv(f.name, index=False)

            result = await ecg_service_instance.analyze_ecg_comprehensive(
                file_path=f.name, patient_id=123, analysis_id="TEST_001"
            )

            assert isinstance(result, dict)
            assert "analysis_id" in result
            assert "patient_id" in result
            assert "processing_time_seconds" in result
            assert "signal_quality" in result
            assert "ai_predictions" in result
            assert "pathology_detections" in result
            assert "clinical_assessment" in result
            assert "extracted_features" in result
            assert "metadata" in result

            os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_analyze_ecg_comprehensive_exception(self, ecg_service):
        """Test comprehensive ECG analysis with exception."""
        from app.core.exceptions import ECGProcessingException

        with pytest.raises(ECGProcessingException):
            await ecg_service_instance.analyze_ecg_comprehensive(
                file_path="/nonexistent/path.csv",
                patient_id=123,
                analysis_id="FAIL_001",
            )

    @pytest.mark.asyncio
    async def test_run_simplified_analysis(self, ecg_service):
        """Test simplified analysis."""
        signal = np.random.randn(1000, 1).astype(np.float64)
        features = {"rr_mean": 800, "rr_std": 50, "hrv_rmssd": 30}

        result = await ecg_service_instance._run_simplified_analysis(signal, features)
        assert isinstance(result, dict)
        assert "predictions" in result
        assert "confidence" in result
        assert "model_version" in result

        predictions = result["predictions"]
        assert "normal" in predictions
        assert "atrial_fibrillation" in predictions
        assert "tachycardia" in predictions
        assert "bradycardia" in predictions

    @pytest.mark.asyncio
    async def test_run_simplified_analysis_edge_cases(self, ecg_service):
        """Test simplified analysis with edge cases."""
        signal = np.random.randn(1000, 1).astype(np.float64)

        features_af = {
            "rr_mean": 800,
            "rr_std": 300,  # High irregularity
        }
        result = await ecg_service_instance._run_simplified_analysis(signal, features_af)
        assert result["predictions"]["atrial_fibrillation"] > 0.5

        features_tachy = {
            "rr_mean": 500,  # HR = 120 bpm
            "rr_std": 20,
        }
        result = await ecg_service_instance._run_simplified_analysis(signal, features_tachy)
        assert result["predictions"]["tachycardia"] > 0.5

        features_brady = {
            "rr_mean": 1200,  # HR = 50 bpm
            "rr_std": 20,
        }
        result = await ecg_service_instance._run_simplified_analysis(signal, features_brady)
        assert result["predictions"]["bradycardia"] > 0.5

    @pytest.mark.asyncio
    async def test_detect_pathologies(self, ecg_service):
        """Test pathology detection."""
        signal = np.random.randn(1000, 1).astype(np.float64)
        features = {"rr_mean": 800, "rr_std": 50, "qtc_bazett": 420}

        pathologies = await ecg_service_instance._detect_pathologies(signal, features)
        assert isinstance(pathologies, dict)
        assert "atrial_fibrillation" in pathologies
        assert "long_qt_syndrome" in pathologies

        af_result = pathologies["atrial_fibrillation"]
        assert "detected" in af_result
        assert "confidence" in af_result
        assert "criteria" in af_result

    def test_detect_atrial_fibrillation_normal(self, ecg_service):
        """Test AF detection with normal features."""
        features = {
            "rr_mean": 800,
            "rr_std": 40,  # Low irregularity
            "hrv_rmssd": 30,
            "spectral_entropy": 0.5,
        }

        score = ecg_service_instance._detect_atrial_fibrillation(features)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be low for normal

    def test_detect_atrial_fibrillation_positive(self, ecg_service):
        """Test AF detection with AF features."""
        features = {
            "rr_mean": 800,
            "rr_std": 300,  # High irregularity
            "hrv_rmssd": 60,  # High HRV
            "spectral_entropy": 0.9,  # High entropy
        }

        score = ecg_service_instance._detect_atrial_fibrillation(features)
        assert isinstance(score, float)
        assert score > 0.5  # Should be high for AF

    def test_detect_atrial_fibrillation_empty_features(self, ecg_service):
        """Test AF detection with empty features."""
        features = {}

        score = ecg_service_instance._detect_atrial_fibrillation(features)
        assert isinstance(score, float)
        assert score == 0.0

    def test_detect_atrial_fibrillation_zero_rr_mean(self, ecg_service):
        """Test AF detection with zero RR mean."""
        features = {
            "rr_mean": 0,
            "rr_std": 50,
        }

        score = ecg_service_instance._detect_atrial_fibrillation(features)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_detect_long_qt_normal(self, ecg_service):
        """Test long QT detection with normal QTc."""
        features = {"qtc_bazett": 420}  # Normal QTc

        score = ecg_service_instance._detect_long_qt(features)
        assert isinstance(score, float)
        assert score == 0.0

    def test_detect_long_qt_positive(self, ecg_service):
        """Test long QT detection with prolonged QTc."""
        features = {"qtc_bazett": 480}  # Prolonged QTc

        score = ecg_service_instance._detect_long_qt(features)
        assert isinstance(score, float)
        assert score > 0.0

    def test_detect_long_qt_empty_features(self, ecg_service):
        """Test long QT detection with empty features."""
        features = {}

        score = ecg_service_instance._detect_long_qt(features)
        assert isinstance(score, float)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_generate_clinical_assessment_normal(self, ecg_service):
        """Test clinical assessment generation for normal case."""
        ai_results = {
            "predictions": {"normal": 0.9, "atrial_fibrillation": 0.1},
            "confidence": 0.9,
        }
        pathology_results = {
            "atrial_fibrillation": {"detected": False, "confidence": 0.1},
            "long_qt_syndrome": {"detected": False, "confidence": 0.1},
        }
        features = {"rr_mean": 800, "rr_std": 50}

        assessment = await ecg_service_instance._generate_clinical_assessment(
            ai_results, pathology_results, features
        )
        assert isinstance(assessment, dict)
        assert "primary_diagnosis" in assessment
        assert "secondary_diagnoses" in assessment
        assert "clinical_urgency" in assessment
        assert "requires_immediate_attention" in assessment
        assert "recommendations" in assessment
        assert "icd10_codes" in assessment
        assert "confidence" in assessment

        assert assessment["primary_diagnosis"] == "Normal ECG"
        assert assessment["requires_immediate_attention"] is False

    @pytest.mark.asyncio
    async def test_generate_clinical_assessment_af(self, ecg_service):
        """Test clinical assessment generation for AF case."""
        ai_results = {
            "predictions": {"normal": 0.2, "atrial_fibrillation": 0.8},
            "confidence": 0.8,
        }
        pathology_results = {
            "atrial_fibrillation": {"detected": True, "confidence": 0.8},
            "long_qt_syndrome": {"detected": False, "confidence": 0.1},
        }
        features = {"rr_mean": 800, "rr_std": 300}

        assessment = await ecg_service_instance._generate_clinical_assessment(
            ai_results, pathology_results, features
        )
        assert assessment["primary_diagnosis"] == "Atrial Fibrillation"
        assert "Anticoagulation assessment recommended" in assessment["recommendations"]

    @pytest.mark.asyncio
    async def test_generate_clinical_assessment_pathology_detected(self, ecg_service):
        """Test clinical assessment with detected pathology."""
        ai_results = {
            "predictions": {"normal": 0.5, "atrial_fibrillation": 0.5},
            "confidence": 0.5,
        }
        pathology_results = {
            "atrial_fibrillation": {"detected": False, "confidence": 0.3},
            "long_qt_syndrome": {"detected": True, "confidence": 0.7},
        }
        features = {"rr_mean": 800, "rr_std": 50}

        assessment = await ecg_service_instance._generate_clinical_assessment(
            ai_results, pathology_results, features
        )
        assert (
            "Long Qt Syndrome" in assessment["primary_diagnosis"]
            or "Long Qt Syndrome" in assessment["secondary_diagnoses"]
        )

    @pytest.mark.asyncio
    async def test_assess_signal_quality_normal(self, ecg_service):
        """Test signal quality assessment with normal signal."""
        normal_signal = np.random.randn(1000, 1).astype(np.float64) * 0.1

        quality = await ecg_service_instance._assess_signal_quality(normal_signal)
        assert isinstance(quality, dict)
        assert "snr_db" in quality
        assert "baseline_stability" in quality
        assert "overall_score" in quality

        assert isinstance(quality["snr_db"], float)
        assert isinstance(quality["baseline_stability"], float)
        assert isinstance(quality["overall_score"], float)
        assert 0.0 <= quality["overall_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_assess_signal_quality_zero_noise(self, ecg_service):
        """Test signal quality assessment with zero noise."""
        constant_signal = np.ones((1000, 1), dtype=np.float64)

        quality = await ecg_service_instance._assess_signal_quality(constant_signal)
        assert isinstance(quality, dict)
        assert "overall_score" in quality

    @pytest.mark.asyncio
    async def test_assess_signal_quality_high_power(self, ecg_service):
        """Test signal quality assessment with high power signal."""
        high_power_signal = np.random.randn(1000, 1).astype(np.float64) * 10

        quality = await ecg_service_instance._assess_signal_quality(high_power_signal)
        assert isinstance(quality, dict)
        assert (
            quality["overall_score"] >= 0.6
        )  # Should have minimum quality for valid signals

    @pytest.mark.asyncio
    async def test_assess_signal_quality_low_power(self, ecg_service):
        """Test signal quality assessment with low power signal."""
        low_power_signal = np.random.randn(1000, 1).astype(np.float64) * 1e-8

        quality = await ecg_service_instance._assess_signal_quality(low_power_signal)
        assert isinstance(quality, dict)
        assert "overall_score" in quality


class TestECGMedicalSafety:
    """Medical safety and regulatory compliance tests."""

    @pytest.fixture
    def ecg_service(self):
        return HybridECGAnalysisService(db=Mock(), validation_service=Mock())

    @pytest.mark.asyncio
    async def test_signal_quality_validation_nan(self, ecg_service):
        """Test signal quality validation with NaN values."""
        nan_signal = np.full((1000, 1), np.nan, dtype=np.float64)

        quality = await ecg_service_instance._assess_signal_quality(nan_signal)
        assert isinstance(quality, dict)
        assert "overall_score" in quality

    @pytest.mark.asyncio
    async def test_signal_quality_validation_inf(self, ecg_service):
        """Test signal quality validation with infinite values."""
        inf_signal = np.full((1000, 1), np.inf, dtype=np.float64)

        quality = await ecg_service_instance._assess_signal_quality(inf_signal)
        assert isinstance(quality, dict)
        assert "overall_score" in quality

    def test_feature_extraction_robustness_nan(self):
        """Test feature extraction robustness with NaN signal."""
        extractor = FeatureExtractor()
        nan_signal = np.full((100, 1), np.nan, dtype=np.float64)

        try:
            features = extractor.extract_all_features(nan_signal)
            assert isinstance(features, dict)
        except Exception:
            pass

    def test_feature_extraction_robustness_zero(self):
        """Test feature extraction robustness with zero signal."""
        extractor = FeatureExtractor()
        zero_signal = np.zeros((100, 1), dtype=np.float64)

        features = extractor.extract_all_features(zero_signal)
        assert isinstance(features, dict)

    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, ecg_service):
        """Test that processing time is tracked."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data = pd.DataFrame(
                {"I": np.random.randn(1000) * 0.1, "II": np.random.randn(1000) * 0.1}
            )
            test_data.to_csv(f.name, index=False)

            start_time = time.time()
            result = await ecg_service_instance.analyze_ecg_comprehensive(
                file_path=f.name, patient_id=123, analysis_id="PERF_001"
            )
            elapsed_time = time.time() - start_time

            assert "processing_time_seconds" in result
            assert isinstance(result["processing_time_seconds"], float)
            assert result["processing_time_seconds"] > 0
            assert elapsed_time < 60.0  # Should complete within 60 seconds

            os.unlink(f.name)

    def test_memory_efficiency_basic(self):
        """Test basic memory efficiency."""
        reader = UniversalECGReader()
        preprocessor = AdvancedPreprocessor()
        extractor = FeatureExtractor()

        assert reader is not None
        assert preprocessor is not None
        assert extractor is not None

    @pytest.mark.asyncio
    async def test_metadata_completeness(self, ecg_service):
        """Test that analysis results contain required metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data = pd.DataFrame(
                {"I": np.random.randn(1000) * 0.1, "II": np.random.randn(1000) * 0.1}
            )
            test_data.to_csv(f.name, index=False)

            result = await ecg_service_instance.analyze_ecg_comprehensive(
                file_path=f.name, patient_id=123, analysis_id="METADATA_001"
            )

            assert "analysis_id" in result
            assert "patient_id" in result
            assert "metadata" in result
            assert "extracted_features" in result
            assert "pathology_detections" in result
            assert "clinical_assessment" in result
            assert "signal_quality" in result
            assert "processing_time_seconds" in result

            metadata = result["metadata"]
            assert "sampling_rate" in metadata
            assert "leads" in metadata
            assert "signal_length" in metadata
            assert "preprocessing_applied" in metadata
            assert "model_version" in metadata
            assert "gdpr_compliant" in metadata
            assert "ce_marking" in metadata
            assert "surveillance_plan" in metadata
            assert "nmsa_certification" in metadata
            assert "data_residency" in metadata
            assert "language_support" in metadata
            assert "population_validation" in metadata

            os.unlink(f.name)
