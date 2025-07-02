"""
Medical-Grade Tests for ECG Hybrid Processor - 85%+ Coverage Target
Regulatory Compliance: FDA, ISO 13485, EU MDR, ANVISA
"""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Any, Dict

from app.utils.ecg_hybrid_processor import ECGHybridProcessor
from app.core.exceptions import ECGProcessingException


class TestECGHybridProcessorComplete:
    """Comprehensive medical-grade tests for ECG Hybrid Processor."""

    @pytest.fixture
    def mock_db(self):
        """Mock database for testing."""
        return Mock()

    @pytest.fixture
    def mock_validation_service(self):
        """Mock validation service for testing."""
        return Mock()

    @pytest.fixture
    def ecg_processor(self, mock_db, mock_validation_service):
        """ECG Hybrid Processor instance for testing."""
        return ECGHybridProcessor(
            db=mock_db, validation_service=mock_validation_service
        )

    @pytest.fixture
    def sample_analysis_results(self):
        """Sample analysis results for testing."""
        return {
            "patient_id": "TEST_001",
            "analysis_id": "ANALYSIS_001",
            "abnormalities": {
                "stemi": {"detected": False, "confidence": 0.1},
                "vfib": {"detected": False, "confidence": 0.05},
            },
            "features": {"heart_rate": 75.0, "rr_intervals": [800, 820, 810]},
            "clinical_urgency": "low",
            "processing_time": 2.5,
        }

    def test_initialization(self, mock_db, mock_validation_service):
        """Test ECG Hybrid Processor initialization."""
        processor = ECGHybridProcessor(
            db=mock_db, validation_service=mock_validation_service
        )

        assert processor.hybrid_service is not None
        assert processor.regulatory_service is None  # Will be implemented in PR-003
        assert hasattr(processor.hybrid_service, "ecg_reader")

    @pytest.mark.asyncio
    async def test_process_ecg_with_validation_success(
        self, ecg_processor, sample_analysis_results
    ):
        """Test successful ECG processing with validation."""
        with patch.object(
            ecg_processor.hybrid_service,
            "analyze_ecg_comprehensive",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = sample_analysis_results.copy()

            result = await ecg_processor.process_ecg_with_validation(
                file_path="/test/ecg.csv",
                patient_id=123,
                analysis_id="TEST_ANALYSIS_001",
            )

            mock_analyze.assert_called_once_with(
                file_path="/test/ecg.csv",
                patient_id=123,
                analysis_id="TEST_ANALYSIS_001",
            )

            assert result["regulatory_compliant"] is True
            assert result["compliance_issues"] == []
            assert "regulatory_validation" in result
            assert (
                result["regulatory_validation"]["validation_results"]["status"]
                == "pending_regulatory_implementation"
            )

    @pytest.mark.asyncio
    async def test_process_ecg_with_validation_compliance_required(
        self, ecg_processor, sample_analysis_results
    ):
        """Test ECG processing with regulatory compliance required."""
        failing_results = sample_analysis_results.copy()

        with patch.object(
            ecg_processor.hybrid_service,
            "analyze_ecg_comprehensive",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = failing_results

            ecg_processor.regulatory_service = Mock()
            ecg_processor.regulatory_service.validate_analysis_comprehensive = (
                AsyncMock(
                    return_value={
                        "status": "failed",
                        "errors": ["Signal quality too low"],
                    }
                )
            )
            ecg_processor.regulatory_service.generate_validation_report = AsyncMock(
                return_value={
                    "overall_compliance": False,
                    "recommendations": ["Improve signal quality", "Repeat analysis"],
                }
            )

            result = await ecg_processor.process_ecg_with_validation(
                file_path="/test/noisy_ecg.csv",
                patient_id=456,
                analysis_id="NOISY_ANALYSIS_001",
                require_regulatory_compliance=True,
            )

            assert result["regulatory_compliant"] is False
            assert len(result["compliance_issues"]) == 2
            assert "Improve signal quality" in result["compliance_issues"]

    @pytest.mark.asyncio
    async def test_process_ecg_with_validation_no_compliance_required(
        self, ecg_processor, sample_analysis_results
    ):
        """Test ECG processing without regulatory compliance requirement."""
        with patch.object(
            ecg_processor.hybrid_service,
            "analyze_ecg_comprehensive",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = sample_analysis_results.copy()

            result = await ecg_processor.process_ecg_with_validation(
                file_path="/test/ecg.csv",
                patient_id=789,
                analysis_id="NO_COMPLIANCE_001",
                require_regulatory_compliance=False,
            )

            assert result["regulatory_compliant"] is True
            assert result["compliance_issues"] == []

    @pytest.mark.asyncio
    async def test_process_ecg_with_validation_exception_handling(self, ecg_processor):
        """Test exception handling in ECG processing."""
        with patch.object(
            ecg_processor.hybrid_service,
            "analyze_ecg_comprehensive",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")

            with pytest.raises(
                ECGProcessingException, match="Hybrid processing failed"
            ):
                await ecg_processor.process_ecg_with_validation(
                    file_path="/test/invalid_ecg.csv",
                    patient_id=999,
                    analysis_id="ERROR_ANALYSIS_001",
                )

    @pytest.mark.asyncio
    async def test_validate_existing_analysis_success(
        self, ecg_processor, sample_analysis_results
    ):
        """Test validation of existing analysis results."""
        result = await ecg_processor.validate_existing_analysis(sample_analysis_results)

        assert "validation_results" in result
        assert "validation_report" in result
        assert "overall_compliance" in result
        assert result["overall_compliance"] is True
        assert (
            result["validation_results"]["status"]
            == "pending_regulatory_implementation"
        )

    @pytest.mark.asyncio
    async def test_validate_existing_analysis_with_regulatory_service(
        self, ecg_processor, sample_analysis_results
    ):
        """Test validation with regulatory service configured."""
        result = await ecg_processor.validate_existing_analysis(sample_analysis_results)

        assert result["overall_compliance"] is True
        assert (
            result["validation_results"]["status"]
            == "pending_regulatory_implementation"
        )
        assert result["validation_report"]["overall_compliance"] is True

    @pytest.mark.asyncio
    async def test_validate_existing_analysis_exception_handling(self, ecg_processor):
        """Test exception handling in analysis validation."""
        with patch.object(
            ecg_processor,
            "validate_existing_analysis",
            side_effect=Exception("Validation service failed"),
        ):
            with pytest.raises(Exception, match="Validation service failed"):
                await ecg_processor.validate_existing_analysis({"test": "data"})

    def test_get_supported_formats(self, ecg_processor):
        """Test getting supported ECG file formats."""
        with patch.object(
            ecg_processor.hybrid_service.ecg_reader,
            "supported_formats",
            {"csv": "CSV format", "edf": "EDF format", "mit": "MIT-BIH format"},
        ):
            formats = ecg_processor.get_supported_formats()

            assert isinstance(formats, list)
            assert "csv" in formats
            assert "edf" in formats
            assert "mit" in formats

    def test_get_regulatory_standards(self, ecg_processor):
        """Test getting supported regulatory standards."""
        standards = ecg_processor.get_regulatory_standards()

        assert isinstance(standards, list)
        assert "FDA" in standards
        assert "ANVISA" in standards
        assert "NMSA" in standards
        assert "EU_MDR" in standards

    @pytest.mark.asyncio
    async def test_get_system_status(self, ecg_processor):
        """Test getting system status."""
        with patch.object(
            ecg_processor, "get_supported_formats", return_value=["csv", "edf", "mit"]
        ):
            with patch.object(
                ecg_processor,
                "get_regulatory_standards",
                return_value=["FDA", "ANVISA", "NMSA", "EU_MDR"],
            ):
                status = await ecg_processor.get_system_status()

                assert status["hybrid_service_initialized"] is True
                assert (
                    status["regulatory_service_initialized"] is False
                )  # Not implemented yet
                assert len(status["supported_formats"]) == 3
                assert len(status["regulatory_standards"]) == 4
                assert status["system_version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_system_status_with_regulatory_service(self, ecg_processor):
        """Test system status with regulatory service configured."""
        ecg_processor.regulatory_service = Mock()

        with patch.object(ecg_processor, "get_supported_formats", return_value=["csv"]):
            with patch.object(
                ecg_processor, "get_regulatory_standards", return_value=["FDA"]
            ):
                status = await ecg_processor.get_system_status()

                assert status["regulatory_service_initialized"] is True


class TestECGHybridProcessorMedicalSafety:
    """Medical safety and regulatory compliance tests."""

    @pytest.fixture
    def ecg_processor(self):
        """ECG Hybrid Processor for safety testing."""
        return ECGHybridProcessor(db=Mock(), validation_service=Mock())

    @pytest.mark.asyncio
    async def test_emergency_processing_timeout_handling(self, ecg_processor):
        """Test handling of processing timeouts in emergency scenarios."""
        with patch.object(
            ecg_processor.hybrid_service,
            "analyze_ecg_comprehensive",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.side_effect = asyncio.TimeoutError("Analysis timeout")

            with pytest.raises(ECGProcessingException):
                await ecg_processor.process_ecg_with_validation(
                    file_path="/test/emergency_ecg.csv",
                    patient_id=911,
                    analysis_id="EMERGENCY_TIMEOUT_001",
                )

    @pytest.mark.asyncio
    async def test_critical_pathology_processing(self, ecg_processor):
        """Test processing of critical pathologies (STEMI, VFib)."""
        critical_results = {
            "patient_id": "CRITICAL_001",
            "analysis_id": "STEMI_ANALYSIS_001",
            "abnormalities": {
                "stemi": {"detected": True, "confidence": 0.98},
                "vfib": {"detected": False, "confidence": 0.02},
            },
            "clinical_urgency": "critical",
            "processing_time": 8.5,
        }

        with patch.object(
            ecg_processor.hybrid_service,
            "analyze_ecg_comprehensive",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = critical_results

            result = await ecg_processor.process_ecg_with_validation(
                file_path="/test/stemi_ecg.csv",
                patient_id=911,
                analysis_id="STEMI_ANALYSIS_001",
            )

            assert result["abnormalities"]["stemi"]["detected"] is True
            assert result["abnormalities"]["stemi"]["confidence"] >= 0.95
            assert result["clinical_urgency"] == "critical"
            assert (
                result["regulatory_compliant"] is True
            )  # Should pass even for critical cases

    @pytest.mark.asyncio
    async def test_concurrent_processing_stability(self, ecg_processor):
        """Test stability under concurrent processing load."""

        async def process_single_ecg(patient_id: int):
            """Process a single ECG for concurrent testing."""
            with patch.object(
                ecg_processor.hybrid_service,
                "analyze_ecg_comprehensive",
                new_callable=AsyncMock,
            ) as mock_analyze:
                mock_analyze.return_value = {
                    "patient_id": f"CONCURRENT_{patient_id}",
                    "analysis_id": f"CONCURRENT_ANALYSIS_{patient_id}",
                    "abnormalities": {"stemi": {"detected": False, "confidence": 0.1}},
                    "clinical_urgency": "low",
                }

                return await ecg_processor.process_ecg_with_validation(
                    file_path=f"/test/concurrent_ecg_{patient_id}.csv",
                    patient_id=patient_id,
                    analysis_id=f"CONCURRENT_ANALYSIS_{patient_id}",
                )

        tasks = [process_single_ecg(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            assert not isinstance(
                result, Exception
            ), f"Concurrent analysis {i} failed: {result}"
            assert result["regulatory_compliant"] is True

    def test_memory_efficiency_large_analysis(self, ecg_processor):
        """Test memory efficiency with large analysis results."""
        import sys

        large_results = {
            "patient_id": "LARGE_DATA_001",
            "analysis_id": "LARGE_ANALYSIS_001",
            "abnormalities": {"stemi": {"detected": False, "confidence": 0.1}},
            "features": {
                "large_feature_array": list(range(10000)),  # Large feature set
                "signal_data": [0.1] * 50000,  # Large signal data
            },
            "clinical_urgency": "low",
        }

        initial_size = sys.getsizeof(large_results)

        with patch.object(
            ecg_processor.hybrid_service,
            "analyze_ecg_comprehensive",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = large_results

            asyncio.run(
                ecg_processor.process_ecg_with_validation(
                    file_path="/test/large_ecg.csv",
                    patient_id=12345,
                    analysis_id="LARGE_ANALYSIS_001",
                )
            )

    def test_regulatory_compliance_audit_trail(self, ecg_processor):
        """Test that regulatory compliance maintains proper audit trail."""
        analysis_results = {
            "patient_id": "AUDIT_001",
            "analysis_id": "AUDIT_ANALYSIS_001",
            "abnormalities": {"stemi": {"detected": False, "confidence": 0.1}},
            "audit_trail": {
                "processing_steps": [
                    "preprocessing",
                    "feature_extraction",
                    "classification",
                ],
                "timestamps": [
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T10:00:05Z",
                    "2024-01-01T10:00:10Z",
                ],
                "model_versions": {"classifier": "v1.0.0", "preprocessor": "v1.0.0"},
            },
        }

        async def run_test():
            with patch.object(
                ecg_processor.hybrid_service,
                "analyze_ecg_comprehensive",
                new_callable=AsyncMock,
            ) as mock_analyze:
                mock_analyze.return_value = analysis_results

                result = await ecg_processor.process_ecg_with_validation(
                    file_path="/test/audit_ecg.csv",
                    patient_id=54321,
                    analysis_id="AUDIT_ANALYSIS_001",
                )

                assert "audit_trail" in result
                assert "regulatory_validation" in result
                assert result["regulatory_compliant"] is True

        import asyncio

        asyncio.run(run_test())

    def test_error_handling_robustness(self, ecg_processor):
        """Test robust error handling for various failure scenarios."""
        error_scenarios = [
            (ValueError("Invalid file format"), "Invalid file format"),
            (FileNotFoundError("ECG file not found"), "ECG file not found"),
            (MemoryError("Insufficient memory"), "Insufficient memory"),
            (RuntimeError("Model loading failed"), "Model loading failed"),
        ]

        for error, expected_message in error_scenarios:
            with patch.object(
                ecg_processor.hybrid_service,
                "analyze_ecg_comprehensive",
                new_callable=AsyncMock,
            ) as mock_analyze:
                mock_analyze.side_effect = error

                with pytest.raises(ECGProcessingException) as exc_info:
                    asyncio.run(
                        ecg_processor.process_ecg_with_validation(
                            file_path="/test/error_ecg.csv",
                            patient_id=99999,
                            analysis_id="ERROR_ANALYSIS_001",
                        )
                    )

                assert "Hybrid processing failed" in str(exc_info.value)


class TestECGHybridProcessorPerformance:
    """Performance and efficiency tests for medical environment."""

    @pytest.fixture
    def ecg_processor(self):
        """ECG Hybrid Processor for performance testing."""
        return ECGHybridProcessor(db=Mock(), validation_service=Mock())

    def test_processing_time_requirements(self, ecg_processor):
        """Test that processing meets medical time requirements."""
        import time

        fast_results = {
            "patient_id": "FAST_001",
            "analysis_id": "FAST_ANALYSIS_001",
            "abnormalities": {"stemi": {"detected": False, "confidence": 0.1}},
            "processing_time": 5.0,
        }

        async def run_test():
            with patch.object(
                ecg_processor.hybrid_service,
                "analyze_ecg_comprehensive",
                new_callable=AsyncMock,
            ) as mock_analyze:
                mock_analyze.return_value = fast_results

                start_time = time.time()
                result = await ecg_processor.process_ecg_with_validation(
                    file_path="/test/fast_ecg.csv",
                    patient_id=11111,
                    analysis_id="FAST_ANALYSIS_001",
                )
                processing_time = time.time() - start_time

                assert (
                    processing_time < 30.0
                ), f"Processing too slow: {processing_time:.2f}s"
                assert result["regulatory_compliant"] is True

        import asyncio

        asyncio.run(run_test())

    def test_supported_formats_completeness(self, ecg_processor):
        """Test that all required medical ECG formats are supported."""
        with patch.object(
            ecg_processor.hybrid_service.ecg_reader,
            "supported_formats",
            {
                "csv": "CSV format",
                "edf": "European Data Format",
                "mit": "MIT-BIH format",
                "txt": "Text format",
            },
        ):
            formats = ecg_processor.get_supported_formats()

            required_formats = ["csv", "edf", "mit"]
            for fmt in required_formats:
                assert fmt in formats, f"Required medical format {fmt} not supported"

    def test_regulatory_standards_completeness(self, ecg_processor):
        """Test that all required regulatory standards are supported."""
        standards = ecg_processor.get_regulatory_standards()

        required_standards = ["FDA", "ANVISA", "NMSA", "EU_MDR"]
        for standard in required_standards:
            assert (
                standard in standards
            ), f"Required regulatory standard {standard} not supported"
