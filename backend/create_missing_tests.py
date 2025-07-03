#!/usr/bin/env python3
"""
Create Missing Tests
Cria testes para módulos com baixa cobertura
"""

import json
from pathlib import Path
from typing import Dict, List


class MissingTestsCreator:
    """Create tests for modules with low coverage"""
    
    def __init__(self):
        self.backend_path = Path(__file__).parent
        self.tests_path = self.backend_path / "tests"
        self.tests_path.mkdir(exist_ok=True)
        
    def run(self):
        """Run test creation"""
        print("=" * 60)
        print("CRIAÇÃO DE TESTES PARA AUMENTAR COBERTURA")
        print("=" * 60)
        
        # Create comprehensive tests
        self.create_ecg_service_tests()
        self.create_memory_monitor_tests()
        self.create_interpretability_tests()
        self.create_report_generator_tests()
        self.create_ecg_classifier_tests()
        self.create_core_tests()
        
        print("\n✅ Testes criados com sucesso!")
        print("\nPróximo passo:")
        print("python run_coverage_test.py")
    
    def create_ecg_service_tests(self):
        """Create comprehensive tests for ECGAnalysisService"""
        print("\n1. Criando testes para ECGAnalysisService...")
        
        test_content = '''"""
Comprehensive tests for ECGAnalysisService
"""

import pytest
import numpy as np
from datetime import datetime
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch

from app.services.ecg_service import ECGAnalysisService
from app.schemas.ecg_analysis import (
    ECGAnalysisCreate, ProcessingStatus, ClinicalUrgency, FileInfo
)
from app.core.exceptions import ECGProcessingException, ResourceNotFoundException


class TestECGAnalysisService:
    """Test ECGAnalysisService"""
    
    @pytest.fixture
    def service(self):
        """Create service instance"""
        return ECGAnalysisService()
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session"""
        db = AsyncMock()
        return db
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user"""
        user = Mock()
        user.id = uuid4()
        return user
    
    @pytest.fixture
    def analysis_data(self):
        """Create test analysis data"""
        return ECGAnalysisCreate(
            patient_id=uuid4(),
            file_path="/test/ecg.csv",
            file_info=FileInfo(
                file_name="test.csv",
                file_size=1024,
                file_hash="testhash"
            )
        )
    
    @pytest.mark.asyncio
    async def test_create_analysis_success(self, service, mock_db, mock_user, analysis_data):
        """Test successful analysis creation"""
        # Mock patient exists
        mock_patient = Mock()
        mock_db.scalar = AsyncMock(return_value=mock_patient)
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()
        
        # Test
        result = await service.create_analysis(mock_db, analysis_data, mock_user)
        
        # Verify
        assert result is not None
        assert mock_db.commit.called
    
    def test_preprocess_signal(self, service):
        """Test signal preprocessing"""
        # Create test signal
        signal = np.random.randn(5000)  # 10 seconds at 500Hz
        
        # Test
        processed = service._preprocess_signal(signal)
        
        # Verify
        assert processed is not None
        assert len(processed) == len(signal)
        assert np.abs(np.mean(processed)) < 0.1  # Near zero mean
    
    def test_extract_measurements(self, service):
        """Test measurement extraction"""
        # Create test signal
        signal = np.sin(2 * np.pi * 1.2 * np.linspace(0, 10, 5000))  # 72 bpm
        
        # Test
        measurements = service._extract_measurements(signal)
        
        # Verify
        assert "heart_rate" in measurements
        assert "pr_interval" in measurements
        assert "qrs_duration" in measurements
        assert "qt_interval" in measurements
        assert "qtc_interval" in measurements
    
    def test_generate_annotations(self, service):
        """Test annotation generation"""
        # Test data
        signal = np.random.randn(5000)
        measurements = {
            "heart_rate": 45,  # Bradycardia
            "qtc_interval": 480,  # Prolonged
            "pr_interval": 220  # First degree AV block
        }
        
        # Test
        annotations = service._generate_annotations(signal, measurements)
        
        # Verify
        assert len(annotations) > 0
        assert any(ann["type"] == "bradycardia" for ann in annotations)
        assert any(ann["type"] == "prolonged_qt" for ann in annotations)
        assert any(ann["type"] == "av_block" for ann in annotations)
    
    def test_assess_clinical_urgency(self, service):
        """Test clinical urgency assessment"""
        # Critical pathologies
        critical_pathologies = [
            {"condition": "ventricular_fibrillation", "severity": "critical"}
        ]
        measurements = {"heart_rate": 180}
        
        urgency = service._assess_clinical_urgency(critical_pathologies, measurements)
        assert urgency == ClinicalUrgency.CRITICAL
        
        # Normal
        normal_pathologies = []
        normal_measurements = {"heart_rate": 75}
        
        urgency = service._assess_clinical_urgency(normal_pathologies, normal_measurements)
        assert urgency == ClinicalUrgency.NORMAL
    
    def test_generate_medical_recommendations(self, service):
        """Test medical recommendations generation"""
        # Test data
        pathologies = [
            {"condition": "atrial_fibrillation", "severity": "high"}
        ]
        urgency = ClinicalUrgency.HIGH
        measurements = {"heart_rate": 150}
        
        # Test
        recommendations = service._generate_medical_recommendations(
            pathologies, urgency, measurements
        )
        
        # Verify
        assert len(recommendations) > 0
        assert any("cardiology" in rec.lower() for rec in recommendations)
    
    def test_calculate_file_info(self, service):
        """Test file info calculation"""
        file_path = "test.csv"
        content = b"test content"
        
        file_info = service.calculate_file_info(file_path, content)
        
        assert file_info["file_name"] == "test.csv"
        assert file_info["file_size"] == len(content)
        assert "file_hash" in file_info
    
    def test_get_normal_range(self, service):
        """Test normal range retrieval"""
        ranges = ["heart_rate", "pr_interval", "qrs_duration", "qt_interval"]
        
        for param in ranges:
            range_info = service.get_normal_range(param)
            assert "min" in range_info
            assert "max" in range_info
            assert "unit" in range_info
    
    def test_assess_quality_issues(self, service):
        """Test signal quality assessment"""
        # Flat line signal
        flat_signal = np.zeros(5000)
        issues = service.assess_quality_issues(flat_signal)
        assert len(issues) > 0
        assert any("flat line" in issue.lower() for issue in issues)
        
        # Good signal
        good_signal = np.sin(2 * np.pi * 1 * np.linspace(0, 10, 5000))
        issues = service.assess_quality_issues(good_signal)
        assert len(issues) == 0
    
    def test_generate_clinical_interpretation(self, service):
        """Test clinical interpretation generation"""
        measurements = {
            "heart_rate": 55,
            "pr_interval": 210,
            "qtc_interval": 420
        }
        pathologies = [
            {"condition": "atrial_fibrillation"}
        ]
        
        interpretation = service.generate_clinical_interpretation(measurements, pathologies)
        
        assert "Bradycardia" in interpretation
        assert "fibrillation" in interpretation
    
    def test_detect_r_peaks(self, service):
        """Test R peak detection"""
        # Create synthetic ECG with known peaks
        fs = 500
        duration = 10
        t = np.linspace(0, duration, duration * fs)
        
        # Add R peaks at regular intervals (60 bpm)
        ecg = np.zeros_like(t)
        peak_interval = fs  # 1 peak per second
        for i in range(0, len(ecg), peak_interval):
            if i < len(ecg):
                ecg[i] = 1.0
        
        # Add noise
        ecg += np.random.normal(0, 0.01, len(ecg))
        
        # Detect peaks
        peaks = service._detect_r_peaks(ecg)
        
        # Should detect approximately 10 peaks
        assert len(peaks) >= 8
        assert len(peaks) <= 12
    
    def test_calculate_heart_rate(self, service):
        """Test heart rate calculation"""
        # R peaks at 1 second intervals (60 bpm)
        r_peaks = np.array([0, 500, 1000, 1500, 2000])
        
        hr = service._calculate_heart_rate(r_peaks)
        
        assert 58 <= hr <= 62  # Allow small variance
    
    def test_calculate_qtc(self, service):
        """Test QTc calculation"""
        qt_interval = 400  # ms
        heart_rate = 60  # bpm
        
        qtc = service._calculate_qtc(qt_interval, heart_rate)
        
        # QTc should be same as QT at 60 bpm
        assert abs(qtc - qt_interval) < 5
        
        # Test with faster heart rate
        heart_rate = 120
        qtc = service._calculate_qtc(qt_interval, heart_rate)
        
        # QTc should be longer than QT
        assert qtc > qt_interval
'''
        
        test_file = self.tests_path / "test_ecg_service_comprehensive.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print(f"   ✅ Criado: {test_file.name}")
    
    def create_memory_monitor_tests(self):
        """Create tests for MemoryMonitor"""
        print("\n2. Criando testes para MemoryMonitor...")
        
        test_content = '''"""
Tests for MemoryMonitor
"""

import pytest
import time
from unittest.mock import Mock, patch

from app.utils.memory_monitor import (
    MemoryMonitor, MemoryStats, MemoryAlert, get_memory_monitor
)


class TestMemoryMonitor:
    """Test MemoryMonitor"""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor instance"""
        return MemoryMonitor(check_interval=1, memory_threshold=80.0)
    
    def test_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.check_interval == 1
        assert monitor.memory_threshold == 80.0
        assert monitor.process_threshold == 70.0
        assert not monitor._monitoring
    
    def test_get_memory_stats(self, monitor):
        """Test memory stats retrieval"""
        stats = monitor.get_memory_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.total_memory > 0
        assert stats.available_memory > 0
        assert 0 <= stats.memory_percent <= 100
        assert stats.process_memory > 0
    
    def test_get_memory_info(self, monitor):
        """Test legacy memory info method"""
        info = monitor.get_memory_info()
        
        assert "timestamp" in info
        assert "system" in info
        assert "process" in info
        assert "swap" in info
        assert info["system"]["total"] > 0
    
    def test_check_memory_threshold(self, monitor):
        """Test memory threshold checking"""
        # Create stats with high memory usage
        mock_stats = MemoryStats(
            timestamp=time.time(),
            total_memory=8_000_000_000,
            available_memory=1_000_000_000,
            used_memory=7_000_000_000,
            memory_percent=87.5,  # Above threshold
            process_memory=1_000_000_000,
            process_percent=12.5,
            swap_total=4_000_000_000,
            swap_used=1_000_000_000,
            swap_percent=25.0
        )
        
        alerts = monitor.check_memory_threshold(mock_stats)
        
        assert len(alerts) > 0
        assert any(alert.alert_type == "system_memory" for alert in alerts)
        assert alerts[0].severity in ["low", "medium", "high", "critical"]
    
    def test_optimize_memory(self, monitor):
        """Test memory optimization"""
        result = monitor.optimize_memory()
        
        assert "timestamp" in result
        assert "garbage_collected" in result
        assert "freed_system_memory" in result
        assert "freed_process_memory" in result
        assert result["garbage_collected"] >= 0
    
    def test_add_remove_callback(self, monitor):
        """Test callback management"""
        callback = Mock()
        
        # Add callback
        monitor.add_alert_callback(callback)
        assert callback in monitor._callbacks
        
        # Remove callback
        monitor.remove_alert_callback(callback)
        assert callback not in monitor._callbacks
    
    def test_stats_history(self, monitor):
        """Test stats history management"""
        # Add some stats
        for _ in range(3):
            monitor._stats_history.append(monitor.get_memory_stats())
            time.sleep(0.1)
        
        # Get all history
        history = monitor.get_stats_history()
        assert len(history) == 3
        
        # Get recent history (should return all since they're recent)
        recent = monitor.get_stats_history(minutes=1)
        assert len(recent) == 3
    
    def test_alerts_history(self, monitor):
        """Test alerts history filtering"""
        # Add mock alerts
        monitor._alerts_history = [
            MemoryAlert(
                timestamp=time.time(),
                alert_type="system_memory",
                threshold=80.0,
                current_value=85.0,
                message="Test",
                severity="high"
            ),
            MemoryAlert(
                timestamp=time.time(),
                alert_type="process_memory",
                threshold=70.0,
                current_value=75.0,
                message="Test",
                severity="medium"
            )
        ]
        
        # Get all alerts
        all_alerts = monitor.get_alerts_history()
        assert len(all_alerts) == 2
        
        # Filter by severity
        high_alerts = monitor.get_alerts_history(severity="high")
        assert len(high_alerts) == 1
        
        # Filter by type
        system_alerts = monitor.get_alerts_history(alert_type="system_memory")
        assert len(system_alerts) == 1
    
    def test_get_summary(self, monitor):
        """Test summary generation"""
        summary = monitor.get_summary()
        
        assert "monitoring_active" in summary
        assert "current" in summary
        assert "average" in summary
        assert "thresholds" in summary
        assert "alerts" in summary
        assert summary["current"]["system_percent"] >= 0
    
    def test_estimate_memory_for_ecg(self, monitor):
        """Test ECG memory estimation"""
        # 10 seconds, 500Hz, 12 leads
        estimate = monitor.estimate_memory_for_ecg(
            duration_seconds=10,
            sampling_rate=500,
            channels=12
        )
        
        assert estimate["samples"] == 5000
        assert estimate["raw_size_mb"] > 0
        assert estimate["estimated_total_mb"] > estimate["raw_size_mb"]
        assert "can_process" in estimate
        assert "recommendation" in estimate
    
    def test_context_manager(self, monitor):
        """Test context manager functionality"""
        with monitor as m:
            assert m._monitoring
        
        assert not monitor._monitoring
    
    def test_monitor_operation_context(self, monitor):
        """Test operation monitoring context"""
        with monitor.monitor_operation("test_operation"):
            # Simulate some work
            data = [i ** 2 for i in range(1000)]
        
        # Operation should complete without error
        assert True
    
    def test_global_monitor(self):
        """Test global monitor instance"""
        monitor1 = get_memory_monitor()
        monitor2 = get_memory_monitor()
        
        # Should return same instance
        assert monitor1 is monitor2
        assert monitor1._monitoring  # Should be started
'''
        
        test_file = self.tests_path / "test_memory_monitor_comprehensive.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print(f"   ✅ Criado: {test_file.name}")
    
    def create_interpretability_tests(self):
        """Create tests for InterpretabilityService"""
        print("\n3. Criando testes para InterpretabilityService...")
        
        test_content = '''"""
Tests for InterpretabilityService
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from app.services.interpretability_service import (
    InterpretabilityService, ExplanationResult
)


class TestInterpretabilityService:
    """Test InterpretabilityService"""
    
    @pytest.fixture
    def service(self):
        """Create service instance"""
        return InterpretabilityService()
    
    @pytest.fixture
    def test_data(self):
        """Create test data"""
        return {
            "analysis_id": "test-123",
            "signal": np.random.randn(5000),
            "features": {
                "heart_rate": 75,
                "pr_interval": 160,
                "qrs_duration": 90,
                "qt_interval": 400,
                "qtc_interval": 400,
                "rms": 0.5,
                "variance": 0.25
            },
            "predictions": {
                "normal_sinus_rhythm": 0.85,
                "atrial_fibrillation": 0.10,
                "long_qt_syndrome": 0.05
            },
            "measurements": {
                "heart_rate": 75,
                "qtc_interval": 400,
                "r_peaks": [100, 600, 1100, 1600]
            }
        }
    
    def test_initialization(self, service):
        """Test service initialization"""
        assert service.diagnostic_criteria is not None
        assert service.feature_descriptions is not None
        assert isinstance(service.diagnostic_criteria, dict)
    
    def test_generate_comprehensive_explanation(self, service, test_data):
        """Test comprehensive explanation generation"""
        result = service.generate_comprehensive_explanation(
            test_data["analysis_id"],
            test_data["signal"],
            test_data["features"],
            test_data["predictions"],
            test_data["measurements"]
        )
        
        assert isinstance(result, ExplanationResult)
        assert result.analysis_id == test_data["analysis_id"]
        assert result.feature_importance is not None
        assert result.clinical_text != ""
        assert result.primary_findings is not None
        assert result.recommendations is not None
    
    def test_generate_shap_explanation(self, service, test_data):
        """Test SHAP explanation generation"""
        explanation = service._generate_shap_explanation(
            test_data["features"],
            test_data["predictions"]
        )
        
        assert "importance" in explanation
        assert len(explanation["importance"]) > 0
        assert all(0 <= v <= 1 for v in explanation["importance"].values())
    
    def test_generate_lime_explanation(self, service, test_data):
        """Test LIME explanation generation"""
        predict_fn = service._create_predict_function(test_data["predictions"])
        
        explanation = service._generate_lime_explanation(
            test_data["signal"],
            test_data["features"],
            predict_fn
        )
        
        assert "weights" in explanation
        assert isinstance(explanation["weights"], dict)
    
    def test_generate_clinical_explanation(self, service, test_data):
        """Test clinical explanation generation"""
        clinical = service._generate_clinical_explanation(
            test_data["features"],
            test_data["predictions"],
            test_data["measurements"]
        )
        
        assert "text" in clinical
        assert "findings" in clinical
        assert "evidence" in clinical
        assert len(clinical["text"]) > 0
    
    def test_generate_attention_maps(self, service, test_data):
        """Test attention map generation"""
        maps = service._generate_attention_maps(
            test_data["signal"],
            test_data["features"]
        )
        
        assert isinstance(maps, dict)
        # May contain various attention maps
    
    def test_identify_risk_factors(self, service, test_data):
        """Test risk factor identification"""
        feature_importance = {
            "heart_rate": 0.3,
            "qtc_interval": 0.4,
            "pr_interval": 0.2
        }
        
        risks = service._identify_risk_factors(
            test_data["features"],
            test_data["predictions"],
            feature_importance
        )
        
        assert isinstance(risks, list)
        # Should identify risks based on predictions
    
    def test_generate_recommendations(self, service, test_data):
        """Test recommendation generation"""
        risks = [
            {"severity": "high", "type": "condition", "name": "atrial fibrillation"}
        ]
        
        recommendations = service._generate_recommendations(
            risks,
            test_data["predictions"],
            test_data["measurements"]
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(r, str) for r in recommendations)
    
    def test_calculate_confidence_scores(self, service, test_data):
        """Test confidence score calculation"""
        feature_importance = {"heart_rate": 0.5, "qtc_interval": 0.3}
        
        confidence = service._calculate_confidence_scores(
            test_data["predictions"],
            feature_importance
        )
        
        assert "overall" in confidence
        assert "feature_reliability" in confidence
        assert "diagnostic" in confidence
        assert all(0 <= v <= 1 for v in confidence.values())
    
    def test_reference_diagnostic_criteria(self, service, test_data):
        """Test diagnostic criteria reference"""
        predictions = {"atrial_fibrillation": 0.8}
        measurements = {"heart_rate": 120, "qtc_interval": 420}
        
        criteria = service._reference_diagnostic_criteria(
            predictions,
            measurements
        )
        
        assert isinstance(criteria, list)
        # Should include criteria for high-probability conditions
'''
        
        test_file = self.tests_path / "test_interpretability_comprehensive.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print(f"   ✅ Criado: {test_file.name}")
    
    def create_report_generator_tests(self):
        """Create tests for ReportGenerator"""
        print("\n4. Criando testes para ReportGenerator...")
        
        test_content = '''"""
Tests for ReportGenerator
"""

import pytest
from unittest.mock import Mock, patch
import base64

from app.utils.report_generator import ReportGenerator


class TestReportGenerator:
    """Test ReportGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance"""
        return ReportGenerator()
    
    @pytest.fixture
    def analysis_data(self):
        """Create test analysis data"""
        return {
            "id": "test-123",
            "patient": {
                "id": "patient-456",
                "name": "Test Patient"
            },
            "status": "completed",
            "clinical_urgency": "normal",
            "quality_score": 95.0,
            "confidence_score": 88.5,
            "measurements": {
                "heart_rate": 72,
                "pr_interval": 160,
                "qrs_duration": 90,
                "qt_interval": 400,
                "qtc_interval": 410
            },
            "pathologies": [
                {
                    "condition": "normal_sinus_rhythm",
                    "probability": 0.95,
                    "severity": "normal",
                    "confidence": "high"
                }
            ],
            "recommendations": [
                "Regular follow-up recommended"
            ],
            "clinical_context": {
                "age": 45,
                "gender": "M"
            }
        }
    
    def test_initialization(self, generator):
        """Test generator initialization"""
        assert generator.template_dir is not None
        assert generator.styles is not None or not generator.REPORTLAB_AVAILABLE
    
    def test_generate_json_report(self, generator, analysis_data):
        """Test JSON report generation"""
        report = generator.generate(
            analysis_data,
            include_images=False,
            format="json"
        )
        
        assert report["format"] == "json"
        assert "content" in report
        assert "filename" in report
        assert report["filename"].endswith(".json")
    
    def test_generate_html_report(self, generator, analysis_data):
        """Test HTML report generation"""
        report = generator.generate(
            analysis_data,
            include_images=False,
            format="html"
        )
        
        assert report["format"] == "html"
        assert "content" in report
        assert "<html>" in report["content"]
        assert "ECG Analysis Report" in report["content"]
        assert report["filename"].endswith(".html")
    
    @pytest.mark.skipif(
        not ReportGenerator.REPORTLAB_AVAILABLE,
        reason="ReportLab not installed"
    )
    def test_generate_pdf_report(self, generator, analysis_data):
        """Test PDF report generation"""
        report = generator.generate(
            analysis_data,
            include_images=False,
            format="pdf"
        )
        
        assert report["format"] == "pdf"
        assert "content" in report
        # Content should be base64 encoded
        assert base64.b64decode(report["content"])
        assert report["filename"].endswith(".pdf")
    
    def test_get_normal_range(self, generator):
        """Test normal range retrieval"""
        params = ["heart_rate", "pr_interval", "qrs_duration"]
        
        for param in params:
            range_str = generator._get_normal_range(param)
            assert isinstance(range_str, str)
            assert len(range_str) > 0
    
    def test_get_unit(self, generator):
        """Test unit retrieval"""
        units_map = {
            "heart_rate": "bpm",
            "pr_interval": "ms",
            "qt_interval": "ms"
        }
        
        for param, expected_unit in units_map.items():
            unit = generator._get_unit(param)
            assert unit == expected_unit
    
    def test_check_status(self, generator):
        """Test parameter status checking"""
        # Normal heart rate
        status = generator._check_status("heart_rate", 75)
        assert status == "Normal"
        
        # Low heart rate
        status = generator._check_status("heart_rate", 45)
        assert status == "Low"
        
        # High heart rate
        status = generator._check_status("heart_rate", 110)
        assert status == "High"
    
    def test_unsupported_format(self, gen