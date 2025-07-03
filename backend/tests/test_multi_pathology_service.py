"""
Comprehensive tests for MultiPathologyService
Tests hierarchical pathology detection with 71 SCP-ECG conditions
"""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
import numpy as np
from unittest.mock import Mock, patch
import asyncio
from typing import Dict, Any

from app.services.multi_pathology_service import MultiPathologyService
from app.core.scp_ecg_conditions import SCP_ECG_CONDITIONS, SCPCategory
from app.core.constants import ClinicalUrgency


class TestMultiPathologyService:
    """Test suite for MultiPathologyService"""

    @pytest.fixture
    def service(self):
        """Create MultiPathologyService instance for testing"""
        return MultiPathologyService()

    @pytest.fixture
    def sample_ecg_signal(self):
        """Create sample ECG signal for testing"""
        return np.random.randn(12, 5000) * 0.5

    @pytest.fixture
    def sample_features(self):
        """Create sample ECG features for testing"""
        return {
            "heart_rate": 75.0,
            "rr_mean": 800.0,
            "rr_std": 50.0,
            "qrs_duration": 100.0,
            "pr_interval": 160.0,
            "qt_interval": 400.0,
            "qtc": 420.0,
            "qrs_axis": 60.0,
            "st_elevation_max": 0.0,
            "st_depression_max": 0.0,
            "signal_quality": 0.9,
        }

    def test_service_initialization(self, service):
        """Test service initialization"""
        assert service is not None
        assert hasattr(service, "scp_conditions")
        assert len(service.scp_conditions) == 71
        assert hasattr(service, "condition_categories")

    @pytest.mark.asyncio
    async def test_level1_normal_vs_abnormal_normal(
        self, service, sample_ecg_signal, sample_features
    ):
        """Test Level 1: Normal vs Abnormal detection - Normal case"""
        normal_features = sample_features.copy()
        normal_features.update(
            {
                "heart_rate": 75.0,
                "rr_std": 30.0,  # Low variability
                "qrs_duration": 90.0,  # Normal QRS
                "qtc": 420.0,  # Normal QTc
                "st_elevation_max": 0.0,
                "st_depression_max": 0.0,
            }
        )

        result = await service._level1_normal_vs_abnormal(
            sample_ecg_signal, normal_features
        )

        assert "is_normal" in result
        assert "confidence" in result
        assert "npv_score" in result
        assert isinstance(result["is_normal"], bool)
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["npv_score"] <= 1.0

        assert result["is_normal"] is True
        assert result["confidence"] > 0.8  # High confidence for normal
        assert result["npv_score"] > 0.99  # NPV > 99% requirement

    @pytest.mark.asyncio
    async def test_level1_normal_vs_abnormal_abnormal(
        self, service, sample_ecg_signal, sample_features
    ):
        """Test Level 1: Normal vs Abnormal detection - Abnormal case"""
        abnormal_features = sample_features.copy()
        abnormal_features.update(
            {
                "heart_rate": 150.0,  # Tachycardia
                "rr_std": 200.0,  # High variability (AF)
                "qrs_duration": 120.0,  # Wide QRS
                "qtc": 480.0,  # Prolonged QTc
                "st_elevation_max": 0.2,  # ST elevation
            }
        )

        result = await service._level1_normal_vs_abnormal(
            sample_ecg_signal, abnormal_features
        )

        assert "is_normal" in result
        assert "confidence" in result
        assert "npv_score" in result

        assert result["is_normal"] is False
        assert result["confidence"] > 0.7  # High confidence for abnormal

    @pytest.mark.asyncio
    async def test_level2_category_classification(
        self, service, sample_ecg_signal, sample_features
    ):
        """Test Level 2: Category classification"""
        arrhythmia_features = sample_features.copy()
        arrhythmia_features.update(
            {
                "heart_rate": 150.0,
                "rr_std": 200.0,  # High RR variability
            }
        )

        result = await service._level2_category_classification(
            sample_ecg_signal, arrhythmia_features
        )

        assert "detected_categories" in result
        assert "primary_category" in result
        assert "confidence_scores" in result

        assert SCPCategory.ARRHYTHMIA in result["detected_categories"]
        assert result["primary_category"] == SCPCategory.ARRHYTHMIA
        assert result["confidence_scores"][SCPCategory.ARRHYTHMIA] > 0.7

    @pytest.mark.asyncio
    async def test_level3_specific_diagnosis_afib(
        self, service, sample_ecg_signal, sample_features
    ):
        """Test Level 3: Specific diagnosis - Atrial Fibrillation"""
        afib_features = sample_features.copy()
        afib_features.update(
            {
                "heart_rate": 120.0,
                "rr_std": 250.0,  # Very high RR variability
                "hrv_rmssd": 80.0,  # High HRV
                "spectral_entropy": 0.9,  # High entropy
            }
        )

        target_categories = [SCPCategory.ARRHYTHMIA]

        result = await service._level3_specific_diagnosis(
            sample_ecg_signal, afib_features, target_categories
        )

        assert "detected_conditions" in result
        assert "primary_diagnosis" in result
        assert "confidence_scores" in result

        afib_conditions = [
            code
            for code in result["detected_conditions"]
            if "AFIB" in code or "ATRIAL_FIBRILLATION" in code
        ]
        assert len(afib_conditions) > 0

        afib_confidence = max(
            [result["confidence_scores"].get(code, 0) for code in afib_conditions]
        )
        assert afib_confidence > 0.95  # Sensitivity > 95% requirement

    @pytest.mark.asyncio
    async def test_level3_specific_diagnosis_stemi(
        self, service, sample_ecg_signal, sample_features
    ):
        """Test Level 3: Specific diagnosis - STEMI (2x sensitivity requirement)"""
        stemi_features = sample_features.copy()
        stemi_features.update(
            {
                "st_elevation_max": 0.3,  # Significant ST elevation
                "st_elevation_v1": 0.25,
                "st_elevation_v2": 0.35,
                "st_elevation_v3": 0.30,
                "q_wave_depth": 0.15,  # Pathological Q waves
                "troponin_equivalent": 0.8,  # Simulated biomarker
            }
        )

        target_categories = [SCPCategory.ISCHEMIC_CHANGES]

        result = await service._level3_specific_diagnosis(
            sample_ecg_signal, stemi_features, target_categories
        )

        assert "detected_conditions" in result

        stemi_conditions = [
            code
            for code in result["detected_conditions"]
            if "STEMI" in code or "ST_ELEVATION" in code
        ]

        if stemi_conditions:
            stemi_confidence = max(
                [result["confidence_scores"].get(code, 0) for code in stemi_conditions]
            )
            assert stemi_confidence > 0.90  # Very high sensitivity for STEMI

    @pytest.mark.asyncio
    async def test_hierarchical_analysis_complete(
        self, service, sample_ecg_signal, sample_features
    ):
        """Test complete hierarchical analysis"""
        result = await service.analyze_hierarchical(
            signal=sample_ecg_signal,
            features=sample_features,
            preprocessing_quality=0.9,
        )

        assert "level_completed" in result
        assert "primary_diagnosis" in result
        assert "secondary_diagnoses" in result
        assert "clinical_urgency" in result
        assert "requires_immediate_attention" in result
        assert "detected_conditions" in result
        assert "confidence" in result
        assert "recommendations" in result
        assert "icd10_codes" in result

        assert result["level_completed"] >= 1
        assert result["level_completed"] <= 3

        assert result["clinical_urgency"] in [
            ClinicalUrgency.LOW,
            ClinicalUrgency.HIGH,
            ClinicalUrgency.CRITICAL,
        ]

        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_hierarchical_analysis_low_quality(
        self, service, sample_ecg_signal, sample_features
    ):
        """Test hierarchical analysis with low quality signal"""
        result = await service.analyze_hierarchical(
            signal=sample_ecg_signal,
            features=sample_features,
            preprocessing_quality=0.3,  # Low quality
        )

        assert result["level_completed"] >= 1
        assert "primary_diagnosis" in result

        if result["level_completed"] == 1:
            assert result["confidence"] < 0.8  # Lower confidence for low quality

    @pytest.mark.asyncio
    async def test_performance_metrics_npv(self, service):
        """Test NPV > 99% requirement for normal vs abnormal"""
        normal_cases = []

        for i in range(100):
            normal_features = {
                "heart_rate": np.random.uniform(60, 100),
                "rr_mean": np.random.uniform(600, 1000),
                "rr_std": np.random.uniform(20, 50),
                "qrs_duration": np.random.uniform(80, 110),
                "pr_interval": np.random.uniform(120, 200),
                "qt_interval": np.random.uniform(350, 450),
                "qtc": np.random.uniform(380, 450),
                "qrs_axis": np.random.uniform(-30, 90),
                "st_elevation_max": np.random.uniform(-0.05, 0.05),
                "st_depression_max": np.random.uniform(-0.05, 0.05),
                "signal_quality": np.random.uniform(0.8, 1.0),
            }

            signal = np.random.randn(12, 5000) * 0.3  # Normal amplitude

            result = await service._level1_normal_vs_abnormal(signal, normal_features)
            normal_cases.append(result)

        true_negatives = sum(1 for case in normal_cases if case["is_normal"])
        npv = true_negatives / len(normal_cases)

        assert npv > 0.99, f"NPV {npv:.3f} does not meet >99% requirement"

    @pytest.mark.asyncio
    async def test_performance_metrics_sensitivity_arrhythmias(self, service):
        """Test sensitivity > 95% for arrhythmias"""
        arrhythmia_cases = []

        for i in range(50):
            arrhythmia_features = {
                "heart_rate": np.random.uniform(100, 180),
                "rr_mean": np.random.uniform(300, 600),
                "rr_std": np.random.uniform(150, 300),  # High variability
                "qrs_duration": np.random.uniform(80, 120),
                "pr_interval": np.random.uniform(120, 200),
                "qt_interval": np.random.uniform(300, 400),
                "qtc": np.random.uniform(400, 500),
                "qrs_axis": np.random.uniform(-30, 90),
                "st_elevation_max": np.random.uniform(-0.1, 0.1),
                "st_depression_max": np.random.uniform(-0.1, 0.1),
                "signal_quality": np.random.uniform(0.7, 1.0),
                "hrv_rmssd": np.random.uniform(60, 120),
                "spectral_entropy": np.random.uniform(0.7, 1.0),
            }

            signal = np.random.randn(12, 5000) * 0.5

            result = await service._level2_category_classification(
                signal, arrhythmia_features
            )
            arrhythmia_cases.append(result)

        detected_arrhythmias = sum(
            1
            for case in arrhythmia_cases
            if SCPCategory.ARRHYTHMIA in case["detected_categories"]
        )
        sensitivity = detected_arrhythmias / len(arrhythmia_cases)

        assert (
            sensitivity > 0.95
        ), f"Arrhythmia sensitivity {sensitivity:.3f} does not meet >95% requirement"

    @pytest.mark.asyncio
    async def test_performance_metrics_specificity_conduction(self, service):
        """Test specificity > 95% for conduction disorders"""
        normal_cases = []

        for i in range(50):
            normal_features = {
                "heart_rate": np.random.uniform(60, 100),
                "rr_mean": np.random.uniform(600, 1000),
                "rr_std": np.random.uniform(20, 50),
                "qrs_duration": np.random.uniform(80, 110),  # Normal QRS
                "pr_interval": np.random.uniform(120, 200),  # Normal PR
                "qt_interval": np.random.uniform(350, 450),
                "qtc": np.random.uniform(380, 450),
                "qrs_axis": np.random.uniform(-30, 90),
                "st_elevation_max": np.random.uniform(-0.05, 0.05),
                "st_depression_max": np.random.uniform(-0.05, 0.05),
                "signal_quality": np.random.uniform(0.8, 1.0),
            }

            signal = np.random.randn(12, 5000) * 0.3

            result = await service._level2_category_classification(
                signal, normal_features
            )
            normal_cases.append(result)

        false_positives = sum(
            1
            for case in normal_cases
            if SCPCategory.CONDUCTION_ABNORMALITIES in case["detected_categories"]
        )
        specificity = (len(normal_cases) - false_positives) / len(normal_cases)

        assert (
            specificity > 0.95
        ), f"Conduction disorder specificity {specificity:.3f} does not meet >95% requirement"

    def test_scp_conditions_coverage(self, service):
        """Test that all 71 SCP-ECG conditions are covered"""
        assert len(service.scp_conditions) == 71

        for condition_code, condition in service.scp_conditions.items():
            assert hasattr(condition, "code")
            assert hasattr(condition, "description")
            assert hasattr(condition, "category")
            assert hasattr(condition, "clinical_urgency")
            assert condition.code == condition_code

    def test_condition_categories_mapping(self, service):
        """Test condition categories mapping"""
        expected_categories = {
            SCPCategory.ARRHYTHMIA,
            SCPCategory.CONDUCTION_ABNORMALITIES,
            SCPCategory.ISCHEMIC_CHANGES,
            SCPCategory.STRUCTURAL_ABNORMALITIES,
            SCPCategory.NORMAL,
        }

        found_categories = set()
        for conditions in service.condition_categories.values():
            found_categories.update(conditions)

        assert expected_categories.issubset(found_categories)

    @pytest.mark.asyncio
    async def test_error_handling_invalid_signal(self, service, sample_features):
        """Test error handling with invalid signal"""
        invalid_signal = np.random.randn(5, 100)  # Too short, wrong shape

        result = await service.analyze_hierarchical(
            signal=invalid_signal, features=sample_features, preprocessing_quality=0.8
        )

        assert "primary_diagnosis" in result
        assert "level_completed" in result
        assert result["level_completed"] >= 1

    @pytest.mark.asyncio
    async def test_error_handling_missing_features(self, service, sample_ecg_signal):
        """Test error handling with missing features"""
        minimal_features = {"heart_rate": 75.0, "signal_quality": 0.8}

        result = await service.analyze_hierarchical(
            signal=sample_ecg_signal,
            features=minimal_features,
            preprocessing_quality=0.8,
        )

        assert "primary_diagnosis" in result
        assert "level_completed" in result

    @pytest.mark.asyncio
    async def test_clinical_urgency_assignment(
        self, service, sample_ecg_signal, sample_features
    ):
        """Test clinical urgency assignment"""
        critical_features = sample_features.copy()
        critical_features.update(
            {
                "st_elevation_max": 0.4,  # Massive ST elevation
                "heart_rate": 45.0,  # Bradycardia
            }
        )

        result = await service.analyze_hierarchical(
            signal=sample_ecg_signal,
            features=critical_features,
            preprocessing_quality=0.9,
        )

        assert result["clinical_urgency"] in [
            ClinicalUrgency.HIGH,
            ClinicalUrgency.CRITICAL,
        ]

        if result["clinical_urgency"] == ClinicalUrgency.CRITICAL:
            assert result["requires_immediate_attention"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
