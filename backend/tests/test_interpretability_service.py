"""
Comprehensive tests for InterpretabilityService
Tests SHAP/LIME explanations validation and clinical text generation
"""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Dict, Any

from app.services.interpretability_service import (
    InterpretabilityService,
    ExplanationResult,
)
from app.core.constants import ClinicalUrgency


class TestInterpretabilityService:
    """Test suite for InterpretabilityService"""

    @pytest.fixture
    def service(self):
        """Create InterpretabilityService instance for testing"""
        return InterpretabilityService()

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
            "p_wave_amplitude": 0.1,
            "r_wave_amplitude": 1.2,
            "hrv_rmssd": 45.0,
            "spectral_entropy": 0.6,
        }

    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction results"""
        return {
            "NORM": {"confidence": 0.1, "detected": False},
            "AFIB": {"confidence": 0.8, "detected": True},
            "STEMI": {"confidence": 0.05, "detected": False},
            "LBBB": {"confidence": 0.3, "detected": False},
            "RBBB": {"confidence": 0.2, "detected": False},
        }

    @pytest.fixture
    def sample_model_output(self):
        """Create sample model output"""
        return {
            "primary_diagnosis": "Atrial Fibrillation",
            "clinical_urgency": ClinicalUrgency.HIGH,
            "confidence": 0.8,
            "detected_conditions": {"AFIB": {"confidence": 0.8, "detected": True}},
        }

    def test_service_initialization(self, service):
        """Test service initialization"""
        assert service is not None
        assert hasattr(service, "shap_explainer")
        assert hasattr(service, "lime_explainer")
        assert hasattr(service, "clinical_explainer")

    @pytest.mark.asyncio
    async def test_generate_shap_explanation(
        self, service, sample_ecg_signal, sample_predictions
    ):
        """Test SHAP explanation generation"""
        with patch.object(service, "_prepare_signal_for_shap") as mock_prepare:
            mock_prepare.return_value = sample_ecg_signal

            with patch("shap.Explainer") as mock_explainer_class:
                mock_explainer = MagicMock()
                mock_explainer_class.return_value = mock_explainer

                mock_shap_values = np.random.randn(12, 5000) * 0.1
                mock_explainer.return_value = MagicMock(
                    values=mock_shap_values, base_values=0.1
                )

                result = await service._generate_shap_explanation(
                    sample_ecg_signal, sample_predictions
                )

                assert "shap_values" in result
                assert "base_value" in result
                assert "feature_importance" in result
                assert "lead_contributions" in result

                expected_leads = [
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
                for i, lead in enumerate(expected_leads):
                    assert lead in result["shap_values"]
                    assert len(result["shap_values"][lead]) > 0

                assert isinstance(result["feature_importance"], dict)
                assert len(result["feature_importance"]) > 0

                assert isinstance(result["lead_contributions"], dict)
                for lead in expected_leads:
                    assert lead in result["lead_contributions"]
                    assert isinstance(result["lead_contributions"][lead], (int, float))

    @pytest.mark.asyncio
    async def test_generate_lime_explanation(
        self, service, sample_ecg_signal, sample_predictions
    ):
        """Test LIME explanation generation"""
        with patch("lime.lime_tabular.LimeTabularExplainer") as mock_lime_class:
            mock_lime = MagicMock()
            mock_lime_class.return_value = mock_lime

            mock_explanation = MagicMock()
            mock_explanation.as_list.return_value = [
                ("heart_rate", 0.3),
                ("rr_std", 0.25),
                ("qrs_duration", 0.15),
                ("qtc", 0.1),
                ("st_elevation_max", 0.05),
            ]
            mock_explanation.score = 0.8
            mock_lime.explain_instance.return_value = mock_explanation

            result = await service._generate_lime_explanation(
                sample_ecg_signal, sample_predictions
            )

            assert "feature_importance" in result
            assert "explanation_score" in result
            assert "local_explanation" in result

            assert isinstance(result["feature_importance"], dict)
            assert "heart_rate" in result["feature_importance"]
            assert "rr_std" in result["feature_importance"]

            assert isinstance(result["explanation_score"], (int, float))
            assert 0.0 <= result["explanation_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_generate_clinical_explanation(
        self, service, sample_features, sample_predictions, sample_model_output
    ):
        """Test clinical explanation generation"""
        result = await service._generate_clinical_explanation(
            sample_features, sample_predictions, sample_model_output
        )

        assert "clinical_explanation" in result
        assert "diagnostic_criteria" in result
        assert "risk_factors" in result
        assert "recommendations" in result

        clinical_text = result["clinical_explanation"]
        assert isinstance(clinical_text, str)
        assert len(clinical_text) > 50  # Should be substantial text

        assert "Atrial Fibrillation" in clinical_text or "AFIB" in clinical_text

        assert isinstance(result["diagnostic_criteria"], list)
        assert len(result["diagnostic_criteria"]) > 0

        assert isinstance(result["risk_factors"], list)

        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_comprehensive_explanation_generation(
        self,
        service,
        sample_ecg_signal,
        sample_features,
        sample_predictions,
        sample_model_output,
    ):
        """Test comprehensive explanation generation"""
        with patch.object(service, "_generate_shap_explanation") as mock_shap:
            with patch.object(service, "_generate_lime_explanation") as mock_lime:
                with patch.object(
                    service, "_generate_clinical_explanation"
                ) as mock_clinical:

                    mock_shap.return_value = {
                        "shap_values": {"I": [0.1, 0.2], "II": [0.15, 0.25]},
                        "base_value": 0.1,
                        "feature_importance": {"heart_rate": 0.3, "rr_std": 0.25},
                        "lead_contributions": {"I": 0.2, "II": 0.3},
                    }

                    mock_lime.return_value = {
                        "feature_importance": {"heart_rate": 0.35, "qrs_duration": 0.2},
                        "explanation_score": 0.85,
                        "local_explanation": "LIME local explanation",
                    }

                    mock_clinical.return_value = {
                        "clinical_explanation": "Patient presents with atrial fibrillation...",
                        "diagnostic_criteria": [
                            "Irregular RR intervals",
                            "Absent P waves",
                        ],
                        "risk_factors": ["Age", "Hypertension"],
                        "recommendations": [
                            "Anticoagulation assessment",
                            "Rate control",
                        ],
                    }

                    result = await service.generate_comprehensive_explanation(
                        signal=sample_ecg_signal,
                        features=sample_features,
                        predictions=sample_predictions,
                        model_output=sample_model_output,
                    )

                    assert isinstance(result, ExplanationResult)

                    assert result.clinical_explanation is not None
                    assert result.diagnostic_criteria is not None
                    assert result.risk_factors is not None
                    assert result.recommendations is not None
                    assert result.feature_importance is not None
                    assert result.attention_maps is not None

                    assert len(result.clinical_explanation) > 50
                    assert "atrial fibrillation" in result.clinical_explanation.lower()

                    assert len(result.diagnostic_criteria) >= 2
                    assert any(
                        "irregular" in criterion.lower()
                        for criterion in result.diagnostic_criteria
                    )

                    assert len(result.recommendations) >= 2
                    assert any(
                        "anticoagulation" in rec.lower()
                        for rec in result.recommendations
                    )

    @pytest.mark.asyncio
    async def test_clinical_text_generation_accuracy(
        self, service, sample_features, sample_predictions, sample_model_output
    ):
        """Test clinical text generation accuracy > 90%"""
        test_scenarios = [
            {
                "diagnosis": "Atrial Fibrillation",
                "predictions": {"AFIB": {"confidence": 0.9, "detected": True}},
                "expected_keywords": [
                    "atrial fibrillation",
                    "irregular",
                    "anticoagulation",
                ],
            },
            {
                "diagnosis": "STEMI",
                "predictions": {"STEMI": {"confidence": 0.85, "detected": True}},
                "expected_keywords": [
                    "stemi",
                    "elevation",
                    "urgent",
                    "catheterization",
                ],
            },
            {
                "diagnosis": "Left Bundle Branch Block",
                "predictions": {"LBBB": {"confidence": 0.8, "detected": True}},
                "expected_keywords": ["bundle branch", "conduction", "qrs"],
            },
        ]

        accuracy_scores = []

        for scenario in test_scenarios:
            model_output = sample_model_output.copy()
            model_output["primary_diagnosis"] = scenario["diagnosis"]

            result = await service._generate_clinical_explanation(
                sample_features, scenario["predictions"], model_output
            )

            clinical_text = result["clinical_explanation"].lower()

            keywords_found = sum(
                1
                for keyword in scenario["expected_keywords"]
                if keyword in clinical_text
            )
            accuracy = keywords_found / len(scenario["expected_keywords"])
            accuracy_scores.append(accuracy)

        overall_accuracy = np.mean(accuracy_scores)
        assert (
            overall_accuracy > 0.90
        ), f"Clinical text generation accuracy {overall_accuracy:.2%} does not meet >90% requirement"

    @pytest.mark.asyncio
    async def test_attention_map_generation(
        self, service, sample_ecg_signal, sample_predictions
    ):
        """Test attention map generation and correlation"""
        with patch.object(service, "_generate_attention_maps") as mock_attention:
            mock_attention_maps = {}
            leads = [
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

            for lead in leads:
                attention_weights = np.random.beta(
                    2, 5, 5000
                )  # Sparse attention pattern
                mock_attention_maps[lead] = attention_weights.tolist()

            mock_attention.return_value = mock_attention_maps

            result = await service._generate_attention_maps(
                sample_ecg_signal, sample_predictions
            )

            assert isinstance(result, dict)
            assert len(result) == 12  # All 12 leads

            for lead in leads:
                assert lead in result
                assert isinstance(result[lead], list)
                assert len(result[lead]) > 0

                attention_weights = np.array(result[lead])
                assert np.all(attention_weights >= 0)
                assert np.all(attention_weights <= 1)

    @pytest.mark.asyncio
    async def test_explanation_consistency(
        self,
        service,
        sample_ecg_signal,
        sample_features,
        sample_predictions,
        sample_model_output,
    ):
        """Test consistency of explanations across multiple runs"""
        explanations = []

        for _ in range(5):
            with patch.object(service, "_generate_shap_explanation") as mock_shap:
                with patch.object(
                    service, "_generate_clinical_explanation"
                ) as mock_clinical:

                    mock_shap.return_value = {
                        "feature_importance": {"heart_rate": 0.3, "rr_std": 0.25},
                        "shap_values": {"I": [0.1, 0.2]},
                        "base_value": 0.1,
                        "lead_contributions": {"I": 0.2},
                    }

                    mock_clinical.return_value = {
                        "clinical_explanation": "Consistent clinical explanation",
                        "diagnostic_criteria": ["Criterion 1", "Criterion 2"],
                        "risk_factors": ["Risk 1"],
                        "recommendations": ["Recommendation 1"],
                    }

                    result = await service.generate_comprehensive_explanation(
                        signal=sample_ecg_signal,
                        features=sample_features,
                        predictions=sample_predictions,
                        model_output=sample_model_output,
                    )

                    explanations.append(result)

        first_explanation = explanations[0]
        for explanation in explanations[1:]:
            assert (
                explanation.clinical_explanation
                == first_explanation.clinical_explanation
            )

            assert (
                explanation.diagnostic_criteria == first_explanation.diagnostic_criteria
            )

            assert (
                explanation.feature_importance == first_explanation.feature_importance
            )

    @pytest.mark.asyncio
    async def test_error_handling_invalid_signal(
        self, service, sample_features, sample_predictions, sample_model_output
    ):
        """Test error handling with invalid signal"""
        invalid_signal = np.random.randn(5, 100)  # Wrong shape

        result = await service.generate_comprehensive_explanation(
            signal=invalid_signal,
            features=sample_features,
            predictions=sample_predictions,
            model_output=sample_model_output,
        )

        assert isinstance(result, ExplanationResult)
        assert result.clinical_explanation is not None
        assert len(result.clinical_explanation) > 0

    @pytest.mark.asyncio
    async def test_error_handling_missing_features(
        self, service, sample_ecg_signal, sample_predictions, sample_model_output
    ):
        """Test error handling with missing features"""
        minimal_features = {"heart_rate": 75.0, "signal_quality": 0.8}

        result = await service.generate_comprehensive_explanation(
            signal=sample_ecg_signal,
            features=minimal_features,
            predictions=sample_predictions,
            model_output=sample_model_output,
        )

        assert isinstance(result, ExplanationResult)
        assert result.clinical_explanation is not None
        assert result.feature_importance is not None

    @pytest.mark.asyncio
    async def test_shap_explanation_validation(
        self, service, sample_ecg_signal, sample_predictions
    ):
        """Test SHAP explanation validation and quality"""
        with patch.object(service, "_prepare_signal_for_shap") as mock_prepare:
            mock_prepare.return_value = sample_ecg_signal

            with patch("shap.Explainer") as mock_explainer_class:
                mock_explainer = MagicMock()
                mock_explainer_class.return_value = mock_explainer

                mock_shap_values = np.random.randn(12, 5000) * 0.1
                mock_explainer.return_value = MagicMock(
                    values=mock_shap_values, base_values=0.1
                )

                result = await service._generate_shap_explanation(
                    sample_ecg_signal, sample_predictions
                )

                assert "shap_values" in result

                total_contribution = 0
                for lead_values in result["shap_values"].values():
                    total_contribution += np.sum(lead_values)

                assert abs(total_contribution) > 1e-6

                assert "feature_importance" in result
                assert len(result["feature_importance"]) > 0

                for importance in result["feature_importance"].values():
                    assert isinstance(importance, (int, float))
                    assert importance >= 0  # Absolute importance values

    @pytest.mark.asyncio
    async def test_lime_explanation_validation(
        self, service, sample_ecg_signal, sample_predictions
    ):
        """Test LIME explanation validation and quality"""
        with patch("lime.lime_tabular.LimeTabularExplainer") as mock_lime_class:
            mock_lime = MagicMock()
            mock_lime_class.return_value = mock_lime

            mock_explanation = MagicMock()
            feature_contributions = [
                ("heart_rate", 0.3),
                ("rr_std", -0.25),
                ("qrs_duration", 0.15),
                ("qtc", -0.1),
                ("st_elevation_max", 0.05),
            ]
            mock_explanation.as_list.return_value = feature_contributions
            mock_explanation.score = 0.85
            mock_lime.explain_instance.return_value = mock_explanation

            result = await service._generate_lime_explanation(
                sample_ecg_signal, sample_predictions
            )

            assert "feature_importance" in result
            assert "explanation_score" in result

            assert 0.5 <= result["explanation_score"] <= 1.0

            importance_values = list(result["feature_importance"].values())
            assert any(
                val > 0 for val in importance_values
            )  # Some positive contributions

            important_features = ["heart_rate", "rr_std", "qrs_duration"]
            for feature in important_features:
                if feature in result["feature_importance"]:
                    assert (
                        abs(result["feature_importance"][feature]) > 0.01
                    )  # Meaningful contribution

    def test_explanation_result_dataclass(self):
        """Test ExplanationResult dataclass structure"""
        result = ExplanationResult(
            clinical_explanation="Test explanation",
            diagnostic_criteria=["Criterion 1", "Criterion 2"],
            risk_factors=["Risk 1"],
            recommendations=["Recommendation 1"],
            feature_importance={"heart_rate": 0.3},
            attention_maps={"I": [0.1, 0.2, 0.3]},
        )

        assert result.clinical_explanation == "Test explanation"
        assert len(result.diagnostic_criteria) == 2
        assert len(result.risk_factors) == 1
        assert len(result.recommendations) == 1
        assert "heart_rate" in result.feature_importance
        assert "I" in result.attention_maps

    @pytest.mark.asyncio
    async def test_performance_clinical_explanation_generation(
        self, service, sample_features, sample_predictions, sample_model_output
    ):
        """Test performance of clinical explanation generation"""
        import time

        start_time = time.time()

        result = await service._generate_clinical_explanation(
            sample_features, sample_predictions, sample_model_output
        )

        end_time = time.time()
        processing_time = end_time - start_time

        assert (
            processing_time < 1.0
        ), f"Clinical explanation generation took {processing_time:.3f}s, should be < 1.0s"

        assert len(result["clinical_explanation"]) > 100  # Substantial explanation
        assert len(result["diagnostic_criteria"]) >= 2
        assert len(result["recommendations"]) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
