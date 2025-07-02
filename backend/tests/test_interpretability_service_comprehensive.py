import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from app.services.interpretability_service import (
    InterpretabilityService,
    ExplanationResult,
)

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"


class TestInterpretabilityServiceComprehensive:
    """Comprehensive test coverage for InterpretabilityService to reach 80% coverage"""

    @pytest.fixture
    def service(self):
        return InterpretabilityService()

    @pytest.fixture
    def sample_signal(self):
        return np.random.randn(12, 5000)

    @pytest.fixture
    def sample_features(self):
        return {"heart_rate": 72, "qrs_duration": 100, "pr_interval": 160}

    @pytest.fixture
    def sample_predictions(self):
        return {"NORMAL": 0.8, "AFIB": 0.2}

    def test_service_initialization(self, service):
        """Test service initialization"""
        assert service is not None
        assert hasattr(service, "lead_names")
        assert hasattr(service, "feature_names")
        assert len(service.lead_names) == 12

    @pytest.mark.asyncio
    async def test_generate_comprehensive_explanation_success(
        self, service, sample_signal, sample_features, sample_predictions
    ):
        """Test comprehensive explanation generation success"""
        model_output = {"confidence": 0.8}

        result = await service.generate_comprehensive_explanation(
            sample_signal, sample_features, sample_predictions, model_output
        )

        assert isinstance(result, ExplanationResult)
        assert result.primary_diagnosis is not None
        assert result.confidence >= 0

    @pytest.mark.asyncio
    async def test_generate_shap_explanation_success(
        self, service, sample_signal, sample_features, sample_predictions
    ):
        """Test SHAP explanation generation success"""
        model_output = {"confidence": 0.8}

        with patch.object(service, "shap_explainer") as mock_explainer:
            mock_explainer.shap_values.return_value = np.random.randn(100)

            result = await service._generate_shap_explanation(
                sample_signal, sample_features, sample_predictions, model_output
            )

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_generate_lime_explanation_success(
        self, service, sample_signal, sample_features, sample_predictions
    ):
        """Test LIME explanation generation success"""

        result = await service._generate_lime_explanation(
            sample_signal, sample_features, sample_predictions
        )

        assert isinstance(result, dict)
        assert "local_explanation" in result

    @pytest.mark.asyncio
    async def test_generate_clinical_explanation_success(
        self, service, sample_features, sample_predictions
    ):
        """Test clinical explanation generation success"""
        primary_diagnosis = "NORMAL"
        shap_explanation = {"lead_importance": {"I": 0.5, "II": 0.3}}

        result = await service._generate_clinical_explanation(
            primary_diagnosis, sample_features, sample_predictions, shap_explanation
        )

        assert isinstance(result, dict)
        assert "primary_diagnosis" in result

    @pytest.mark.asyncio
    async def test_generate_attention_maps_success(
        self, service, sample_signal, sample_predictions
    ):
        """Test attention maps generation success"""
        shap_explanation = {"lead_importance": {"I": 0.5, "II": 0.3}}

        result = await service._generate_attention_maps(
            sample_signal, sample_predictions, shap_explanation
        )

        assert isinstance(result, dict)
        assert len(result) <= len(service.lead_names)

    def test_extract_feature_importance_success(self, service, sample_features):
        """Test feature importance extraction success"""
        shap_explanation = {
            "shap_values": [0.5, 0.3, 0.2],
            "feature_names": ["heart_rate", "qrs_duration", "pr_interval"],
        }
        lime_explanation = {
            "local_explanation": [("heart_rate", 0.5), ("qrs_duration", 0.3)]
        }

        result = service._extract_feature_importance(shap_explanation, lime_explanation)

        assert isinstance(result, dict)

    def test_reference_diagnostic_criteria_success(self, service, sample_features):
        """Test diagnostic criteria reference success"""
        result = service._reference_diagnostic_criteria("STEMI", sample_features)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_identify_risk_factors_success(self, service, sample_features):
        """Test risk factors identification success"""
        result = service._identify_risk_factors("AFIB", sample_features)

        assert isinstance(result, list)

    def test_generate_recommendations_success(self, service, sample_features):
        """Test recommendations generation success"""
        result = service._generate_recommendations("STEMI", sample_features)

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_comprehensive_explanation_empty_predictions(
        self, service, sample_signal, sample_features
    ):
        """Test explanation generation with empty predictions"""
        empty_predictions = {}
        model_output = {"confidence": 0.0}

        result = await service.generate_comprehensive_explanation(
            sample_signal, sample_features, empty_predictions, model_output
        )

        assert isinstance(result, ExplanationResult)
        assert result.primary_diagnosis == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_generate_comprehensive_explanation_error_handling(
        self, service, sample_signal, sample_features, sample_predictions
    ):
        """Test explanation generation error handling"""
        model_output = {"confidence": 0.8}

        with patch.object(service, "_generate_shap_explanation") as mock_shap:
            mock_shap.side_effect = Exception("SHAP error")

            result = await service.generate_comprehensive_explanation(
                sample_signal, sample_features, sample_predictions, model_output
            )

            assert isinstance(result, ExplanationResult)
            assert result.primary_diagnosis is not None

    def test_initialize_feature_names_success(self, service):
        """Test feature names initialization success"""
        result = service._initialize_feature_names()

        assert isinstance(result, list)
        assert len(result) > 0
        assert "heart_rate" in result

    @pytest.mark.asyncio
    async def test_clinical_explanation_afib(self, service, sample_features):
        """Test clinical explanation for AFIB"""
        predictions = {"AFIB": 0.9, "NORMAL": 0.1}
        primary_diagnosis = "AFIB"
        shap_explanation = {"lead_importance": {"I": 0.5, "II": 0.3}}

        result = await service._generate_clinical_explanation(
            primary_diagnosis, sample_features, predictions, shap_explanation
        )

        assert "Atrial fibrillation" in result.get("description", "")

    @pytest.mark.asyncio
    async def test_clinical_explanation_stemi(self, service, sample_features):
        """Test clinical explanation for STEMI"""
        predictions = {"STEMI": 0.95, "NORMAL": 0.05}
        primary_diagnosis = "STEMI"
        shap_explanation = {"lead_importance": {"I": 0.5, "II": 0.3}}

        result = await service._generate_clinical_explanation(
            primary_diagnosis, sample_features, predictions, shap_explanation
        )

        assert "ST-elevation myocardial infarction" in result.get("description", "")
