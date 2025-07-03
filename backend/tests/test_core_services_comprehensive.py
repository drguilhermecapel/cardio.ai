import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

torch_mock = MagicMock()
torch_mock.nn = MagicMock()
torch_mock.optim = MagicMock()
torch_mock.utils = MagicMock()
torch_mock.utils.data = MagicMock()
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available = MagicMock(return_value=False)

sys.modules["torch"] = torch_mock
sys.modules["torch.nn"] = torch_mock.nn
sys.modules["torch.optim"] = torch_mock.optim
sys.modules["torch.utils"] = torch_mock.utils
sys.modules["torch.utils.data"] = torch_mock.utils.data
sys.modules["torch.cuda"] = torch_mock.cuda
sys.modules["onnxruntime"] = MagicMock()

sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.ensemble"] = MagicMock()
sys.modules["sklearn.preprocessing"] = MagicMock()
sys.modules["shap"] = MagicMock()
sys.modules["lime"] = MagicMock()
sys.modules["lime.lime_tabular"] = MagicMock()

from app.services.advanced_ml_service import AdvancedMLService
from app.services.hybrid_ecg_service import HybridECGAnalysisService
from app.services.multi_pathology_service import MultiPathologyService
from app.services.interpretability_service import (
    InterpretabilityService,
    ExplanationResult,
)


class TestAdvancedMLServiceComprehensive:
    """Comprehensive test coverage for AdvancedMLService"""

    @pytest.fixture
    def service(self):
        return AdvancedMLService()

    @pytest.fixture
    def sample_ecg_data(self):
        return np.random.randn(12, 5000)

    def test_service_initialization(self, service):
        """Test service initialization"""
        assert service is not None
        assert hasattr(service, "models")

    @pytest.mark.asyncio
    async def test_predict_comprehensive_success(self, service, sample_ecg_data):
        """Test comprehensive prediction success"""
        with patch.object(service, "_load_models") as mock_load:
            mock_load.return_value = True

            with patch.object(service, "_extract_features") as mock_features:
                mock_features.return_value = {"heart_rate": 72, "qrs_duration": 100}

                with patch.object(service, "_ensemble_predict") as mock_predict:
                    mock_predict.return_value = {
                        "predictions": {"NORMAL": 0.85, "AFIB": 0.15},
                        "confidence": 0.85,
                    }

                    result = await service.predict_comprehensive(sample_ecg_data)
                    assert "predictions" in result
                    assert "confidence" in result

    def test_load_models_success(self, service):
        """Test model loading success"""
        with patch("onnxruntime.InferenceSession") as mock_session:
            mock_session.return_value = MagicMock()

            result = service._load_models()
            assert result is True

    def test_extract_features_success(self, service, sample_ecg_data):
        """Test feature extraction success"""
        result = service._extract_features(sample_ecg_data, 500)
        assert isinstance(result, dict)
        assert "heart_rate" in result

    def test_ensemble_predict_success(self, service):
        """Test ensemble prediction success"""
        features = {"heart_rate": 72, "qrs_duration": 100}

        with patch.object(service, "models") as mock_models:
            mock_model = MagicMock()
            mock_model.run.return_value = [np.array([[0.8, 0.2]])]
            mock_models = {"model1": mock_model}
            service.models = mock_models

            result = service._ensemble_predict(features)
            assert "predictions" in result
            assert "confidence" in result


class TestHybridECGAnalysisServiceComprehensive:
    """Comprehensive test coverage for HybridECGAnalysisService"""

    @pytest.fixture
    def service(self):
        return HybridECGAnalysisService()

    @pytest.fixture
    def sample_ecg_data(self):
        return np.random.randn(12, 5000)

    def test_service_initialization(self, service):
        """Test service initialization"""
        assert service is not None
        assert hasattr(service, "universal_reader")
        assert hasattr(service, "advanced_preprocessor")

    @pytest.mark.asyncio
    async def test_analyze_ecg_comprehensive_success(self, service, sample_ecg_data):
        """Test comprehensive ECG analysis success"""
        with patch.object(service, "_read_ecg_file") as mock_read:
            mock_read.return_value = sample_ecg_data

            with patch.object(service, "_preprocess_signal") as mock_preprocess:
                mock_preprocess.return_value = sample_ecg_data

                with patch.object(service, "_analyze_with_ml_models") as mock_analyze:
                    mock_analyze.return_value = {
                        "predictions": {"NORMAL": 0.8},
                        "confidence": 0.8,
                    }

                    result = await service.analyze_ecg_comprehensive("/fake/path.csv")
                    assert "predictions" in result
                    assert "confidence" in result

    def test_read_ecg_file_csv(self, service):
        """Test reading CSV ECG file"""
        with patch("pandas.read_csv") as mock_read:
            mock_df = Mock()
            mock_df.values = np.random.randn(5000, 12)
            mock_read.return_value = mock_df

            result = service._read_ecg_file("/fake/path.csv")
            assert isinstance(result, np.ndarray)

    def test_preprocess_signal_success(self, service, sample_ecg_data):
        """Test signal preprocessing success"""
        result = service._preprocess_signal(sample_ecg_data, 500)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_ecg_data.shape

    def test_analyze_with_ml_models_success(self, service, sample_ecg_data):
        """Test ML model analysis success"""
        with patch.object(service, "ml_service") as mock_ml:
            mock_ml.predict_comprehensive = AsyncMock(
                return_value={"predictions": {"NORMAL": 0.8}, "confidence": 0.8}
            )

            result = service._analyze_with_ml_models(sample_ecg_data)
            assert "predictions" in result


class TestMultiPathologyServiceComprehensive:
    """Comprehensive test coverage for MultiPathologyService"""

    @pytest.fixture
    def service(self):
        return MultiPathologyService()

    @pytest.fixture
    def sample_ecg_data(self):
        return np.random.randn(12, 5000)

    def test_service_initialization(self, service):
        """Test service initialization"""
        assert service is not None
        assert hasattr(service, "scp_conditions")

    @pytest.mark.asyncio
    async def test_analyze_pathologies_success(self, service, sample_ecg_data):
        """Test pathology analysis success"""
        with patch.object(service, "_extract_features") as mock_features:
            mock_features.return_value = {"heart_rate": 72}

            with patch.object(service, "_classify_conditions") as mock_classify:
                mock_classify.return_value = {"NORMAL": 0.8, "AFIB": 0.2}

                result = await service.analyze_pathologies(sample_ecg_data)
                assert isinstance(result, dict)
                assert "NORMAL" in result

    def test_extract_features_success(self, service, sample_ecg_data):
        """Test feature extraction success"""
        result = service._extract_features(sample_ecg_data, 500)
        assert isinstance(result, dict)
        assert "heart_rate" in result

    def test_classify_conditions_success(self, service):
        """Test condition classification success"""
        features = {"heart_rate": 72, "qrs_duration": 100}

        result = service._classify_conditions(features)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_analyze_hierarchical_success(self, service):
        """Test hierarchical analysis success"""
        predictions = {"NORMAL": 0.8, "AFIB": 0.2}

        result = service.analyze_hierarchical(predictions)
        assert isinstance(result, dict)
        assert "primary_category" in result


class TestInterpretabilityServiceComprehensive:
    """Comprehensive test coverage for InterpretabilityService"""

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

    def test_initialize_feature_names_success(self, service):
        """Test feature names initialization success"""
        result = service._initialize_feature_names()

        assert isinstance(result, list)
        assert len(result) > 0
        assert "heart_rate" in result
