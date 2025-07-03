"""
Testes para aumentar a cobertura do ML Model Service de 79% para >80%
Foco em casos extremos, exceções e fluxos alternativos
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from app.services.ml_model_service import MLModelService
from app.core.exceptions import (
    MLModelException,
    ModelNotLoadedException,
    InvalidInputException,
    PredictionException,
)
from app.models.ecg_analysis import ECGAnalysis
from app.schemas.ecg_analysis import ECGFeatures


class TestMLModelServiceCoverage:
    """Testes focados em aumentar cobertura do ML Model Service"""

    @pytest.fixture
    def ml_service(self):
        """Fixture para criar instância do serviço ML"""
        service = MLModelService()
        service.model_loaded = False
        service.models = {}
        return service

    @pytest.fixture
    def sample_ecg_data(self):
        """Dados de ECG simulados para testes"""
        return {
            "signal": np.random.randn(5000).tolist(),
            "sampling_rate": 360,
            "leads": ["II"],
            "duration": 13.89,
            "patient_age": 45,
            "patient_sex": "M",
        }

    @pytest.fixture
    def sample_features(self):
        """Features extraídas simuladas"""
        return ECGFeatures(
            heart_rate=72.5,
            pr_interval=160,
            qrs_duration=90,
            qt_interval=380,
            qtc_interval=395,
            p_wave_amplitude=0.15,
            r_wave_amplitude=1.2,
            t_wave_amplitude=0.3,
            st_segment_level=0.0,
            rr_intervals=[830, 825, 835, 820],
            hrv_sdnn=45.2,
            hrv_rmssd=38.5,
            signal_quality_score=0.92,
        )

    @pytest.mark.asyncio
    async def test_load_model_success(self, ml_service):
        """Teste de carregamento bem-sucedido do modelo"""
        with patch("app.services.ml_model_service.joblib.load") as mock_load:
            mock_model = Mock()
            mock_model.predict_proba = Mock(return_value=np.array([[0.1, 0.9]]))
            mock_load.return_value = mock_model

            with patch("pathlib.Path.exists", return_value=True):
                await ml_service.load_model("cardiac_classifier")

                assert ml_service.model_loaded is True
                assert "cardiac_classifier" in ml_service.models
                mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_file_not_found(self, ml_service):
        """Teste quando arquivo do modelo não existe"""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(MLModelException, match="Model file not found"):
                await ml_service.load_model("nonexistent_model")

    @pytest.mark.asyncio
    async def test_load_model_invalid_format(self, ml_service):
        """Teste quando modelo tem formato inválido"""
        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "app.services.ml_model_service.joblib.load",
                side_effect=Exception("Invalid model format"),
            ):
                with pytest.raises(MLModelException, match="Failed to load model"):
                    await ml_service.load_model("corrupt_model")

    @pytest.mark.asyncio
    async def test_analyze_ecg_model_not_loaded(self, ml_service, sample_ecg_data):
        """Teste de análise quando modelo não está carregado"""
        ml_service.model_loaded = False

        with pytest.raises(ModelNotLoadedException, match="Model not loaded"):
            await ml_service.analyze_ecg(sample_ecg_data, "test_patient_123")

    @pytest.mark.asyncio
    async def test_analyze_ecg_invalid_input_data(self, ml_service):
        """Teste com dados de entrada inválidos"""
        ml_service.model_loaded = True

        # Teste com dados vazios
        with pytest.raises(InvalidInputException, match="Invalid ECG data"):
            await ml_service.analyze_ecg({}, "test_patient")

        # Teste com sinal inválido
        invalid_data = {"signal": None, "sampling_rate": 360}
        with pytest.raises(InvalidInputException, match="Invalid signal data"):
            await ml_service.analyze_ecg(invalid_data, "test_patient")

        # Teste com taxa de amostragem inválida
        invalid_data = {"signal": [1, 2, 3], "sampling_rate": 0}
        with pytest.raises(InvalidInputException, match="Invalid sampling rate"):
            await ml_service.analyze_ecg(invalid_data, "test_patient")

    @pytest.mark.asyncio
    async def test_analyze_ecg_preprocessing_failure(self, ml_service, sample_ecg_data):
        """Teste quando pré-processamento falha"""
        ml_service.model_loaded = True

        with patch.object(
            ml_service, "_preprocess_ecg", side_effect=Exception("Preprocessing failed")
        ):
            with pytest.raises(MLModelException, match="ECG preprocessing failed"):
                await ml_service.analyze_ecg(sample_ecg_data, "test_patient")

    @pytest.mark.asyncio
    async def test_analyze_ecg_feature_extraction_failure(
        self, ml_service, sample_ecg_data
    ):
        """Teste quando extração de features falha"""
        ml_service.model_loaded = True

        with patch.object(
            ml_service, "_preprocess_ecg", return_value=np.array([1, 2, 3])
        ):
            with patch.object(
                ml_service,
                "_extract_features",
                side_effect=Exception("Feature extraction failed"),
            ):
                with pytest.raises(MLModelException, match="Feature extraction failed"):
                    await ml_service.analyze_ecg(sample_ecg_data, "test_patient")

    @pytest.mark.asyncio
    async def test_analyze_ecg_prediction_failure(self, ml_service, sample_ecg_data):
        """Teste quando predição do modelo falha"""
        ml_service.model_loaded = True
        ml_service.models["cardiac_classifier"] = Mock()

        with patch.object(
            ml_service, "_preprocess_ecg", return_value=np.array([1, 2, 3])
        ):
            with patch.object(ml_service, "_extract_features", return_value=Mock()):
                ml_service.models["cardiac_classifier"].predict_proba = Mock(
                    side_effect=Exception("Model prediction failed")
                )

                with pytest.raises(PredictionException, match="Prediction failed"):
                    await ml_service.analyze_ecg(sample_ecg_data, "test_patient")

    @pytest.mark.asyncio
    async def test_analyze_ecg_edge_cases(self, ml_service):
        """Teste casos extremos de análise ECG"""
        ml_service.model_loaded = True

        # ECG muito curto
        short_ecg = {"signal": [1, 2], "sampling_rate": 360}  # Apenas 2 pontos
        with pytest.raises(InvalidInputException, match="Signal too short"):
            await ml_service.analyze_ecg(short_ecg, "test_patient")

        # ECG com valores NaN
        nan_ecg = {"signal": [1, 2, np.nan, 4, 5] * 100, "sampling_rate": 360}
        with pytest.raises(
            InvalidInputException, match="Signal contains invalid values"
        ):
            await ml_service.analyze_ecg(nan_ecg, "test_patient")

        # ECG com valores infinitos
        inf_ecg = {"signal": [1, 2, np.inf, 4, 5] * 100, "sampling_rate": 360}
        with pytest.raises(
            InvalidInputException, match="Signal contains invalid values"
        ):
            await ml_service.analyze_ecg(inf_ecg, "test_patient")

    @pytest.mark.asyncio
    async def test_predict_batch_success(self, ml_service, sample_features):
        """Teste de predição em lote bem-sucedida"""
        ml_service.model_loaded = True
        mock_model = Mock()
        mock_model.predict_proba = Mock(
            return_value=np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        )
        ml_service.models["cardiac_classifier"] = mock_model

        features_batch = [sample_features] * 3
        results = await ml_service.predict_batch(features_batch)

        assert len(results) == 3
        assert all("predictions" in r for r in results)
        assert all("confidence_scores" in r for r in results)
        mock_model.predict_proba.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_batch_empty_input(self, ml_service):
        """Teste de predição em lote com entrada vazia"""
        ml_service.model_loaded = True

        results = await ml_service.predict_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_predict_batch_partial_failure(self, ml_service, sample_features):
        """Teste quando algumas predições falham no lote"""
        ml_service.model_loaded = True
        mock_model = Mock()

        # Simula falha na segunda predição
        mock_model.predict_proba = Mock(
            side_effect=[
                np.array([[0.1, 0.9]]),
                Exception("Prediction failed"),
                np.array([[0.3, 0.7]]),
            ]
        )
        ml_service.models["cardiac_classifier"] = mock_model

        features_batch = [sample_features] * 3

        with patch.object(ml_service, "_handle_batch_prediction_error"):
            results = await ml_service.predict_batch(
                features_batch, fail_on_error=False
            )

            assert len(results) == 3
            assert results[1] is None  # Segunda predição falhou

    def test_get_model_info(self, ml_service):
        """Teste obtenção de informações do modelo"""
        ml_service.models = {
            "cardiac_classifier": Mock(
                version="1.0.0", training_date="2024-01-01", accuracy=0.95
            )
        }

        info = ml_service.get_model_info("cardiac_classifier")
        assert info["version"] == "1.0.0"
        assert info["training_date"] == "2024-01-01"
        assert info["accuracy"] == 0.95
        assert info["loaded"] is True

    def test_get_model_info_not_loaded(self, ml_service):
        """Teste informações de modelo não carregado"""
        info = ml_service.get_model_info("nonexistent_model")
        assert info["loaded"] is False
        assert info["error"] == "Model not found"

    @pytest.mark.asyncio
    async def test_validate_model_performance(self, ml_service):
        """Teste validação de performance do modelo"""
        ml_service.model_loaded = True
        mock_model = Mock()
        mock_model.score = Mock(return_value=0.92)
        ml_service.models["cardiac_classifier"] = mock_model

        validation_data = Mock()
        validation_labels = Mock()

        performance = await ml_service.validate_model_performance(
            "cardiac_classifier", validation_data, validation_labels
        )

        assert performance["accuracy"] == 0.92
        assert performance["status"] == "valid"
        mock_model.score.assert_called_once_with(validation_data, validation_labels)

    @pytest.mark.asyncio
    async def test_update_model_success(self, ml_service):
        """Teste atualização bem-sucedida do modelo"""
        with patch("app.services.ml_model_service.download_model") as mock_download:
            with patch.object(ml_service, "load_model") as mock_load:
                mock_download.return_value = "/tmp/new_model.pkl"

                result = await ml_service.update_model(
                    "cardiac_classifier",
                    "https://models.cardioai.com/v2/cardiac_classifier.pkl",
                )

                assert result["success"] is True
                assert result["message"] == "Model updated successfully"
                mock_load.assert_called_once_with("cardiac_classifier")

    @pytest.mark.asyncio
    async def test_update_model_download_failure(self, ml_service):
        """Teste falha no download durante atualização do modelo"""
        with patch(
            "app.services.ml_model_service.download_model",
            side_effect=Exception("Download failed"),
        ):

            result = await ml_service.update_model(
                "cardiac_classifier", "https://invalid-url.com/model.pkl"
            )

            assert result["success"] is False
            assert "Download failed" in result["error"]

    def test_cleanup_resources(self, ml_service):
        """Teste limpeza de recursos do modelo"""
        ml_service.models = {"model1": Mock(), "model2": Mock()}
        ml_service.model_loaded = True

        ml_service.cleanup_resources()

        assert len(ml_service.models) == 0
        assert ml_service.model_loaded is False

    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, ml_service, sample_features):
        """Teste predições concorrentes para verificar thread safety"""
        ml_service.model_loaded = True
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.1, 0.9]]))
        ml_service.models["cardiac_classifier"] = mock_model

        import asyncio

        # Simula 10 predições concorrentes
        tasks = []
        for _ in range(10):
            task = ml_service.predict_single(sample_features)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r is not None for r in results)
        assert mock_model.predict_proba.call_count == 10

    def test_model_cache_management(self, ml_service):
        """Teste gerenciamento de cache de modelos"""
        # Adiciona múltiplos modelos
        ml_service.models = {f"model_{i}": Mock(size_mb=100) for i in range(5)}

        # Verifica limite de cache
        ml_service._enforce_cache_limit(max_models=3)

        assert len(ml_service.models) == 3

    @pytest.mark.asyncio
    async def test_model_warm_up(self, ml_service):
        """Teste aquecimento do modelo após carregamento"""
        with patch.object(ml_service, "load_model") as mock_load:
            with patch.object(ml_service, "_warm_up_model") as mock_warm_up:
                await ml_service.initialize_model("cardiac_classifier", warm_up=True)

                mock_load.assert_called_once()
                mock_warm_up.assert_called_once()

    def test_get_supported_conditions(self, ml_service):
        """Teste obtenção de condições suportadas pelo modelo"""
        ml_service.supported_conditions = [
            "atrial_fibrillation",
            "ventricular_tachycardia",
            "bradycardia",
            "normal_sinus_rhythm",
        ]

        conditions = ml_service.get_supported_conditions()
        assert len(conditions) == 4
        assert "atrial_fibrillation" in conditions

    @pytest.mark.asyncio
    async def test_explain_prediction(self, ml_service, sample_features):
        """Teste explicação de predições (SHAP/LIME integration)"""
        ml_service.model_loaded = True
        ml_service.explainer_enabled = True

        with patch.object(ml_service, "_generate_explanation") as mock_explain:
            mock_explain.return_value = {
                "feature_importance": {
                    "heart_rate": 0.25,
                    "qt_interval": 0.20,
                    "qrs_duration": 0.15,
                },
                "explanation_text": "High heart rate contributed most to the prediction",
            }

            explanation = await ml_service.explain_prediction(
                sample_features,
                prediction_result={
                    "condition": "atrial_fibrillation",
                    "confidence": 0.85,
                },
            )

            assert "feature_importance" in explanation
            assert "explanation_text" in explanation
            mock_explain.assert_called_once()


# Testes de integração específicos para componentes não cobertos
class TestMLModelIntegration:
    """Testes de integração para aumentar cobertura"""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_real_ecg(self):
        """Teste pipeline completo com dados ECG reais simulados"""
        service = MLModelService()

        # Simula ECG real de 10 segundos
        sampling_rate = 360
        duration = 10
        t = np.linspace(0, duration, sampling_rate * duration)

        # Simula batimentos cardíacos com ruído
        heart_rate = 72
        ecg_signal = (
            0.5 * np.sin(2 * np.pi * heart_rate / 60 * t)  # Componente P
            + 1.5 * np.sin(2 * np.pi * heart_rate / 60 * t * 2)  # Componente QRS
            + 0.3 * np.sin(2 * np.pi * heart_rate / 60 * t * 0.5)  # Componente T
            + 0.1 * np.random.randn(len(t))  # Ruído
        )

        ecg_data = {
            "signal": ecg_signal.tolist(),
            "sampling_rate": sampling_rate,
            "leads": ["II"],
            "patient_age": 55,
            "patient_sex": "F",
            "metadata": {
                "device": "CardioAI Pro Device",
                "timestamp": datetime.now().isoformat(),
            },
        }

        with patch.object(service, "model_loaded", True):
            with patch.object(
                service,
                "models",
                {
                    "cardiac_classifier": Mock(
                        predict_proba=Mock(return_value=np.array([[0.05, 0.95]]))
                    )
                },
            ):
                result = await service.analyze_ecg(ecg_data, "patient_123")

                assert result is not None
                assert "predictions" in result
                assert "features" in result
                assert "quality_score" in result
