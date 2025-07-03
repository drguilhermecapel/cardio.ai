"""
Testes para componentes médicos críticos - Meta: 100% de cobertura
Componentes críticos: ECG Analysis, Signal Quality, Diagnosis Engine
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

from app.services.ecg import ECGAnalysisService
from app.services.ecg.signal_quality import SignalQualityAnalyzer
from app.services.diagnosis import DiagnosisEngine
from app.core.exceptions import (
    CriticalMedicalException,
    SignalQualityException,
    DiagnosisException,
)


class TestECGAnalysisServiceComplete:
    """Testes completos para ECG Analysis Service - 100% cobertura"""

    @pytest.fixture
    def ecg_service(self):
        db = AsyncMock()
        ml_service = Mock()
        validation_service = Mock()
        return ECGAnalysisService(db, ml_service, validation_service)

    @pytest.fixture
    def raw_ecg_signal(self):
        """Sinal ECG bruto simulado"""
        # Simula 10 segundos de ECG a 360Hz
        sampling_rate = 360
        duration = 10
        samples = sampling_rate * duration

        # Gera sinal ECG sintético com componentes PQRST
        t = np.linspace(0, duration, samples)

        # Frequência cardíaca: 72 bpm
        heart_rate = 72
        beat_interval = 60.0 / heart_rate

        ecg = np.zeros(samples)

        # Adiciona complexos QRS
        for i in range(int(duration / beat_interval)):
            qrs_center = int(i * beat_interval * sampling_rate)
            if qrs_center < samples:
                # Onda P
                p_start = max(0, qrs_center - int(0.08 * sampling_rate))
                p_end = min(samples, qrs_center - int(0.02 * sampling_rate))
                if p_start < p_end:
                    ecg[p_start:p_end] += 0.15 * np.sin(
                        np.linspace(0, np.pi, p_end - p_start)
                    )

                # Complexo QRS
                qrs_start = max(0, qrs_center - int(0.04 * sampling_rate))
                qrs_end = min(samples, qrs_center + int(0.04 * sampling_rate))
                if qrs_start < qrs_end:
                    qrs_len = qrs_end - qrs_start
                    # Onda Q
                    ecg[qrs_start : qrs_start + qrs_len // 3] -= 0.1
                    # Onda R
                    ecg[qrs_start + qrs_len // 3 : qrs_start + 2 * qrs_len // 3] += 1.5
                    # Onda S
                    ecg[qrs_start + 2 * qrs_len // 3 : qrs_end] -= 0.3

                # Onda T
                t_start = min(samples - 1, qrs_center + int(0.1 * sampling_rate))
                t_end = min(samples, qrs_center + int(0.3 * sampling_rate))
                if t_start < t_end:
                    ecg[t_start:t_end] += 0.3 * np.sin(
                        np.linspace(0, np.pi, t_end - t_start)
                    )

        # Adiciona ruído realista
        ecg += 0.05 * np.random.randn(samples)

        return ecg, sampling_rate

    @pytest.mark.asyncio
    async def test_analyze_ecg_complete_flow(self, ecg_service, raw_ecg_signal):
        """Teste fluxo completo de análise ECG"""
        ecg_data, sampling_rate = raw_ecg_signal

        analysis_request = {
            "ecg_data": ecg_data.tolist(),
            "sampling_rate": sampling_rate,
            "patient_id": "PAT001",
            "lead_configuration": "II",
            "device_id": "DEV001",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Mock das dependências
        ecg_service.ml_service.analyze_ecg = AsyncMock(
            return_value={
                "predictions": {
                    "normal_sinus_rhythm": 0.95,
                    "atrial_fibrillation": 0.03,
                    "other": 0.02,
                },
                "confidence": 0.95,
            }
        )

        ecg_service.repository.create_analysis = AsyncMock(return_value=Mock(id=1))
        ecg_service.repository.save_raw_signal = AsyncMock(return_value=True)
        ecg_service.validation_service.create_validation_request = AsyncMock()

        result = await ecg_service.analyze_ecg(analysis_request)

        assert result is not None
        assert "analysis_id" in result
        assert "predictions" in result
        assert "requires_validation" in result
        assert result["predictions"]["normal_sinus_rhythm"] == 0.95

    @pytest.mark.asyncio
    async def test_analyze_ecg_critical_finding(self, ecg_service, raw_ecg_signal):
        """Teste quando achado crítico é detectado"""
        ecg_data, sampling_rate = raw_ecg_signal

        analysis_request = {
            "ecg_data": ecg_data.tolist(),
            "sampling_rate": sampling_rate,
            "patient_id": "PAT002",
        }

        # Simula detecção de condição crítica
        ecg_service.ml_service.analyze_ecg = AsyncMock(
            return_value={
                "predictions": {
                    "ventricular_tachycardia": 0.89,  # Condição crítica
                    "normal_sinus_rhythm": 0.11,
                },
                "confidence": 0.89,
                "critical": True,
            }
        )

        ecg_service.repository.create_analysis = AsyncMock(return_value=Mock(id=2))
        ecg_service.repository.flag_critical = AsyncMock()
        ecg_service.notification_service = Mock()
        ecg_service.notification_service.send_critical_alert = AsyncMock()

        result = await ecg_service.analyze_ecg(analysis_request)

        assert result["critical"] is True
        assert result["requires_immediate_attention"] is True
        ecg_service.repository.flag_critical.assert_called_once()
        ecg_service.notification_service.send_critical_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_ecg_poor_signal_quality(self, ecg_service):
        """Teste com sinal de baixa qualidade"""
        # Sinal com muito ruído
        noisy_signal = np.random.randn(3600) * 5  # Alto ruído

        analysis_request = {
            "ecg_data": noisy_signal.tolist(),
            "sampling_rate": 360,
            "patient_id": "PAT003",
        }

        with pytest.raises(SignalQualityException, match="Signal quality too poor"):
            await ecg_service.analyze_ecg(analysis_request)

    @pytest.mark.asyncio
    async def test_analyze_ecg_all_error_paths(self, ecg_service, raw_ecg_signal):
        """Teste todos os caminhos de erro possíveis"""
        ecg_data, sampling_rate = raw_ecg_signal

        # Erro na análise ML
        ecg_service.ml_service.analyze_ecg = AsyncMock(
            side_effect=Exception("ML model error")
        )

        with pytest.raises(CriticalMedicalException):
            await ecg_service.analyze_ecg(
                {
                    "ecg_data": ecg_data.tolist(),
                    "sampling_rate": sampling_rate,
                    "patient_id": "PAT004",
                }
            )

        # Erro ao salvar no banco
        ecg_service.ml_service.analyze_ecg = AsyncMock(return_value={"predictions": {}})
        ecg_service.repository.create_analysis = AsyncMock(
            side_effect=Exception("Database error")
        )

        with pytest.raises(CriticalMedicalException):
            await ecg_service.analyze_ecg(
                {
                    "ecg_data": ecg_data.tolist(),
                    "sampling_rate": sampling_rate,
                    "patient_id": "PAT005",
                }
            )

    @pytest.mark.asyncio
    async def test_get_analysis_by_id_all_cases(self, ecg_service):
        """Teste obtenção de análise por ID - todos os casos"""
        # Caso 1: Análise existe
        mock_analysis = Mock(
            id=1,
            patient_id="PAT001",
            predictions={"normal_sinus_rhythm": 0.95},
            created_at=datetime.utcnow(),
        )
        ecg_service.repository.get_analysis_by_id = AsyncMock(
            return_value=mock_analysis
        )

        result = await ecg_service.get_analysis_by_id(1)
        assert result.id == 1

        # Caso 2: Análise não existe
        ecg_service.repository.get_analysis_by_id = AsyncMock(return_value=None)
        result = await ecg_service.get_analysis_by_id(999)
        assert result is None

        # Caso 3: Erro no banco
        ecg_service.repository.get_analysis_by_id = AsyncMock(
            side_effect=Exception("DB error")
        )
        with pytest.raises(Exception):
            await ecg_service.get_analysis_by_id(1)

    @pytest.mark.asyncio
    async def test_update_analysis_validation(self, ecg_service):
        """Teste atualização de validação da análise"""
        validation_data = {
            "validated_by": "DR001",
            "validation_status": "approved",
            "clinical_notes": "Confirmed normal sinus rhythm",
            "timestamp": datetime.utcnow(),
        }

        ecg_service.repository.update_validation = AsyncMock(return_value=True)
        ecg_service.repository.get_analysis_by_id = AsyncMock(
            return_value=Mock(id=1, requires_validation=True)
        )

        result = await ecg_service.update_analysis_validation(1, validation_data)
        assert result is True

        # Teste quando análise não requer validação
        ecg_service.repository.get_analysis_by_id = AsyncMock(
            return_value=Mock(id=2, requires_validation=False)
        )

        result = await ecg_service.update_analysis_validation(2, validation_data)
        assert result is False


class TestSignalQualityAnalyzerComplete:
    """Testes completos para Signal Quality Analyzer - 100% cobertura"""

    @pytest.fixture
    def analyzer(self):
        return SignalQualityAnalyzer()

    def test_analyze_signal_quality_perfect_signal(self, analyzer):
        """Teste com sinal perfeito"""
        # Sinal ECG limpo
        t = np.linspace(0, 10, 3600)
        clean_signal = np.sin(2 * np.pi * 1.2 * t)  # 72 bpm

        quality = analyzer.analyze_signal_quality(clean_signal, 360)

        assert quality["overall_score"] > 0.9
        assert quality["noise_level"] == "low"
        assert quality["baseline_wander"] is False
        assert quality["motion_artifacts"] is False
        assert quality["electrode_contact"] == "good"

    def test_analyze_signal_quality_noisy_signal(self, analyzer):
        """Teste com sinal ruidoso"""
        # Sinal com muito ruído
        noisy_signal = np.random.randn(3600) * 2

        quality = analyzer.analyze_signal_quality(noisy_signal, 360)

        assert quality["overall_score"] < 0.3
        assert quality["noise_level"] == "high"
        assert quality["usable"] is False

    def test_analyze_signal_quality_baseline_wander(self, analyzer):
        """Teste com desvio de linha de base"""
        t = np.linspace(0, 10, 3600)
        # Sinal com desvio de linha de base
        signal_with_wander = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(
            2 * np.pi * 0.1 * t
        )

        quality = analyzer.analyze_signal_quality(signal_with_wander, 360)

        assert quality["baseline_wander"] is True
        assert quality["overall_score"] < 0.8

    def test_analyze_signal_quality_motion_artifacts(self, analyzer):
        """Teste com artefatos de movimento"""
        t = np.linspace(0, 10, 3600)
        clean_signal = np.sin(2 * np.pi * 1.2 * t)

        # Adiciona artefatos de movimento
        motion_artifact = np.zeros_like(clean_signal)
        motion_artifact[1000:1200] = np.random.randn(200) * 3
        signal_with_motion = clean_signal + motion_artifact

        quality = analyzer.analyze_signal_quality(signal_with_motion, 360)

        assert quality["motion_artifacts"] is True
        assert quality["artifact_regions"] is not None
        assert len(quality["artifact_regions"]) > 0

    def test_analyze_signal_quality_poor_electrode_contact(self, analyzer):
        """Teste com contato ruim do eletrodo"""
        # Simula perda intermitente de contato
        signal = np.random.randn(3600) * 0.1
        signal[500:700] = 0  # Perda de sinal
        signal[1500:1600] = 0
        signal[2500:2800] = 0

        quality = analyzer.analyze_signal_quality(signal, 360)

        assert quality["electrode_contact"] == "poor"
        assert quality["signal_loss_regions"] is not None
        assert len(quality["signal_loss_regions"]) >= 3

    def test_analyze_signal_quality_edge_cases(self, analyzer):
        """Teste casos extremos"""
        # Sinal vazio
        with pytest.raises(SignalQualityException):
            analyzer.analyze_signal_quality([], 360)

        # Taxa de amostragem inválida
        with pytest.raises(SignalQualityException):
            analyzer.analyze_signal_quality([1, 2, 3], 0)

        # Sinal muito curto
        with pytest.raises(SignalQualityException):
            analyzer.analyze_signal_quality([1, 2, 3], 360)

        # Sinal com NaN
        signal_with_nan = np.ones(3600)
        signal_with_nan[100:200] = np.nan

        quality = analyzer.analyze_signal_quality(signal_with_nan, 360)
        assert quality["has_invalid_values"] is True
        assert quality["overall_score"] < 0.5

    def test_calculate_snr(self, analyzer):
        """Teste cálculo de relação sinal-ruído"""
        # Sinal com SNR conhecido
        signal = np.sin(np.linspace(0, 10 * np.pi, 1000))
        noise = np.random.randn(1000) * 0.1
        signal_with_noise = signal + noise

        snr = analyzer._calculate_snr(signal_with_noise)
        assert snr > 10  # SNR alto para sinal limpo

        # Sinal com muito ruído
        noisy_signal = signal + np.random.randn(1000) * 2
        snr_noisy = analyzer._calculate_snr(noisy_signal)
        assert snr_noisy < snr

    def test_detect_baseline_wander(self, analyzer):
        """Teste detecção de desvio de linha de base"""
        t = np.linspace(0, 10, 3600)

        # Sem desvio
        clean_signal = np.sin(2 * np.pi * 1.2 * t)
        assert analyzer._detect_baseline_wander(clean_signal, 360) is False

        # Com desvio significativo
        wandering_signal = clean_signal + 2 * np.sin(2 * np.pi * 0.05 * t)
        assert analyzer._detect_baseline_wander(wandering_signal, 360) is True

    def test_detect_motion_artifacts(self, analyzer):
        """Teste detecção de artefatos de movimento"""
        signal = np.sin(np.linspace(0, 10 * np.pi, 3600))

        # Sem artefatos
        artifacts = analyzer._detect_motion_artifacts(signal, 360)
        assert len(artifacts) == 0

        # Com artefatos
        signal[1000:1100] += np.random.randn(100) * 5
        signal[2000:2050] += np.random.randn(50) * 5

        artifacts = analyzer._detect_motion_artifacts(signal, 360)
        assert len(artifacts) >= 2

    def test_assess_electrode_contact(self, analyzer):
        """Teste avaliação de contato do eletrodo"""
        # Bom contato
        good_signal = (
            np.sin(np.linspace(0, 10 * np.pi, 3600)) + np.random.randn(3600) * 0.1
        )
        contact = analyzer._assess_electrode_contact(good_signal)
        assert contact == "good"

        # Contato intermitente
        intermittent_signal = good_signal.copy()
        intermittent_signal[500:600] = 0
        intermittent_signal[1500:1600] = 0

        contact = analyzer._assess_electrode_contact(intermittent_signal)
        assert contact in ["fair", "poor"]

        # Sem contato
        no_contact_signal = np.zeros(3600)
        contact = analyzer._assess_electrode_contact(no_contact_signal)
        assert contact == "no_contact"


class TestDiagnosisEngineComplete:
    """Testes completos para Diagnosis Engine - 100% cobertura"""

    @pytest.fixture
    def diagnosis_engine(self):
        return DiagnosisEngine()

    @pytest.fixture
    def normal_ecg_features(self):
        return {
            "heart_rate": 72,
            "pr_interval": 160,
            "qrs_duration": 90,
            "qt_interval": 380,
            "qtc_interval": 395,
            "p_wave": {"present": True, "morphology": "normal"},
            "qrs_complex": {"morphology": "normal", "axis": 45},
            "t_wave": {"morphology": "normal", "amplitude": 0.3},
            "rhythm": "regular",
            "rr_variability": 0.05,
        }

    @pytest.mark.asyncio
    async def test_diagnose_normal_sinus_rhythm(
        self, diagnosis_engine, normal_ecg_features
    ):
        """Teste diagnóstico de ritmo sinusal normal"""
        ml_predictions = {
            "normal_sinus_rhythm": 0.95,
            "atrial_fibrillation": 0.03,
            "other": 0.02,
        }

        diagnosis = await diagnosis_engine.diagnose(normal_ecg_features, ml_predictions)

        assert diagnosis["primary_diagnosis"] == "normal_sinus_rhythm"
        assert diagnosis["confidence"] > 0.9
        assert diagnosis["severity"] == "normal"
        assert len(diagnosis["findings"]) > 0
        assert "recommendations" in diagnosis

    @pytest.mark.asyncio
    async def test_diagnose_atrial_fibrillation(self, diagnosis_engine):
        """Teste diagnóstico de fibrilação atrial"""
        af_features = {
            "heart_rate": 110,
            "pr_interval": None,  # Ausente na FA
            "qrs_duration": 85,
            "rhythm": "irregularly_irregular",
            "p_wave": {"present": False},
            "rr_variability": 0.25,
        }

        ml_predictions = {
            "atrial_fibrillation": 0.88,
            "normal_sinus_rhythm": 0.10,
            "other": 0.02,
        }

        diagnosis = await diagnosis_engine.diagnose(af_features, ml_predictions)

        assert diagnosis["primary_diagnosis"] == "atrial_fibrillation"
        assert diagnosis["severity"] in ["moderate", "high"]
        assert "anticoagulation assessment" in diagnosis["recommendations"]

    @pytest.mark.asyncio
    async def test_diagnose_ventricular_tachycardia(self, diagnosis_engine):
        """Teste diagnóstico de taquicardia ventricular - condição crítica"""
        vt_features = {
            "heart_rate": 180,
            "qrs_duration": 140,  # QRS largo
            "rhythm": "regular",
            "p_wave": {"present": False},
            "qrs_complex": {"morphology": "wide", "axis": -30},
        }

        ml_predictions = {
            "ventricular_tachycardia": 0.92,
            "svt_aberrancy": 0.06,
            "other": 0.02,
        }

        diagnosis = await diagnosis_engine.diagnose(vt_features, ml_predictions)

        assert diagnosis["primary_diagnosis"] == "ventricular_tachycardia"
        assert diagnosis["severity"] == "critical"
        assert diagnosis["urgent"] is True
        assert "immediate medical attention" in diagnosis["recommendations"]

    @pytest.mark.asyncio
    async def test_diagnose_bradycardia(self, diagnosis_engine):
        """Teste diagnóstico de bradicardia"""
        brady_features = {
            "heart_rate": 45,
            "pr_interval": 200,
            "qrs_duration": 95,
            "rhythm": "regular",
            "p_wave": {"present": True, "morphology": "normal"},
        }

        ml_predictions = {
            "sinus_bradycardia": 0.85,
            "normal_sinus_rhythm": 0.10,
            "other": 0.05,
        }

        diagnosis = await diagnosis_engine.diagnose(brady_features, ml_predictions)

        assert diagnosis["primary_diagnosis"] == "sinus_bradycardia"
        assert "heart_rate" in diagnosis["findings"]
        assert diagnosis["severity"] in ["mild", "moderate"]

    @pytest.mark.asyncio
    async def test_diagnose_long_qt_syndrome(self, diagnosis_engine):
        """Teste diagnóstico de síndrome do QT longo"""
        lqt_features = {
            "heart_rate": 70,
            "pr_interval": 160,
            "qrs_duration": 90,
            "qt_interval": 480,
            "qtc_interval": 495,  # QTc prolongado
            "rhythm": "regular",
        }

        ml_predictions = {
            "long_qt_syndrome": 0.78,
            "normal_sinus_rhythm": 0.20,
            "other": 0.02,
        }

        diagnosis = await diagnosis_engine.diagnose(lqt_features, ml_predictions)

        assert diagnosis["primary_diagnosis"] == "long_qt_syndrome"
        assert diagnosis["risk_factors"] is not None
        assert "medication review" in diagnosis["recommendations"]

    @pytest.mark.asyncio
    async def test_diagnose_multiple_abnormalities(self, diagnosis_engine):
        """Teste com múltiplas anormalidades"""
        complex_features = {
            "heart_rate": 95,
            "pr_interval": 220,  # Prolongado
            "qrs_duration": 120,  # Limítrofe
            "qt_interval": 440,
            "qtc_interval": 470,  # Limítrofe
            "rhythm": "irregular",
            "pvc_count": 15,  # PVCs frequentes
        }

        ml_predictions = {
            "first_degree_av_block": 0.70,
            "pvc": 0.65,
            "normal_sinus_rhythm": 0.15,
        }

        diagnosis = await diagnosis_engine.diagnose(complex_features, ml_predictions)

        assert len(diagnosis["secondary_diagnoses"]) > 0
        assert diagnosis["complexity"] == "complex"
        assert "holter monitoring" in diagnosis["recommendations"]

    @pytest.mark.asyncio
    async def test_diagnose_inconclusive_results(self, diagnosis_engine):
        """Teste resultados inconclusivos"""
        unclear_features = {
            "heart_rate": 75,
            "signal_quality": "poor",
            "artifacts": True,
        }

        ml_predictions = {
            "normal_sinus_rhythm": 0.45,
            "atrial_fibrillation": 0.40,
            "other": 0.15,
        }

        diagnosis = await diagnosis_engine.diagnose(unclear_features, ml_predictions)

        assert diagnosis["confidence"] < 0.5
        assert diagnosis["quality_issues"] is True
        assert "repeat ECG" in diagnosis["recommendations"]

    @pytest.mark.asyncio
    async def test_diagnose_pediatric_considerations(self, diagnosis_engine):
        """Teste considerações pediátricas"""
        pediatric_features = {
            "heart_rate": 120,  # Normal para criança
            "pr_interval": 120,
            "qrs_duration": 70,
            "patient_age": 5,
            "rhythm": "regular",
        }

        ml_predictions = {
            "normal_sinus_rhythm": 0.90,
            "sinus_tachycardia": 0.08,
            "other": 0.02,
        }

        diagnosis = await diagnosis_engine.diagnose(
            pediatric_features,
            ml_predictions,
            patient_context={"age": 5, "pediatric": True},
        )

        assert diagnosis["age_adjusted"] is True
        assert diagnosis["primary_diagnosis"] == "normal_sinus_rhythm"
        assert "pediatric" in diagnosis["interpretation_context"]

    def test_calculate_severity_score(self, diagnosis_engine):
        """Teste cálculo de score de severidade"""
        # Condição normal
        score = diagnosis_engine._calculate_severity_score(
            "normal_sinus_rhythm", confidence=0.95
        )
        assert score < 0.2

        # Condição crítica
        score = diagnosis_engine._calculate_severity_score(
            "ventricular_fibrillation", confidence=0.90
        )
        assert score > 0.9

        # Condição moderada com baixa confiança
        score = diagnosis_engine._calculate_severity_score(
            "atrial_fibrillation", confidence=0.60
        )
        assert 0.3 < score < 0.7

    def test_generate_clinical_recommendations(self, diagnosis_engine):
        """Teste geração de recomendações clínicas"""
        # Ritmo normal
        recommendations = diagnosis_engine._generate_recommendations(
            "normal_sinus_rhythm", severity="normal"
        )
        assert "routine follow-up" in recommendations

        # Condição urgente
        recommendations = diagnosis_engine._generate_recommendations(
            "ventricular_tachycardia", severity="critical"
        )
        assert "emergency" in recommendations
        assert "immediate" in recommendations

        # Condição que requer monitoramento
        recommendations = diagnosis_engine._generate_recommendations(
            "atrial_fibrillation", severity="moderate"
        )
        assert any("anticoagulation" in r for r in recommendations)
        assert any("cardiology referral" in r for r in recommendations)

    @pytest.mark.asyncio
    async def test_diagnose_error_handling(self, diagnosis_engine):
        """Teste tratamento de erros no diagnóstico"""
        # Features inválidas
        with pytest.raises(DiagnosisException):
            await diagnosis_engine.diagnose(None, {"normal": 0.5})

        # Predições inválidas
        with pytest.raises(DiagnosisException):
            await diagnosis_engine.diagnose({"heart_rate": 70}, None)

        # Features incompletas
        incomplete_features = {"heart_rate": 70}  # Faltam muitos parâmetros
        ml_predictions = {"normal_sinus_rhythm": 0.8}

        diagnosis = await diagnosis_engine.diagnose(incomplete_features, ml_predictions)
        assert diagnosis["data_completeness"] < 0.5
        assert "incomplete data" in diagnosis["limitations"]

    def test_validate_diagnosis_consistency(self, diagnosis_engine):
        """Teste validação de consistência do diagnóstico"""
        # Diagnóstico consistente
        is_consistent = diagnosis_engine._validate_consistency(
            features={"heart_rate": 72, "rhythm": "regular"},
            diagnosis="normal_sinus_rhythm",
        )
        assert is_consistent is True

        # Diagnóstico inconsistente
        is_consistent = diagnosis_engine._validate_consistency(
            features={"heart_rate": 180, "qrs_duration": 140},
            diagnosis="normal_sinus_rhythm",
        )
        assert is_consistent is False

    @pytest.mark.asyncio
    async def test_generate_differential_diagnosis(self, diagnosis_engine):
        """Teste geração de diagnóstico diferencial"""
        features = {
            "heart_rate": 150,
            "qrs_duration": 90,
            "rhythm": "regular",
            "p_wave": {"present": False},
        }

        ml_predictions = {
            "svt": 0.60,
            "sinus_tachycardia": 0.30,
            "atrial_flutter": 0.10,
        }

        diagnosis = await diagnosis_engine.diagnose(features, ml_predictions)

        assert "differential_diagnosis" in diagnosis
        assert len(diagnosis["differential_diagnosis"]) >= 2
        assert diagnosis["differential_diagnosis"][0]["condition"] == "svt"
        assert diagnosis["differential_diagnosis"][0]["probability"] == 0.60
