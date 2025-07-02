"""
Structured logging configuration for ECG analysis system
"""

import logging
import sys
from datetime import UTC, datetime
from typing import Any

import structlog


def setup_structured_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the ECG system"""

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class ECGLogger:
    """Structured logger for ECG analysis operations"""

    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)

    def log_analysis_start(
        self,
        patient_id: str,
        format: str,
        lead_count: int,
        sampling_rate: int,
        analysis_id: str,
    ) -> None:
        """Log start of ECG analysis"""
        self.logger.info(
            "ecg_analysis_started",
            event_type="analysis_start",
            patient_id=patient_id,
            analysis_id=analysis_id,
            ecg_format=format,
            lead_count=lead_count,
            sampling_rate=sampling_rate,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def log_analysis_complete(
        self,
        patient_id: str,
        analysis_id: str,
        pathologies_detected: int,
        confidence_mean: float,
        processing_time: float,
        regulatory_compliant: bool,
    ) -> None:
        """Log completion of ECG analysis"""
        self.logger.info(
            "ecg_analysis_completed",
            event_type="analysis_complete",
            patient_id=patient_id,
            analysis_id=analysis_id,
            pathologies_detected=pathologies_detected,
            confidence_mean=confidence_mean,
            processing_time_seconds=processing_time,
            regulatory_compliant=regulatory_compliant,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def log_analysis_error(
        self,
        patient_id: str,
        analysis_id: str,
        error_type: str,
        error_message: str,
        step: str,
        exc_info: bool = True,
    ) -> None:
        """Log ECG analysis error"""
        self.logger.error(
            "ecg_analysis_failed",
            event_type="analysis_error",
            patient_id=patient_id,
            analysis_id=analysis_id,
            error_type=error_type,
            error_message=error_message,
            failed_step=step,
            timestamp=datetime.now(UTC).isoformat(),
            exc_info=exc_info,
        )

    def log_pathology_detection(
        self,
        patient_id: str,
        analysis_id: str,
        pathology_type: str,
        confidence: float,
        model_name: str,
        clinical_significance: str,
    ) -> None:
        """Log pathology detection"""
        self.logger.info(
            "pathology_detected",
            event_type="pathology_detection",
            patient_id=patient_id,
            analysis_id=analysis_id,
            pathology_type=pathology_type,
            confidence=confidence,
            model_name=model_name,
            clinical_significance=clinical_significance,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def log_regulatory_validation(
        self,
        analysis_id: str,
        standard: str,
        compliant: bool,
        validation_details: dict[str, Any],
    ) -> None:
        """Log regulatory validation result"""
        self.logger.info(
            "regulatory_validation",
            event_type="regulatory_check",
            analysis_id=analysis_id,
            regulatory_standard=standard,
            compliant=compliant,
            validation_details=validation_details,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def log_model_performance(
        self,
        model_name: str,
        model_version: str,
        inference_time: float,
        memory_usage_mb: float,
        batch_size: int,
    ) -> None:
        """Log model performance metrics"""
        self.logger.info(
            "model_performance",
            event_type="model_metrics",
            model_name=model_name,
            model_version=model_version,
            inference_time_seconds=inference_time,
            memory_usage_mb=memory_usage_mb,
            batch_size=batch_size,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def log_signal_quality(
        self,
        patient_id: str,
        analysis_id: str,
        lead_qualities: dict[str, float],
        overall_quality: float,
        quality_issues: list[str],
    ) -> None:
        """Log signal quality assessment"""
        self.logger.info(
            "signal_quality_assessment",
            event_type="quality_check",
            patient_id=patient_id,
            analysis_id=analysis_id,
            lead_qualities=lead_qualities,
            overall_quality=overall_quality,
            quality_issues=quality_issues,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def log_preprocessing_step(
        self,
        analysis_id: str,
        step_name: str,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        parameters: dict[str, Any],
        duration: float,
    ) -> None:
        """Log preprocessing step details"""
        self.logger.debug(
            "preprocessing_step",
            event_type="preprocessing",
            analysis_id=analysis_id,
            step_name=step_name,
            input_shape=input_shape,
            output_shape=output_shape,
            parameters=parameters,
            duration_seconds=duration,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def log_feature_extraction(
        self,
        analysis_id: str,
        feature_types: list[str],
        feature_count: int,
        extraction_time: float,
    ) -> None:
        """Log feature extraction results"""
        self.logger.info(
            "feature_extraction",
            event_type="feature_extraction",
            analysis_id=analysis_id,
            feature_types=feature_types,
            feature_count=feature_count,
            extraction_time_seconds=extraction_time,
            timestamp=datetime.now(UTC).isoformat(),
        )


def get_ecg_logger(name: str) -> ECGLogger:
    """Get a structured ECG logger instance"""
    return ECGLogger(name)
