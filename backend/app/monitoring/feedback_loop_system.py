"""
Continuous Learning and Feedback Loop System for CardioAI Pro
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RetrainingTrigger:
    """Configuration for when to trigger model retraining"""

    performance_drop_threshold: float = 0.05  # 5% drop in performance
    critical_miss_threshold: int = 3  # Number of critical misses
    time_based_days: int = 30  # Retrain every 30 days
    minimum_samples: int = 1000  # Minimum new samples before retraining
    confidence_calibration_threshold: float = 0.1  # Calibration error threshold


@dataclass
class PerformanceMetrics:
    """Model performance tracking metrics"""

    timestamp: datetime = field(default_factory=datetime.now)
    overall_accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    critical_miss_rate: float = 0.0
    confidence_calibration_error: float = 0.0
    processing_time_ms: float = 0.0
    sample_count: int = 0


@dataclass
class FeedbackEntry:
    """Individual feedback entry from validation"""

    analysis_id: str
    timestamp: datetime
    ai_prediction: str
    clinical_validation: str
    discrepancy: bool
    severity: str  # low, medium, high, critical
    confidence_score: float
    signal_quality: float
    validator_notes: Optional[str] = None


@dataclass
class ErrorPattern:
    """Identified error pattern in predictions"""

    pattern_type: str
    frequency: float
    conditions: List[str]
    severity: str
    recommendations: List[str]
    first_observed: datetime
    last_observed: datetime


class PerformanceTracker:
    """Track model performance over time"""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_metrics = PerformanceMetrics()
        self.baseline_metrics = PerformanceMetrics()
        self.performance_window_days = 7

    def update_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        confidence_scores: np.ndarray,
        processing_times: List[float],
    ) -> PerformanceMetrics:
        """Update performance metrics with new batch"""
        try:
            # Calculate metrics
            accuracy = np.mean(predictions == ground_truth)
            
            # Calculate sensitivity and specificity for binary case
            true_positives = np.sum((predictions == 1) & (ground_truth == 1))
            true_negatives = np.sum((predictions == 0) & (ground_truth == 0))
            false_positives = np.sum((predictions == 1) & (ground_truth == 0))
            false_negatives = np.sum((predictions == 0) & (ground_truth == 1))

            sensitivity = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            specificity = (
                true_negatives / (true_negatives + false_positives)
                if (true_negatives + false_positives) > 0
                else 0
            )

            # Calculate confidence calibration error
            avg_confidence = np.mean(confidence_scores)
            calibration_error = abs(accuracy - avg_confidence)

            # Update metrics
            self.current_metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                overall_accuracy=float(accuracy),
                sensitivity=float(sensitivity),
                specificity=float(specificity),
                critical_miss_rate=float(false_negatives / len(predictions)),
                confidence_calibration_error=float(calibration_error),
                processing_time_ms=float(np.mean(processing_times)),
                sample_count=len(predictions),
            )

            self.metrics_history.append(self.current_metrics)
            
            # Maintain window
            cutoff_date = datetime.now() - timedelta(days=self.performance_window_days)
            self.metrics_history = [
                m for m in self.metrics_history if m.timestamp > cutoff_date
            ]

            return self.current_metrics

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            return self.current_metrics

    def get_performance_trend(self) -> Dict[str, float]:
        """Calculate performance trend over time window"""
        if len(self.metrics_history) < 2:
            return {"trend": "stable", "change": 0.0}

        # Compare recent to baseline
        recent_metrics = self.metrics_history[-min(10, len(self.metrics_history)) :]
        older_metrics = self.metrics_history[: min(10, len(self.metrics_history))]

        recent_accuracy = np.mean([m.overall_accuracy for m in recent_metrics])
        older_accuracy = np.mean([m.overall_accuracy for m in older_metrics])

        change = recent_accuracy - older_accuracy

        return {
            "trend": "improving" if change > 0.01 else "declining" if change < -0.01 else "stable",
            "change": float(change),
            "recent_accuracy": float(recent_accuracy),
            "baseline_accuracy": float(older_accuracy),
        }


class ErrorAnalyzer:
    """Analyze error patterns in model predictions"""

    def __init__(self):
        self.error_buffer: List[FeedbackEntry] = []
        self.identified_patterns: List[ErrorPattern] = []
        self.pattern_threshold = 0.05  # 5% occurrence threshold

    def add_error(self, feedback: FeedbackEntry) -> None:
        """Add error to analysis buffer"""
        if feedback.discrepancy:
            self.error_buffer.append(feedback)
            
            # Maintain buffer size
            if len(self.error_buffer) > 10000:
                self.error_buffer = self.error_buffer[-5000:]

    def analyze_patterns(self) -> List[ErrorPattern]:
        """Analyze error buffer for patterns"""
        if len(self.error_buffer) < 100:
            return []

        patterns = []

        # Group by prediction type
        prediction_groups = {}
        for error in self.error_buffer:
            pred = error.ai_prediction
            if pred not in prediction_groups:
                prediction_groups[pred] = []
            prediction_groups[pred].append(error)

        # Analyze each group
        for prediction, errors in prediction_groups.items():
            if len(errors) / len(self.error_buffer) > self.pattern_threshold:
                # Significant pattern found
                severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
                for e in errors:
                    severity_counts[e.severity] += 1

                most_common_severity = max(severity_counts, key=severity_counts.get)

                pattern = ErrorPattern(
                    pattern_type=f"Misclassification of {prediction}",
                    frequency=len(errors) / len(self.error_buffer),
                    conditions=[prediction],
                    severity=most_common_severity,
                    recommendations=self._generate_recommendations(prediction, errors),
                    first_observed=min(e.timestamp for e in errors),
                    last_observed=max(e.timestamp for e in errors),
                )
                patterns.append(pattern)

        self.identified_patterns = patterns
        return patterns

    def _generate_recommendations(
        self, prediction: str, errors: List[FeedbackEntry]
    ) -> List[str]:
        """Generate recommendations based on error pattern"""
        recommendations = []

        # Analyze signal quality correlation
        avg_quality = np.mean([e.signal_quality for e in errors])
        if avg_quality < 0.7:
            recommendations.append(
                "Consider additional preprocessing for low-quality signals"
            )

        # Analyze confidence correlation
        avg_confidence = np.mean([e.confidence_score for e in errors])
        if avg_confidence > 0.8:
            recommendations.append(
                "Model is overconfident on these cases - consider confidence calibration"
            )
        elif avg_confidence < 0.5:
            recommendations.append(
                "Low confidence suggests model uncertainty - consider additional training data"
            )

        # Specific condition recommendations
        if "AFIB" in prediction:
            recommendations.append(
                "Review AFIB detection algorithm - consider rhythm analysis improvements"
            )
        elif "STEMI" in prediction:
            recommendations.append(
                "Enhance ST-segment analysis and consider multi-lead correlation"
            )

        return recommendations


class ContinuousLearningSystem:
    """Main continuous learning system with feedback integration"""

    def __init__(self, model: Any = None, config: Optional[RetrainingTrigger] = None):
        """Initialize with optional model and configuration"""
        self.model = model
        self.config = config or RetrainingTrigger()
        self.performance_tracker = PerformanceTracker()
        self.error_analyzer = ErrorAnalyzer()
        self.feedback_buffer: List[FeedbackEntry] = []
        self.last_retrain_date = datetime.now()
        self.total_samples_since_retrain = 0
        self.retraining_in_progress = False
        self.retraining_history: List[Dict[str, Any]] = []

        logger.info("Continuous Learning System initialized")

    def add_feedback(
        self,
        analysis_id: str,
        ai_prediction: str,
        clinical_validation: str,
        confidence_score: float,
        signal_quality: float,
        validator_notes: Optional[str] = None,
    ) -> None:
        """Add clinical feedback for an analysis"""
        try:
            # Determine if there's a discrepancy
            discrepancy = ai_prediction != clinical_validation
            
            # Determine severity
            severity = self._calculate_severity(ai_prediction, clinical_validation)

            # Create feedback entry
            feedback = FeedbackEntry(
                analysis_id=analysis_id,
                timestamp=datetime.now(),
                ai_prediction=ai_prediction,
                clinical_validation=clinical_validation,
                discrepancy=discrepancy,
                severity=severity,
                confidence_score=confidence_score,
                signal_quality=signal_quality,
                validator_notes=validator_notes,
            )

            # Add to buffer
            self.feedback_buffer.append(feedback)
            self.total_samples_since_retrain += 1

            # Add to error analyzer if discrepancy
            if discrepancy:
                self.error_analyzer.add_error(feedback)

            # Check if retraining is needed
            if self._should_trigger_retraining():
                self._initiate_retraining()

            logger.info(
                f"Feedback added for analysis {analysis_id}: "
                f"discrepancy={discrepancy}, severity={severity}"
            )

        except Exception as e:
            logger.error(f"Error adding feedback: {e}")

    def _calculate_severity(self, ai_prediction: str, clinical_validation: str) -> str:
        """Calculate severity of prediction discrepancy"""
        critical_conditions = ["VF", "VT", "STEMI", "Complete Heart Block"]
        
        # Critical miss - AI missed critical condition
        if clinical_validation in critical_conditions and ai_prediction == "Normal":
            return "critical"
        
        # High severity - misclassified between major categories
        if (ai_prediction == "Normal" and clinical_validation != "Normal") or (
            ai_prediction != "Normal" and clinical_validation == "Normal"
        ):
            return "high"
        
        # Medium severity - wrong specific condition
        if ai_prediction != clinical_validation:
            return "medium"
        
        return "low"

    def _should_trigger_retraining(self) -> bool:
        """Check if retraining should be triggered"""
        if self.retraining_in_progress:
            return False

        # Time-based trigger
        days_since_retrain = (datetime.now() - self.last_retrain_date).days
        if days_since_retrain >= self.config.time_based_days:
            logger.info(f"Time-based retraining trigger: {days_since_retrain} days")
            return True

        # Sample count trigger
        if self.total_samples_since_retrain >= self.config.minimum_samples:
            # Check performance metrics
            performance_trend = self.performance_tracker.get_performance_trend()
            
            if performance_trend["change"] < -self.config.performance_drop_threshold:
                logger.info(
                    f"Performance drop trigger: {performance_trend['change']:.3f}"
                )
                return True

            # Check critical misses
            critical_misses = sum(
                1 for f in self.feedback_buffer[-100:] 
                if f.severity == "critical" and f.discrepancy
            )
            if critical_misses >= self.config.critical_miss_threshold:
                logger.info(f"Critical miss trigger: {critical_misses} misses")
                return True

        return False

    def _initiate_retraining(self) -> None:
        """Initiate model retraining process"""
        try:
            self.retraining_in_progress = True
            
            # Analyze error patterns
            patterns = self.error_analyzer.analyze_patterns()
            
            # Prepare retraining data focusing on errors
            retraining_config = {
                "timestamp": datetime.now(),
                "trigger_reasons": self._get_trigger_reasons(),
                "error_patterns": patterns,
                "performance_metrics": self.performance_tracker.current_metrics,
                "feedback_samples": len(self.feedback_buffer),
                "focus_conditions": self._identify_focus_conditions(),
            }

            # Log retraining event
            self.retraining_history.append(
                {
                    "timestamp": datetime.now(),
                    "config": retraining_config,
                    "status": "initiated",
                    "trigger_reasons": retraining_config["trigger_reasons"],
                }
            )

            logger.info(
                f"Model retraining initiated with {len(patterns)} identified patterns"
            )

            # In production, this would trigger actual retraining pipeline
            # For now, we'll simulate completion
            self._complete_retraining(success=True)

        except Exception as e:
            logger.error(f"Error initiating retraining: {e}")
            self.retraining_in_progress = False

    def _get_trigger_reasons(self) -> List[str]:
        """Get list of reasons that triggered retraining"""
        reasons = []
        
        days_since = (datetime.now() - self.last_retrain_date).days
        if days_since >= self.config.time_based_days:
            reasons.append(f"Time-based: {days_since} days since last retrain")
        
        performance_trend = self.performance_tracker.get_performance_trend()
        if performance_trend["change"] < -self.config.performance_drop_threshold:
            reasons.append(
                f"Performance drop: {performance_trend['change']:.3f} decline"
            )
        
        critical_misses = sum(
            1 for f in self.feedback_buffer[-100:] 
            if f.severity == "critical" and f.discrepancy
        )
        if critical_misses >= self.config.critical_miss_threshold:
            reasons.append(f"Critical misses: {critical_misses} recent misses")
        
        return reasons

    def _identify_focus_conditions(self) -> List[str]:
        """Identify conditions that need focus in retraining"""
        condition_errors = {}
        
        for feedback in self.feedback_buffer:
            if feedback.discrepancy:
                condition = feedback.clinical_validation
                if condition not in condition_errors:
                    condition_errors[condition] = 0
                condition_errors[condition] += 1
        
        # Sort by error count
        sorted_conditions = sorted(
            condition_errors.items(), key=lambda x: x[1], reverse=True
        )
        
        # Return top conditions
        return [c[0] for c in sorted_conditions[:5]]

    def _complete_retraining(self, success: bool) -> None:
        """Complete retraining process"""
        self.retraining_in_progress = False
        
        if success:
            self.last_retrain_date = datetime.now()
            self.total_samples_since_retrain = 0
            self.feedback_buffer = []  # Clear old feedback
            
            # Update history
            if self.retraining_history:
                self.retraining_history[-1]["status"] = "completed"
                self.retraining_history[-1]["completion_time"] = datetime.now()
            
            logger.info("Model retraining completed successfully")
        else:
            if self.retraining_history:
                self.retraining_history[-1]["status"] = "failed"
            
            logger.error("Model retraining failed")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        performance_trend = self.performance_tracker.get_performance_trend()
        error_patterns = self.error_analyzer.analyze_patterns()
        
        # Calculate feedback statistics
        total_feedback = len(self.feedback_buffer)
        discrepancies = sum(1 for f in self.feedback_buffer if f.discrepancy)
        
        return {
            "current_metrics": {
                "accuracy": self.performance_tracker.current_metrics.overall_accuracy,
                "sensitivity": self.performance_tracker.current_metrics.sensitivity,
                "specificity": self.performance_tracker.current_metrics.specificity,
                "critical_miss_rate": self.performance_tracker.current_metrics.critical_miss_rate,
            },
            "performance_trend": performance_trend,
            "feedback_statistics": {
                "total_samples": total_feedback,
                "discrepancy_rate": discrepancies / total_feedback if total_feedback > 0 else 0,
                "samples_since_retrain": self.total_samples_since_retrain,
            },
            "error_patterns": [
                {
                    "type": p.pattern_type,
                    "frequency": p.frequency,
                    "severity": p.severity,
                }
                for p in error_patterns
            ],
            "last_retrain_date": self.last_retrain_date.isoformat(),
            "retraining_status": "in_progress" if self.retraining_in_progress else "idle",
        }

    def export_performance_data(self, filepath: Path) -> None:
        """Export performance data for analysis"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "performance_metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "accuracy": m.overall_accuracy,
                    "sensitivity": m.sensitivity,
                    "specificity": m.specificity,
                    "critical_miss_rate": m.critical_miss_rate,
                    "confidence_calibration_error": m.confidence_calibration_error,
                    "sample_count": m.sample_count,
                }
                for m in self.performance_tracker.metrics_history
            ],
            "feedback_entries": [
                {
                    "analysis_id": entry.analysis_id,
                    "timestamp": entry.timestamp.isoformat(),
                    "discrepancy": entry.discrepancy,
                    "confidence_score": entry.confidence_score,
                    "signal_quality": entry.signal_quality,
                }
                for entry in self.feedback_buffer
            ],
            "performance_history": [
                {
                    "timestamp": record["timestamp"].isoformat(),
                    "metrics": {
                        "overall_accuracy": record["metrics"].overall_accuracy,
                        "critical_miss_rate": record["metrics"].critical_miss_rate,
                        "confidence_calibration_error": record[
                            "metrics"
                        ].confidence_calibration_error,
                    },
                    "sample_count": record["sample_count"],
                }
                for record in self.performance_tracker.performance_history
            ],
            "error_patterns": [
                {
                    "pattern_type": pattern.pattern_type,
                    "frequency": pattern.frequency,
                    "conditions": pattern.conditions,
                    "severity": pattern.severity,
                    "recommendations": pattern.recommendations,
                    "first_observed": pattern.first_observed.isoformat(),
                    "last_observed": pattern.last_observed.isoformat(),
                }
                for pattern in self.error_analyzer.identified_patterns
            ],
            "retraining_history": [
                {
                    "timestamp": record["timestamp"].isoformat(),
                    "trigger_reasons": record["trigger_reasons"],
                    "status": record["status"],
                }
                for record in self.retraining_history
            ],
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Performance data exported to {filepath}")


def create_continuous_learning_system(model: Any = None, **kwargs) -> ContinuousLearningSystem:
    """
    Factory function to create continuous learning system

    Args:
        model: The ML model to monitor and improve (optional)
        **kwargs: Additional configuration parameters

    Returns:
        Configured continuous learning system
    """
    config = RetrainingTrigger(**kwargs)
    return ContinuousLearningSystem(model=model, config=config)


class ContinuousLearningService(ContinuousLearningSystem):
    """Backward compatible alias for ContinuousLearningSystem."""
    
    def __init__(self, model: Any = None, config: Optional[RetrainingTrigger] = None):
        """Initialize with optional model parameter for compatibility"""
        super().__init__(model=model, config=config)
