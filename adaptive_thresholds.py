"""
Adaptive Thresholds System for Multi-pathology ECG Detection
Implements Platt scaling and pathology-specific threshold optimization
Based on scientific recommendations for CardioAI Pro
"""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    LogisticRegression = None

from app.core.scp_ecg_conditions import (
    get_conditions_by_urgency,
    get_critical_conditions,
)

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for adaptive thresholds"""

    condition_code: str
    base_threshold: float
    sensitivity_target: float
    specificity_target: float
    clinical_urgency: str
    platt_scaling_params: dict[str, float] | None = None
    performance_history: list[dict[str, float]] | None = None


class AdaptiveThresholdManager:
    """
    Manages adaptive thresholds for 71 SCP-ECG conditions
    Implements Platt scaling for probability calibration
    Optimizes thresholds based on clinical urgency and performance metrics
    """

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or "adaptive_thresholds_config.json"
        self.threshold_configs: dict[str, ThresholdConfig] = {}
        self.platt_scalers: dict[str, Any] = {}
        self.performance_tracker: dict[str, dict[str, Any]] = {}

        self._initialize_default_thresholds()

        self._load_configuration()

    def _initialize_default_thresholds(self) -> None:
        """Initialize default thresholds based on clinical urgency and condition characteristics"""

        critical_conditions = get_critical_conditions()
        for condition in critical_conditions:
            self.threshold_configs[condition.code] = ThresholdConfig(
                condition_code=condition.code,
                base_threshold=0.65,  # Lower threshold for higher sensitivity
                sensitivity_target=condition.sensitivity_target,
                specificity_target=condition.specificity_target,
                clinical_urgency=condition.clinical_urgency,
                performance_history=[],
            )

        high_urgency_conditions = get_conditions_by_urgency("high")
        for condition in high_urgency_conditions:
            if condition.code not in self.threshold_configs:
                self.threshold_configs[condition.code] = ThresholdConfig(
                    condition_code=condition.code,
                    base_threshold=0.70,
                    sensitivity_target=condition.sensitivity_target,
                    specificity_target=condition.specificity_target,
                    clinical_urgency=condition.clinical_urgency,
                    performance_history=[],
                )

        medium_urgency_conditions = get_conditions_by_urgency("medium")
        for condition in medium_urgency_conditions:
            if condition.code not in self.threshold_configs:
                self.threshold_configs[condition.code] = ThresholdConfig(
                    condition_code=condition.code,
                    base_threshold=0.75,
                    sensitivity_target=condition.sensitivity_target,
                    specificity_target=condition.specificity_target,
                    clinical_urgency=condition.clinical_urgency,
                    performance_history=[],
                )

        low_urgency_conditions = get_conditions_by_urgency("low")
        for condition in low_urgency_conditions:
            if condition.code not in self.threshold_configs:
                self.threshold_configs[condition.code] = ThresholdConfig(
                    condition_code=condition.code,
                    base_threshold=0.80,
                    sensitivity_target=condition.sensitivity_target,
                    specificity_target=condition.specificity_target,
                    clinical_urgency=condition.clinical_urgency,
                    performance_history=[],
                )

    def get_adaptive_threshold(
        self, condition_code: str, context: dict[str, Any] | None = None
    ) -> float:
        """
        Get adaptive threshold for a specific condition
        Considers clinical context, recent performance, and calibration
        """

        if condition_code not in self.threshold_configs:
            logger.warning(
                f"No threshold config found for {condition_code}, using default 0.8"
            )
            return 0.8

        config = self.threshold_configs[condition_code]
        base_threshold = config.base_threshold

        adjusted_threshold = self._apply_context_adjustments(
            base_threshold, condition_code, context or {}
        )

        performance_adjusted = self._apply_performance_adjustments(
            adjusted_threshold, condition_code
        )

        final_threshold = max(0.1, min(0.95, performance_adjusted))

        logger.debug(
            f"Adaptive threshold for {condition_code}: {final_threshold:.3f} "
            f"(base: {base_threshold:.3f})"
        )

        return final_threshold

    def _apply_context_adjustments(
        self, base_threshold: float, condition_code: str, context: dict[str, Any]
    ) -> float:
        """Apply context-based threshold adjustments"""

        adjusted_threshold = base_threshold

        age = context.get("patient_age", 50)
        if condition_code in ["AFIB", "VTAC", "STEMI"]:
            if age > 70:
                adjusted_threshold -= 0.05  # More sensitive for elderly
            elif age < 30:
                adjusted_threshold += 0.05  # Less sensitive for young patients

        signal_quality = context.get("signal_quality", 0.8)
        if signal_quality < 0.7:
            adjusted_threshold += 0.1  # Higher threshold for poor quality signals
        elif signal_quality > 0.9:
            adjusted_threshold -= 0.05  # Lower threshold for high quality signals

        setting = context.get("clinical_setting", "general")
        if setting == "emergency":
            if condition_code in ["STEMI", "VTAC", "VFIB", "AVB3"]:
                adjusted_threshold -= 0.1
        elif setting == "screening":
            adjusted_threshold += 0.05

        medications = context.get("medications", [])
        if "digitalis" in medications and condition_code in [
            "VTAC",
            "BIDIRECTIONAL_VT",
        ]:
            adjusted_threshold -= 0.05  # More sensitive to digitalis toxicity

        return adjusted_threshold

    def _apply_performance_adjustments(
        self, threshold: float, condition_code: str
    ) -> float:
        """Apply performance-based threshold adjustments using historical data"""

        config = self.threshold_configs.get(condition_code)
        if not config or not config.performance_history:
            return threshold

        recent_performance = config.performance_history[-10:]
        if len(recent_performance) < 3:
            return threshold

        avg_sensitivity = np.mean([p["sensitivity"] for p in recent_performance])
        avg_specificity = np.mean([p["specificity"] for p in recent_performance])

        target_sensitivity = config.sensitivity_target
        target_specificity = config.specificity_target

        sensitivity_gap = target_sensitivity - avg_sensitivity
        specificity_gap = target_specificity - avg_specificity

        if sensitivity_gap > 0.05:
            threshold -= min(0.1, sensitivity_gap * 0.5)

        if specificity_gap > 0.05:
            threshold += min(0.1, specificity_gap * 0.3)

        urgency_weight = self._get_urgency_weight(config.clinical_urgency)

        if config.clinical_urgency == "critical":
            if sensitivity_gap > 0.02:
                threshold -= sensitivity_gap * urgency_weight
        else:
            net_adjustment = (
                sensitivity_gap * -0.5 + specificity_gap * 0.3
            ) * urgency_weight
            threshold += net_adjustment

        return threshold

    def _get_urgency_weight(self, urgency: str) -> float:
        """Get adjustment weight based on clinical urgency"""
        weights = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}
        return weights.get(urgency, 0.5)

    def calibrate_probabilities_platt(
        self,
        condition_code: str,
        raw_scores: np.ndarray[np.floating, np.dtype[np.floating]],
        true_labels: np.ndarray[np.integer, np.dtype[np.integer]],
    ) -> np.ndarray:
        """
        Apply Platt scaling to calibrate probabilities
        Fits sigmoid function to map raw scores to calibrated probabilities
        """

        try:
            if LogisticRegression is None:
                raise ImportError("scikit-learn is required for Platt scaling")
            if condition_code not in self.platt_scalers:
                self.platt_scalers[condition_code] = LogisticRegression()

            scaler = self.platt_scalers[condition_code]

            scores_reshaped = raw_scores.reshape(-1, 1)

            scaler.fit(scores_reshaped, true_labels)

            calibrated_probs = scaler.predict_proba(scores_reshaped)[:, 1]

            config = self.threshold_configs.get(condition_code)
            if config:
                config.platt_scaling_params = {
                    "coef": float(scaler.coef_[0][0]),
                    "intercept": float(scaler.intercept_[0]),
                }

            logger.info(
                f"Platt scaling calibrated for {condition_code}: "
                f"coef={scaler.coef_[0][0]:.3f}, intercept={scaler.intercept_[0]:.3f}"
            )

            return calibrated_probs

        except Exception as e:
            logger.error(f"Error in Platt scaling for {condition_code}: {e}")
            return raw_scores

    def apply_platt_scaling(self, condition_code: str, raw_score: float) -> float:
        """Apply existing Platt scaling to a single raw score"""

        if condition_code not in self.platt_scalers:
            logger.warning(f"No Platt scaler found for {condition_code}")
            return raw_score

        try:
            scaler = self.platt_scalers[condition_code]
            score_reshaped = np.array([[raw_score]])
            calibrated_prob = scaler.predict_proba(score_reshaped)[0, 1]
            return float(calibrated_prob)

        except Exception as e:
            logger.error(f"Error applying Platt scaling for {condition_code}: {e}")
            return raw_score

    def optimize_threshold_roc(
        self,
        condition_code: str,
        probabilities: np.ndarray[np.floating, np.dtype[np.floating]],
        true_labels: np.ndarray[np.integer, np.dtype[np.integer]],
        optimization_metric: str = "youden",
    ) -> float:
        """
        Optimize threshold using ROC analysis
        Supports different optimization metrics: youden, f1, balanced_accuracy
        """

        from sklearn.metrics import precision_recall_curve, roc_curve

        try:
            fpr, tpr, thresholds = roc_curve(true_labels, probabilities)

            if optimization_metric == "youden":
                youden_scores = tpr - fpr
                optimal_idx = np.argmax(youden_scores)
                optimal_threshold = thresholds[optimal_idx]

            elif optimization_metric == "f1":
                precision, recall, pr_thresholds = precision_recall_curve(
                    true_labels, probabilities
                )
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = (
                    pr_thresholds[optimal_idx]
                    if optimal_idx < len(pr_thresholds)
                    else 0.5
                )

            elif optimization_metric == "balanced_accuracy":
                balanced_acc = (tpr + (1 - fpr)) / 2
                optimal_idx = np.argmax(balanced_acc)
                optimal_threshold = thresholds[optimal_idx]

            else:
                logger.warning(
                    f"Unknown optimization metric: {optimization_metric}, using youden"
                )
                youden_scores = tpr - fpr
                optimal_idx = np.argmax(youden_scores)
                optimal_threshold = thresholds[optimal_idx]

            if condition_code in self.threshold_configs:
                self.threshold_configs[condition_code].base_threshold = float(
                    optimal_threshold
                )

            logger.info(
                f"Optimized threshold for {condition_code} using {optimization_metric}: "
                f"{optimal_threshold:.3f}"
            )

            return float(optimal_threshold)

        except Exception as e:
            logger.error(f"Error optimizing threshold for {condition_code}: {e}")
            return self.threshold_configs.get(
                condition_code, ThresholdConfig("", 0.8, 0.8, 0.8, "low")
            ).base_threshold

    def update_performance_metrics(
        self,
        condition_code: str,
        sensitivity: float,
        specificity: float,
        precision: float,
        f1_score: float,
        auc_roc: float | None = None,
    ) -> None:
        """Update performance metrics for a condition"""

        if condition_code not in self.threshold_configs:
            logger.warning(f"No threshold config found for {condition_code}")
            return

        config = self.threshold_configs[condition_code]

        performance_record = {
            "timestamp": np.datetime64("now").astype(str),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1_score,
            "auc_roc": auc_roc,
        }

        if config.performance_history is None:
            config.performance_history = []

        config.performance_history.append(performance_record)

        if len(config.performance_history) > 50:
            config.performance_history = config.performance_history[-50:]

        logger.debug(
            f"Updated performance metrics for {condition_code}: "
            f"sens={sensitivity:.3f}, spec={specificity:.3f}, f1={f1_score:.3f}"
        )

    def get_threshold_recommendations(self, condition_code: str) -> dict[str, Any]:
        """Get threshold recommendations based on current performance"""

        if condition_code not in self.threshold_configs:
            return {"error": f"No configuration found for {condition_code}"}

        config = self.threshold_configs[condition_code]
        current_threshold = config.base_threshold

        recommendations = {
            "current_threshold": current_threshold,
            "condition_code": condition_code,
            "clinical_urgency": config.clinical_urgency,
            "targets": {
                "sensitivity": config.sensitivity_target,
                "specificity": config.specificity_target,
            },
        }

        if config.performance_history and len(config.performance_history) >= 3:
            recent_performance = config.performance_history[-5:]
            avg_sensitivity = np.mean([p["sensitivity"] for p in recent_performance])
            avg_specificity = np.mean([p["specificity"] for p in recent_performance])

            recommendations["recent_performance"] = {
                "avg_sensitivity": avg_sensitivity,
                "avg_specificity": avg_specificity,
                "sensitivity_gap": config.sensitivity_target - avg_sensitivity,
                "specificity_gap": config.specificity_target - avg_specificity,
            }

            suggestions = []

            if avg_sensitivity < config.sensitivity_target - 0.05:
                suggestions.append(
                    f"Consider lowering threshold to improve sensitivity "
                    f"(current: {avg_sensitivity:.3f}, target: {config.sensitivity_target:.3f})"
                )

            if avg_specificity < config.specificity_target - 0.05:
                suggestions.append(
                    f"Consider raising threshold to improve specificity "
                    f"(current: {avg_specificity:.3f}, target: {config.specificity_target:.3f})"
                )

            if not suggestions:
                suggestions.append(
                    "Performance is meeting targets - current threshold is appropriate"
                )

            recommendations["suggestions"] = suggestions

        return recommendations

    def _save_configuration(self) -> None:
        """Save threshold configurations to file"""

        try:
            config_data = {}

            for condition_code, config in self.threshold_configs.items():
                config_data[condition_code] = {
                    "base_threshold": config.base_threshold,
                    "sensitivity_target": config.sensitivity_target,
                    "specificity_target": config.specificity_target,
                    "clinical_urgency": config.clinical_urgency,
                    "platt_scaling_params": config.platt_scaling_params,
                    "performance_history": (
                        config.performance_history[-10:]
                        if config.performance_history
                        else []
                    ),
                }

            with open(self.config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Saved threshold configuration to {self.config_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def _load_configuration(self) -> None:
        """Load threshold configurations from file"""

        try:
            if not Path(self.config_path).exists():
                logger.info(
                    f"No existing configuration file found at {self.config_path}"
                )
                return

            with open(self.config_path) as f:
                config_data = json.load(f)

            for condition_code, data in config_data.items():
                if condition_code in self.threshold_configs:
                    config = self.threshold_configs[condition_code]
                    config.base_threshold = data.get(
                        "base_threshold", config.base_threshold
                    )
                    config.platt_scaling_params = data.get("platt_scaling_params")
                    config.performance_history = data.get("performance_history", [])

            logger.info(f"Loaded threshold configuration from {self.config_path}")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def save_state(self) -> None:
        """Save current state including configurations and scalers"""

        self._save_configuration()

        scalers_path = self.config_path.replace(".json", "_scalers.pkl")
        try:
            with open(scalers_path, "wb") as f:
                pickle.dump(self.platt_scalers, f)
            logger.info(f"Saved Platt scalers to {scalers_path}")
        except Exception as e:
            logger.error(f"Error saving Platt scalers: {e}")

    def load_state(self) -> None:
        """Load saved state including configurations and scalers"""

        self._load_configuration()

        scalers_path = self.config_path.replace(".json", "_scalers.pkl")
        try:
            if Path(scalers_path).exists():
                with open(scalers_path, "rb") as f:
                    self.platt_scalers = pickle.load(f)
                logger.info(f"Loaded Platt scalers from {scalers_path}")
        except Exception as e:
            logger.error(f"Error loading Platt scalers: {e}")

    def get_all_thresholds(self) -> dict[str, float]:
        """Get all current thresholds for all conditions"""

        return {
            condition_code: config.base_threshold
            for condition_code, config in self.threshold_configs.items()
        }

    def reset_condition_threshold(self, condition_code: str) -> None:
        """Reset a condition's threshold to default value"""

        if condition_code not in self.threshold_configs:
            logger.warning(f"No configuration found for {condition_code}")
            return

        config = self.threshold_configs[condition_code]
        urgency = config.clinical_urgency

        default_thresholds = {
            "critical": 0.65,
            "high": 0.70,
            "medium": 0.75,
            "low": 0.80,
        }

        config.base_threshold = default_thresholds.get(urgency, 0.75)
        config.performance_history = []
        config.platt_scaling_params = None

        if condition_code in self.platt_scalers:
            del self.platt_scalers[condition_code]

        logger.info(f"Reset threshold for {condition_code} to {config.base_threshold}")
