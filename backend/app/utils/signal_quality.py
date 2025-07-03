"""
Signal quality analysis utilities.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SignalQualityAnalyzer:
    """Analyzer for ECG signal quality assessment."""

    async def analyze_quality(self, ecg_data: NDArray[np.float64]) -> dict[str, Any]:
        """Analyze ECG signal quality."""
        try:
            quality_metrics = {
                "overall_score": 0.0,
                "noise_level": 0.0,
                "baseline_wander": 0.0,
                "signal_to_noise_ratio": 0.0,
                "artifacts_detected": [],
                "quality_issues": [],
            }

            lead_scores = []

            for i in range(ecg_data.shape[1]):
                lead_data = ecg_data[:, i]
                lead_quality = await self._analyze_lead_quality(lead_data)
                lead_scores.append(lead_quality["score"])

                artifacts = lead_quality.get("artifacts", [])
                if isinstance(artifacts, list):
                    artifacts_detected = quality_metrics["artifacts_detected"]
                    if isinstance(artifacts_detected, list):
                        artifacts_detected.extend(artifacts)

                issues = lead_quality.get("issues", [])
                if isinstance(issues, list):
                    quality_issues = quality_metrics["quality_issues"]
                    if isinstance(quality_issues, list):
                        quality_issues.extend(issues)

            quality_metrics["overall_score"] = np.mean(lead_scores)
            quality_metrics["noise_level"] = await self._calculate_noise_level(ecg_data)
            quality_metrics["baseline_wander"] = await self._calculate_baseline_wander(
                ecg_data
            )
            quality_metrics["signal_to_noise_ratio"] = await self._calculate_snr(
                ecg_data
            )

            overall_score = quality_metrics["overall_score"]
            if isinstance(overall_score, int | float) and overall_score < 0.5:
                quality_issues = quality_metrics["quality_issues"]
                if isinstance(quality_issues, list):
                    quality_issues.append("Poor overall signal quality")

            noise_level = quality_metrics["noise_level"]
            if isinstance(noise_level, int | float) and noise_level > 0.3:
                quality_issues = quality_metrics["quality_issues"]
                if isinstance(quality_issues, list):
                    quality_issues.append("High noise level detected")

            baseline_wander = quality_metrics["baseline_wander"]
            if isinstance(baseline_wander, int | float) and baseline_wander > 0.2:
                quality_issues = quality_metrics["quality_issues"]
                if isinstance(quality_issues, list):
                    quality_issues.append("Significant baseline wander")

            snr = quality_metrics["signal_to_noise_ratio"]
            if isinstance(snr, int | float) and snr < 10:
                quality_issues = quality_metrics["quality_issues"]
                if isinstance(quality_issues, list):
                    quality_issues.append("Low signal-to-noise ratio")

            return quality_metrics

        except Exception as e:
            logger.error("Signal quality analysis failed: %s", str(e))
            return {
                "overall_score": 0.5,
                "noise_level": 0.0,
                "baseline_wander": 0.0,
                "signal_to_noise_ratio": 0.0,
                "artifacts_detected": [],
                "quality_issues": ["Quality analysis failed"],
            }

    async def _analyze_lead_quality(
        self, lead_data: NDArray[np.float64]
    ) -> dict[str, Any]:
        """Analyze quality of a single ECG lead."""
        try:
            quality = {
                "score": 0.0,
                "artifacts": [],
                "issues": [],
            }

            if np.std(lead_data) < 0.01:
                quality["score"] = 0.0
                artifacts_list = quality["artifacts"]
                if isinstance(artifacts_list, list):
                    artifacts_list.append("flat_line")
                issues_list = quality["issues"]
                if isinstance(issues_list, list):
                    issues_list.append("Possible electrode disconnection")
                return quality

            max_val = np.max(np.abs(lead_data))
            if max_val > 10:  # Assuming mV units
                artifacts_list = quality["artifacts"]
                if isinstance(artifacts_list, list):
                    artifacts_list.append("saturation")
                issues_list = quality["issues"]
                if isinstance(issues_list, list):
                    issues_list.append("Signal saturation detected")

            signal_variance = np.var(lead_data)
            if signal_variance > 5:
                artifacts_list = quality["artifacts"]
                if isinstance(artifacts_list, list):
                    artifacts_list.append("high_noise")
                issues_list = quality["issues"]
                if isinstance(issues_list, list):
                    issues_list.append("High noise level")

            score = 1.0

            if signal_variance > 2:
                score -= 0.3

            if max_val > 5:
                score -= 0.2

            if max_val < 0.1:
                score -= 0.2

            quality["score"] = max(0.0, score)

            return quality

        except Exception as e:
            logger.error("Lead quality analysis failed: %s", str(e))
            return {"score": 0.5, "artifacts": [], "issues": []}

    async def _calculate_noise_level(self, ecg_data: NDArray[np.float64]) -> float:
        """Calculate noise level in ECG signal."""
        try:
            noise_levels = []

            for i in range(ecg_data.shape[1]):
                lead_data = ecg_data[:, i]

                from scipy import signal

                frequencies, power = signal.welch(lead_data, fs=500, nperseg=1024)

                high_freq_mask = frequencies > 50
                noise_power = np.sum(power[high_freq_mask])
                total_power = np.sum(power)

                noise_ratio = noise_power / (total_power + 1e-8)
                noise_levels.append(noise_ratio)

            return float(np.mean(noise_levels))

        except Exception as e:
            logger.error("Noise level calculation failed: %s", str(e))
            return 0.1

    async def _calculate_baseline_wander(self, ecg_data: NDArray[np.float64]) -> float:
        """Calculate baseline wander in ECG signal."""
        try:
            wander_levels = []

            for i in range(ecg_data.shape[1]):
                lead_data = ecg_data[:, i]

                from scipy import signal

                frequencies, power = signal.welch(lead_data, fs=500, nperseg=1024)

                low_freq_mask = frequencies < 1
                wander_power = np.sum(power[low_freq_mask])
                total_power = np.sum(power)

                wander_ratio = wander_power / (total_power + 1e-8)
                wander_levels.append(wander_ratio)

            return float(np.mean(wander_levels))

        except Exception as e:
            logger.error("Baseline wander calculation failed: %s", str(e))
            return 0.1

    async def _calculate_snr(self, ecg_data: NDArray[np.float64]) -> float:
        """Calculate signal-to-noise ratio."""
        try:
            snr_values = []

            for i in range(ecg_data.shape[1]):
                lead_data = ecg_data[:, i]

                signal_power = np.var(lead_data)

                from scipy import signal

                b, a = signal.butter(4, 0.1, btype="high", fs=500)
                noise_estimate = signal.filtfilt(b, a, lead_data)
                noise_power = np.var(noise_estimate)

                snr = signal_power / (noise_power + 1e-8)
                snr_db = 10 * np.log10(snr + 1e-8)
                snr_values.append(snr_db)

            return float(np.mean(snr_values))

        except Exception as e:
            logger.error("SNR calculation failed: %s", str(e))
            return 20.0
