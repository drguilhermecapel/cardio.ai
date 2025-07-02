"""
Enhanced Signal Quality Analyzer for Advanced ECG Preprocessing
Implements real-time quality assessment with 95% efficiency target.
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class EnhancedSignalQualityAnalyzer:
    """
    Enhanced signal quality analyzer with real-time assessment capabilities
    and adaptive filtering recommendations.
    """

    def __init__(self, sampling_rate: int = 360) -> None:
        self.fs = sampling_rate
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "acceptable": 0.5,
            "poor": 0.3,
        }

    def assess_signal_quality_comprehensive(
        self, ecg_signal: npt.NDArray[np.float64]
    ) -> dict[str, Any]:
        """
        Comprehensive signal quality assessment with detailed metrics.

        Args:
            ecg_signal: ECG signal data (samples x leads)

        Returns:
            Dictionary with comprehensive quality metrics
        """
        try:
            quality_metrics: dict[str, Any] = {
                "overall_score": 0.0,
                "lead_scores": [],
                "noise_characteristics": {},
                "artifact_detection": {},
                "frequency_analysis": {},
                "morphology_assessment": {},
                "recommendations": [],
            }

            if ecg_signal.ndim == 1:
                ecg_signal = ecg_signal.reshape(-1, 1)

            lead_scores: list[float] = []
            for lead_idx in range(ecg_signal.shape[1]):
                lead_data = ecg_signal[:, lead_idx]
                lead_quality = self._analyze_lead_comprehensive(lead_data, lead_idx)
                lead_scores.append(float(lead_quality["score"]))
                quality_metrics["lead_scores"].append(lead_quality)

            quality_metrics["overall_score"] = np.mean(lead_scores)

            quality_metrics["noise_characteristics"] = self._characterize_noise(
                ecg_signal
            )

            quality_metrics["artifact_detection"] = self._detect_artifacts(ecg_signal)

            quality_metrics["frequency_analysis"] = self._analyze_frequency_domain(
                ecg_signal
            )

            quality_metrics["morphology_assessment"] = self._assess_morphology(
                ecg_signal
            )

            quality_metrics["recommendations"] = self._generate_quality_recommendations(
                quality_metrics
            )

            return quality_metrics

        except Exception as e:
            logger.error(f"Comprehensive quality assessment failed: {e}")
            return self._get_default_quality_metrics()

    def _analyze_lead_comprehensive(
        self, lead_data: npt.NDArray[np.float64], lead_idx: int
    ) -> dict[str, Any]:
        """
        Comprehensive analysis of a single ECG lead.
        """
        try:
            lead_quality = {
                "lead_index": lead_idx,
                "score": 0.0,
                "snr_db": 0.0,
                "baseline_stability": 0.0,
                "amplitude_consistency": 0.0,
                "frequency_quality": 0.0,
                "artifacts": [],
                "issues": [],
            }

            signal_power = np.var(lead_data)
            noise_estimate = np.var(np.diff(lead_data))
            snr = signal_power / (noise_estimate + 1e-10)
            snr_db = 10 * np.log10(snr + 1e-10)
            lead_quality["snr_db"] = float(snr_db)

            baseline_var = np.var(lead_data)
            baseline_stability = 1.0 / (1.0 + baseline_var)
            lead_quality["baseline_stability"] = float(baseline_stability)

            amplitude_std = np.std(np.abs(lead_data))
            amplitude_mean = np.mean(np.abs(lead_data))
            amplitude_consistency = 1.0 - min(
                amplitude_std / (amplitude_mean + 1e-10), 1.0
            )
            lead_quality["amplitude_consistency"] = float(amplitude_consistency)

            if len(lead_data) >= 256:
                freqs, psd = signal.welch(
                    lead_data, fs=self.fs, nperseg=min(256, len(lead_data) // 4)
                )
                ecg_band_mask = (freqs >= 0.5) & (freqs <= 40)
                ecg_power = np.sum(psd[ecg_band_mask])
                total_power = np.sum(psd)
                freq_quality = ecg_power / (total_power + 1e-10)
                lead_quality["frequency_quality"] = float(freq_quality)

            artifacts, issues = self._detect_lead_artifacts(lead_data)
            lead_quality["artifacts"] = artifacts
            lead_quality["issues"] = issues

            frequency_quality = lead_quality.get("frequency_quality", 0.5)
            if not isinstance(frequency_quality, int | float):
                frequency_quality = 0.5

            score_components = [
                min(snr_db / 20, 1.0) * 0.3,  # SNR component (normalized to 20dB max)
                baseline_stability * 0.25,
                amplitude_consistency * 0.25,
                float(frequency_quality) * 0.2,
            ]

            lead_quality["score"] = float(sum(score_components))

            return lead_quality

        except Exception as e:
            logger.error(f"Lead analysis failed for lead {lead_idx}: {e}")
            return {"lead_index": lead_idx, "score": 0.5, "artifacts": [], "issues": []}

    def _characterize_noise(
        self, ecg_signal: npt.NDArray[np.float64]
    ) -> dict[str, Any]:
        """
        Characterize noise patterns in the ECG signal.
        """
        try:
            noise_characteristics = {
                "powerline_interference": 0.0,
                "baseline_wander": 0.0,
                "muscle_artifact": 0.0,
                "electrode_noise": 0.0,
                "dominant_noise_type": "none",
            }

            for lead_idx in range(ecg_signal.shape[1]):
                lead_data = ecg_signal[:, lead_idx]

                if len(lead_data) >= 512:
                    freqs, psd = signal.welch(lead_data, fs=self.fs, nperseg=512)

                    powerline_50_mask = (freqs >= 49) & (freqs <= 51)
                    powerline_60_mask = (freqs >= 59) & (freqs <= 61)
                    powerline_power = np.sum(psd[powerline_50_mask]) + np.sum(
                        psd[powerline_60_mask]
                    )
                    total_power = np.sum(psd)
                    noise_characteristics[
                        "powerline_interference"
                    ] += powerline_power / (total_power + 1e-10)

                    baseline_mask = freqs < 0.5
                    baseline_power = np.sum(psd[baseline_mask])
                    noise_characteristics["baseline_wander"] += baseline_power / (
                        total_power + 1e-10
                    )

                    muscle_mask = freqs > 40
                    muscle_power = np.sum(psd[muscle_mask])
                    noise_characteristics["muscle_artifact"] += muscle_power / (
                        total_power + 1e-10
                    )

                    electrode_mask = freqs > 100
                    electrode_power = np.sum(psd[electrode_mask])
                    noise_characteristics["electrode_noise"] += electrode_power / (
                        total_power + 1e-10
                    )

            num_leads = ecg_signal.shape[1]
            for key in [
                "powerline_interference",
                "baseline_wander",
                "muscle_artifact",
                "electrode_noise",
            ]:
                current_value = noise_characteristics[key]
                if isinstance(current_value, int | float):
                    noise_characteristics[key] = float(current_value / num_leads)
                else:
                    noise_characteristics[key] = 0.0

            noise_levels = {
                "powerline": noise_characteristics["powerline_interference"],
                "baseline": noise_characteristics["baseline_wander"],
                "muscle": noise_characteristics["muscle_artifact"],
                "electrode": noise_characteristics["electrode_noise"],
            }

            dominant_noise = max(
                noise_levels.items(),
                key=lambda x: float(x[1]) if isinstance(x[1], int | float) else 0.0,
            )
            dominant_value = dominant_noise[1]
            if (
                isinstance(dominant_value, int | float) and float(dominant_value) > 0.1
            ):  # Threshold for significant noise
                noise_characteristics["dominant_noise_type"] = dominant_noise[0]

            return noise_characteristics

        except Exception as e:
            logger.error(f"Noise characterization failed: {e}")
            return {"dominant_noise_type": "unknown"}

    def _detect_artifacts(self, ecg_signal: npt.NDArray[np.float64]) -> dict[str, Any]:
        """
        Detect various types of artifacts in the ECG signal.
        """
        try:
            artifacts = {
                "saturation_detected": False,
                "flat_line_segments": [],
                "sudden_jumps": [],
                "periodic_interference": False,
                "lead_disconnection": [],
            }

            for lead_idx in range(ecg_signal.shape[1]):
                lead_data = ecg_signal[:, lead_idx]

                max_val = np.max(np.abs(lead_data))
                if max_val > 10:  # Assuming mV units
                    artifacts["saturation_detected"] = True

                flat_segments = self._detect_flat_segments(lead_data)
                if flat_segments:
                    flat_line_segments = artifacts.get("flat_line_segments", [])
                    if isinstance(flat_line_segments, list):
                        flat_line_segments.extend(
                            [(lead_idx, seg) for seg in flat_segments]
                        )
                        artifacts["flat_line_segments"] = flat_line_segments

                jumps = self._detect_sudden_jumps(lead_data)
                if jumps:
                    sudden_jumps = artifacts.get("sudden_jumps", [])
                    if isinstance(sudden_jumps, list):
                        sudden_jumps.extend([(lead_idx, jump) for jump in jumps])
                        artifacts["sudden_jumps"] = sudden_jumps

                if np.std(lead_data) < 0.01:
                    lead_disconnection = artifacts.get("lead_disconnection", [])
                    if isinstance(lead_disconnection, list):
                        lead_disconnection.append(lead_idx)
                        artifacts["lead_disconnection"] = lead_disconnection

            artifacts["periodic_interference"] = self._detect_periodic_interference(
                ecg_signal
            )

            return artifacts

        except Exception as e:
            logger.error(f"Artifact detection failed: {e}")
            return {}

    def _detect_flat_segments(
        self, lead_data: npt.NDArray[np.float64]
    ) -> list[tuple[int, int]]:
        """
        Detect flat line segments in a lead.
        """
        try:
            threshold = 0.001  # Very small variation threshold
            min_length = int(0.1 * self.fs)  # 100ms minimum

            window_size = int(0.05 * self.fs)  # 50ms windows
            variances = []

            for i in range(0, len(lead_data) - window_size, window_size // 2):
                window = lead_data[i : i + window_size]
                variances.append(np.var(window))

            flat_mask = np.array(variances) < threshold
            flat_regions = []

            in_flat = False
            start_idx = 0

            for i, is_flat in enumerate(flat_mask):
                if is_flat and not in_flat:
                    start_idx = i
                    in_flat = True
                elif not is_flat and in_flat:
                    if i - start_idx >= min_length // (window_size // 2):
                        flat_regions.append(
                            (start_idx * window_size // 2, i * window_size // 2)
                        )
                    in_flat = False

            return flat_regions

        except Exception as e:
            logger.error(f"Flat segment detection failed: {e}")
            return []

    def _detect_sudden_jumps(self, lead_data: npt.NDArray[np.float64]) -> list[int]:
        """
        Detect sudden amplitude jumps in a lead.
        """
        try:
            diff = np.diff(lead_data)

            threshold = 5 * np.std(diff)

            jump_indices = np.where(np.abs(diff) > threshold)[0]

            return [int(idx) for idx in jump_indices.tolist()]

        except Exception as e:
            logger.error(f"Sudden jump detection failed: {e}")
            return []

    def _detect_periodic_interference(
        self, ecg_signal: npt.NDArray[np.float64]
    ) -> bool:
        """
        Detect periodic interference patterns.
        """
        try:
            for lead_idx in range(ecg_signal.shape[1]):
                lead_data = ecg_signal[:, lead_idx]

                if len(lead_data) >= 1024:
                    autocorr = np.correlate(lead_data, lead_data, mode="full")
                    autocorr = autocorr[len(autocorr) // 2 :]

                    peaks, _ = signal.find_peaks(
                        autocorr[1:], height=0.5 * np.max(autocorr)
                    )

                    if len(peaks) > 3:  # Multiple strong periodic components
                        return True

            return False

        except Exception as e:
            logger.error(f"Periodic interference detection failed: {e}")
            return False

    def _analyze_frequency_domain(
        self, ecg_signal: npt.NDArray[np.float64]
    ) -> dict[str, Any]:
        """
        Analyze frequency domain characteristics.
        """
        try:
            freq_analysis = {
                "spectral_entropy": 0.0,
                "dominant_frequency": 0.0,
                "bandwidth_90": 0.0,
                "ecg_band_power_ratio": 0.0,
            }

            spectral_entropies = []
            dominant_freqs = []
            bandwidths = []
            ecg_ratios = []

            for lead_idx in range(ecg_signal.shape[1]):
                lead_data = ecg_signal[:, lead_idx]

                if len(lead_data) >= 512:
                    freqs, psd = signal.welch(lead_data, fs=self.fs, nperseg=512)

                    psd_norm = psd / (np.sum(psd) + 1e-10)
                    spectral_entropy_val = entropy(psd_norm + 1e-10)
                    spectral_entropies.append(spectral_entropy_val)

                    dominant_idx = np.argmax(psd)
                    dominant_freqs.append(freqs[dominant_idx])

                    cumsum_psd = np.cumsum(psd)
                    total_power = cumsum_psd[-1]
                    idx_90 = np.where(cumsum_psd >= 0.9 * total_power)[0]
                    if len(idx_90) > 0:
                        bandwidths.append(freqs[idx_90[0]])

                    ecg_mask = (freqs >= 0.5) & (freqs <= 40)
                    ecg_power = np.sum(psd[ecg_mask])
                    total_power = np.sum(psd)
                    ecg_ratios.append(ecg_power / (total_power + 1e-10))

            if spectral_entropies:
                spectral_entropies_array = np.array(
                    spectral_entropies, dtype=np.float64
                )
                freq_analysis["spectral_entropy"] = float(
                    np.mean(spectral_entropies_array)
                )
            if dominant_freqs:
                freq_analysis["dominant_frequency"] = float(np.mean(dominant_freqs))
            if bandwidths:
                freq_analysis["bandwidth_90"] = float(np.mean(bandwidths))
            if ecg_ratios:
                freq_analysis["ecg_band_power_ratio"] = float(np.mean(ecg_ratios))

            return freq_analysis

        except Exception as e:
            logger.error(f"Frequency domain analysis failed: {e}")
            return {}

    def _assess_morphology(self, ecg_signal: npt.NDArray[np.float64]) -> dict[str, Any]:
        """
        Assess ECG morphology characteristics.
        """
        try:
            morphology = {
                "qrs_width_consistency": 0.0,
                "amplitude_variation": 0.0,
                "morphology_stability": 0.0,
            }

            for lead_idx in range(ecg_signal.shape[1]):
                lead_data = ecg_signal[:, lead_idx]

                amplitude_std = np.std(np.abs(lead_data))
                amplitude_mean = np.mean(np.abs(lead_data))
                amp_variation = amplitude_std / (amplitude_mean + 1e-10)
                morphology["amplitude_variation"] += amp_variation

                if len(lead_data) >= 256:
                    freqs, psd = signal.welch(lead_data, fs=self.fs, nperseg=256)
                    hf_mask = freqs > 40
                    hf_power = np.sum(psd[hf_mask])
                    total_power = np.sum(psd)
                    stability = 1.0 - (hf_power / (total_power + 1e-10))
                    morphology["morphology_stability"] += stability

            num_leads = ecg_signal.shape[1]
            morphology["amplitude_variation"] = float(
                morphology["amplitude_variation"] / num_leads
            )
            morphology["morphology_stability"] = float(
                morphology["morphology_stability"] / num_leads
            )

            return morphology

        except Exception as e:
            logger.error(f"Morphology assessment failed: {e}")
            return {}

    def _detect_lead_artifacts(
        self, lead_data: npt.NDArray[np.float64]
    ) -> tuple[list[str], list[str]]:
        """
        Detect artifacts in a single lead.
        """
        artifacts = []
        issues = []

        try:
            if np.max(np.abs(lead_data)) > 10:
                artifacts.append("saturation")
                issues.append("Signal saturation detected")

            if np.std(lead_data) < 0.01:
                artifacts.append("flat_line")
                issues.append("Possible electrode disconnection")

            if np.var(lead_data) > 5:
                artifacts.append("high_noise")
                issues.append("High noise level")

            return artifacts, issues

        except Exception as e:
            logger.error(f"Lead artifact detection failed: {e}")
            return [], []

    def _generate_quality_recommendations(
        self, quality_metrics: dict[str, Any]
    ) -> list[str]:
        """
        Generate recommendations based on quality assessment.
        """
        recommendations = []

        try:
            overall_score = quality_metrics.get("overall_score", 0.5)
            noise_char = quality_metrics.get("noise_characteristics", {})
            artifacts = quality_metrics.get("artifact_detection", {})

            if overall_score < self.quality_thresholds["poor"]:
                recommendations.append("Signal quality is poor - consider re-recording")
            elif overall_score < self.quality_thresholds["acceptable"]:
                recommendations.append(
                    "Signal quality is marginal - apply aggressive preprocessing"
                )
            elif overall_score < self.quality_thresholds["good"]:
                recommendations.append(
                    "Signal quality is acceptable - apply standard preprocessing"
                )

            dominant_noise = noise_char.get("dominant_noise_type", "none")
            if dominant_noise == "powerline":
                recommendations.append("Apply notch filter for powerline interference")
            elif dominant_noise == "baseline":
                recommendations.append("Apply high-pass filter for baseline wander")
            elif dominant_noise == "muscle":
                recommendations.append("Apply low-pass filter for muscle artifacts")
            elif dominant_noise == "electrode":
                recommendations.append("Check electrode connections")

            if artifacts.get("saturation_detected", False):
                recommendations.append("Reduce amplifier gain to prevent saturation")
            if artifacts.get("lead_disconnection"):
                recommendations.append(
                    "Check electrode connections for disconnected leads"
                )
            if artifacts.get("periodic_interference", False):
                recommendations.append(
                    "Identify and eliminate periodic interference source"
                )

            return recommendations

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Quality assessment incomplete"]

    def _get_default_quality_metrics(self) -> dict[str, Any]:
        """
        Return default quality metrics in case of failure.
        """
        return {
            "overall_score": 0.5,
            "lead_scores": [],
            "noise_characteristics": {"dominant_noise_type": "unknown"},
            "artifact_detection": {},
            "frequency_analysis": {},
            "morphology_assessment": {},
            "recommendations": ["Quality assessment failed - use with caution"],
        }


class EnhancedQualityAnalyzer(EnhancedSignalQualityAnalyzer):
    """Backward compatible alias for EnhancedSignalQualityAnalyzer."""

    pass
