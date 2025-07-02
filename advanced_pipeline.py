"""
Advanced ECG Preprocessing Pipeline - Phase 1 Implementation
Based on scientific recommendations for improved diagnostic precision.
"""

import logging
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pywt  # type: ignore
from scipy import signal

from app.preprocessing.adaptive_filters import create_adaptive_filter

logger = logging.getLogger(__name__)


class AdvancedECGPreprocessor:
    """
    Advanced ECG preprocessing pipeline implementing state-of-the-art techniques
    for improved diagnostic precision according to Phase 1 specifications.
    """

    def __init__(self, sampling_rate: int = 500) -> None:
        if sampling_rate < 500:
            raise ValueError(
                f"Taxa de amostragem {sampling_rate} Hz é inadequada! "
                f"Mínimo: 500 Hz, Recomendado: 1000 Hz"
            )
        self.fs = sampling_rate
        self.quality_threshold = 0.7

        self.adaptive_filter = create_adaptive_filter(
            adaptation_mode="rls",  # Use RLS for better performance with non-stationary noise
            filter_order=32,
            rls_forgetting_factor=0.99,
        )

    def advanced_preprocessing_pipeline(
        self, ecg_signal: npt.NDArray[np.float64], clinical_mode: bool = True
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """
        Complete advanced preprocessing pipeline as specified in Phase 1.

        Args:
            ecg_signal: Raw ECG signal data
            clinical_mode: If True, use fixed window segmentation; else beat-by-beat

        Returns:
            Tuple of (processed_signal, quality_metrics)
        """
        start_time = time.time()

        filtered_signal = self._butterworth_bandpass_filter(
            ecg_signal, 0.05, 150, self.fs
        )

        denoised_signal = self._wavelet_artifact_removal(filtered_signal)

        try:
            adaptive_filtered_signal = (
                self.adaptive_filter.multi_channel_adaptive_filtering(denoised_signal)
            )
            logger.info("Applied adaptive filtering for non-stationary noise removal")
            denoised_signal = adaptive_filtered_signal
        except Exception as e:
            logger.warning(
                f"Adaptive filtering failed: {e}, using standard denoised signal"
            )

        r_peaks = self._pan_tompkins_detector(denoised_signal, self.fs)

        if clinical_mode:
            segments = self._fixed_window_segmentation(
                denoised_signal, window_seconds=10
            )
        else:
            segments = self._beat_by_beat_segmentation(
                denoised_signal, r_peaks, samples_per_beat=400
            )

        normalized_signal = self._robust_normalize(segments, method="median_iqr")

        quality_score = self._assess_signal_quality_realtime(normalized_signal)

        processing_time = time.time() - start_time

        quality_metrics = {
            "quality_score": quality_score,
            "r_peaks_detected": len(r_peaks),
            "processing_time_ms": processing_time * 1000,
            "segments_created": len(segments) if isinstance(segments, list) else 1,
            "meets_quality_threshold": quality_score >= self.quality_threshold,
        }

        logger.info(
            f"Advanced preprocessing completed: quality={quality_score:.3f}, "
            f"r_peaks={len(r_peaks)}, time={processing_time*1000:.1f}ms"
        )

        return normalized_signal, quality_metrics

    def _butterworth_bandpass_filter(
        self,
        signal_data: npt.NDArray[np.float64],
        low_freq: float,
        high_freq: float,
        fs: int,
    ) -> npt.NDArray[np.float64]:
        """
        Apply Butterworth bandpass filter (0.05-150 Hz) for diagnostic mode.
        CRITICAL: Uses 0.05 Hz high-pass to preserve ST segment morphology.
        """
        try:
            nyquist = fs / 2
            low = low_freq / nyquist
            high = high_freq / nyquist

            b, a = signal.butter(4, [low, high], btype="band")

            if signal_data.ndim == 1:
                return np.array(signal.filtfilt(b, a, signal_data), dtype=np.float64)
            else:
                filtered = np.zeros_like(signal_data)
                for i in range(signal_data.shape[1]):
                    filtered[:, i] = signal.filtfilt(b, a, signal_data[:, i])
                return filtered.astype(np.float64)

        except Exception as e:
            logger.warning(f"Butterworth filter failed: {e}")
            return signal_data

    def _wavelet_artifact_removal(
        self, signal_data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Remove artifacts using Wavelet Daubechies db6 with adaptive thresholding.
        """
        try:
            if signal_data.ndim == 1:
                coeffs = pywt.wavedec(signal_data, "db6", level=9)

                coeffs_thresh = []
                for i, c in enumerate(coeffs):
                    if i == 0:  # Keep approximation coefficients
                        coeffs_thresh.append(c)
                    else:
                        sigma = np.median(np.abs(c)) / 0.6745
                        threshold = sigma * np.sqrt(2 * np.log(len(c)))
                        coeffs_thresh.append(pywt.threshold(c, threshold, mode="soft"))

                return np.array(
                    pywt.waverec(coeffs_thresh, "db6")[: len(signal_data)],
                    dtype=np.float64,
                )

            else:
                processed = np.zeros_like(signal_data)
                for i in range(signal_data.shape[1]):
                    processed[:, i] = self._wavelet_artifact_removal(signal_data[:, i])
                return processed

        except Exception as e:
            logger.warning(f"Wavelet denoising failed: {e}")
            return signal_data

    def _pan_tompkins_detector(
        self, signal_data: npt.NDArray[np.float64], fs: int
    ) -> list[int]:
        """
        Enhanced Pan-Tompkins algorithm for R-peak detection with >99.5% accuracy target.
        """
        try:
            if signal_data.ndim > 1:
                lead_idx = 1 if signal_data.shape[1] > 1 else 0
                signal_1d = signal_data[:, lead_idx]
            else:
                signal_1d = signal_data

            nyquist = fs / 2
            low = 5 / nyquist
            high = 15 / nyquist
            b, a = signal.butter(2, [low, high], btype="band")
            filtered = signal.filtfilt(b, a, signal_1d)

            derivative = np.zeros_like(filtered)
            for i in range(2, len(filtered) - 2):
                derivative[i] = (
                    2 * filtered[i + 1]
                    + filtered[i + 2]
                    - filtered[i - 1]
                    - 2 * filtered[i - 2]
                ) / 8

            squared = derivative**2

            window_size = int(0.15 * fs)
            integrated = np.convolve(
                squared, np.ones(window_size) / window_size, mode="same"
            )

            initial_threshold = np.percentile(
                integrated, 75
            )  # Use 75th percentile instead of mean
            initial_peaks, properties = signal.find_peaks(
                integrated,
                height=initial_threshold,
                distance=int(0.2 * fs),  # 200ms minimum distance
                prominence=initial_threshold * 0.3,
            )

            if len(initial_peaks) < 3:
                initial_threshold = np.percentile(integrated, 60)
                initial_peaks, properties = signal.find_peaks(
                    integrated,
                    height=initial_threshold,
                    distance=int(0.15 * fs),  # 150ms minimum distance
                    prominence=initial_threshold * 0.2,
                )

            r_peaks = []
            search_window = int(0.08 * fs)  # 80ms search window (wider than before)

            for peak in initial_peaks:
                start = max(0, peak - search_window)
                end = min(len(filtered), peak + search_window)

                local_signal = filtered[start:end]
                if len(local_signal) > 0:
                    local_max_idx = np.argmax(np.abs(local_signal))
                    refined_peak = start + local_max_idx
                    r_peaks.append(refined_peak)

            if len(r_peaks) > 1:
                validated_peaks: list[int] = []

                for _, peak in enumerate(r_peaks):
                    is_duplicate = False
                    for validated_peak in validated_peaks:
                        if abs(peak - validated_peak) < int(0.1 * fs):  # 100ms
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        validated_peaks.append(peak)

                final_peaks: list[int] = []
                min_rr_samples = int(0.3 * fs)  # 300ms minimum RR interval

                for peak in sorted(validated_peaks):
                    if not final_peaks or (peak - final_peaks[-1]) >= min_rr_samples:
                        final_peaks.append(peak)

                if len(final_peaks) > 2:
                    peak_amplitudes = [abs(filtered[peak]) for peak in final_peaks]
                    median_amplitude = np.median(peak_amplitudes)
                    amplitude_threshold = median_amplitude * 0.3  # 30% of median

                    quality_peaks = []
                    for peak in final_peaks:
                        if abs(filtered[peak]) >= amplitude_threshold:
                            quality_peaks.append(peak)

                    if (
                        len(quality_peaks) >= len(final_peaks) * 0.7
                    ):  # Keep if at least 70% pass
                        final_peaks = quality_peaks

                return final_peaks

            return r_peaks

        except Exception as e:
            logger.warning(f"Enhanced Pan-Tompkins detection failed: {e}")
            return []

    def _fixed_window_segmentation(
        self, signal_data: npt.NDArray[np.float64], window_seconds: int = 10
    ) -> list[npt.NDArray[np.float64]]:
        """
        Fixed window segmentation for clinical mode.
        """
        try:
            window_samples = window_seconds * self.fs
            segments = []

            for start in range(0, len(signal_data), window_samples):
                end = min(start + window_samples, len(signal_data))
                segment = signal_data[start:end]
                if (
                    len(segment) >= window_samples // 2
                ):  # Keep segments with at least 50% data
                    segments.append(segment)

            return segments

        except Exception as e:
            logger.warning(f"Fixed window segmentation failed: {e}")
            return [signal_data]

    def _beat_by_beat_segmentation(
        self,
        signal_data: npt.NDArray[np.float64],
        r_peaks: list[int],
        samples_per_beat: int = 400,
    ) -> list[npt.NDArray[np.float64]]:
        """
        Beat-by-beat segmentation based on R-peak locations.
        """
        try:
            if len(r_peaks) < 2:
                return [signal_data]

            segments = []
            half_beat = samples_per_beat // 2

            for r_peak in r_peaks:
                start = max(0, r_peak - half_beat)
                end = min(len(signal_data), r_peak + half_beat)

                segment = signal_data[start:end]
                if len(segment) >= samples_per_beat // 2:
                    if len(segment) < samples_per_beat:
                        padded = np.zeros((samples_per_beat,) + segment.shape[1:])
                        padded[: len(segment)] = segment
                        segments.append(padded)
                    else:
                        segments.append(segment[:samples_per_beat])

            return segments

        except Exception as e:
            logger.warning(f"Beat-by-beat segmentation failed: {e}")
            return [signal_data]

    def _robust_normalize(
        self, segments: list[npt.NDArray[np.float64]], method: str = "median_iqr"
    ) -> npt.NDArray[np.float64]:
        """
        Robust normalization using median and IQR for outlier resistance.
        """
        try:
            if not segments:
                return np.array([])

            if len(segments) == 1:
                combined_signal = segments[0]
            else:
                combined_signal = np.concatenate(segments, axis=0)

            if method == "median_iqr":
                if combined_signal.ndim == 1:
                    median_val = np.median(combined_signal)
                    q75, q25 = np.percentile(combined_signal, [75, 25])
                    iqr = q75 - q25

                    if iqr > 1e-10:
                        normalized = (combined_signal - median_val) / iqr
                    else:
                        normalized = combined_signal - median_val
                else:
                    normalized = np.zeros_like(combined_signal)
                    for i in range(combined_signal.shape[1]):
                        lead_data = combined_signal[:, i]
                        median_val = np.median(lead_data)
                        q75, q25 = np.percentile(lead_data, [75, 25])
                        iqr = q75 - q25

                        if iqr > 1e-10:
                            normalized[:, i] = (lead_data - median_val) / iqr
                        else:
                            normalized[:, i] = lead_data - median_val

                result_multi: npt.NDArray[np.float64] = normalized.astype(np.float64)
                return result_multi

            else:
                normalized_signal = (combined_signal - np.mean(combined_signal)) / (
                    np.std(combined_signal) + 1e-10
                )
                result_single: npt.NDArray[np.float64] = normalized_signal.astype(
                    np.float64
                )
                return result_single

        except Exception as e:
            logger.warning(f"Robust normalization failed: {e}")
            if segments:
                return segments[0].astype(np.float64)
            else:
                return np.array([], dtype=np.float64)

    def _assess_signal_quality_realtime(
        self, signal_data: npt.NDArray[np.float64]
    ) -> float:
        """
        Real-time signal quality assessment with 95% efficiency target.
        """
        try:
            if len(signal_data) == 0:
                return 0.0

            quality_factors = []

            signal_power = np.var(signal_data)
            if signal_data.ndim == 1:
                noise_estimate = np.var(np.diff(signal_data))
            else:
                noise_estimate = np.mean(
                    [
                        np.var(np.diff(signal_data[:, i]))
                        for i in range(signal_data.shape[1])
                    ]
                )

            snr = signal_power / (noise_estimate + 1e-10)
            snr_score = min(float(snr / 100), 1.0)  # Normalize to 0-1
            quality_factors.append(snr_score)

            if signal_data.ndim == 1:
                baseline_var = np.var(signal_data)
            else:
                baseline_var = np.mean(
                    [np.var(signal_data[:, i]) for i in range(signal_data.shape[1])]
                )

            baseline_score = 1.0 / (1.0 + baseline_var)
            quality_factors.append(baseline_score)

            if signal_data.ndim == 1:
                amplitude_std = np.std(np.abs(signal_data))
                amplitude_mean = np.mean(np.abs(signal_data))
            else:
                amplitude_std = np.mean(
                    [
                        np.std(np.abs(signal_data[:, i]))
                        for i in range(signal_data.shape[1])
                    ]
                )
                amplitude_mean = np.mean(
                    [
                        np.mean(np.abs(signal_data[:, i]))
                        for i in range(signal_data.shape[1])
                    ]
                )

            amplitude_consistency = 1.0 - min(
                float(amplitude_std / (amplitude_mean + 1e-10)), 1.0
            )
            quality_factors.append(amplitude_consistency)

            if len(signal_data) >= 256:  # Minimum for meaningful FFT
                if signal_data.ndim == 1:
                    freqs, psd = signal.welch(
                        signal_data, fs=self.fs, nperseg=min(256, len(signal_data) // 4)
                    )
                else:
                    freqs, psd = signal.welch(
                        signal_data[:, 0],
                        fs=self.fs,
                        nperseg=min(256, len(signal_data) // 4),
                    )

                ecg_band_mask = (freqs >= 0.5) & (freqs <= 40)
                ecg_power = np.sum(psd[ecg_band_mask])
                total_power = np.sum(psd)

                freq_quality = ecg_power / (total_power + 1e-10)
                quality_factors.append(freq_quality)

            weights = (
                [0.3, 0.25, 0.25, 0.2]
                if len(quality_factors) == 4
                else [1 / len(quality_factors)] * len(quality_factors)
            )
            overall_quality = sum(
                w * f for w, f in zip(weights, quality_factors, strict=False)
            )

            return float(min(max(overall_quality, 0.0), 1.0))

        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5  # Default moderate quality
