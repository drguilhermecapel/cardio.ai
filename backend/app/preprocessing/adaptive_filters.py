"""
Adaptive Filtering Implementation for ECG Signal Processing
Implements LMS and RLS adaptive filters for non-stationary noise removal
Based on Phase 1 optimization specifications for CardioAI Pro
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveFilterConfig:
    """Configuration for adaptive filters"""

    lms_step_size: float = 0.01  # Learning rate for LMS
    rls_forgetting_factor: float = 0.99  # Forgetting factor for RLS
    filter_order: int = 32  # Filter order
    regularization: float = 1e-6  # Regularization parameter
    adaptation_mode: str = "lms"  # "lms" or "rls"


class LMSAdaptiveFilter:
    """
    Least Mean Squares (LMS) Adaptive Filter
    Suitable for stationary and slowly varying non-stationary environments
    """

    def __init__(self, filter_order: int = 32, step_size: float = 0.01):
        self.filter_order = filter_order
        self.step_size = step_size
        self.weights = np.zeros(filter_order)
        self.input_buffer = np.zeros(filter_order)

    def filter_signal(
        self,
        signal: npt.NDArray[np.float64],
        reference_noise: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Apply LMS adaptive filtering to remove non-stationary noise

        Args:
            signal: Input ECG signal
            reference_noise: Reference noise signal (if available)

        Returns:
            Filtered ECG signal
        """
        if reference_noise is None:
            reference_noise = np.concatenate([np.zeros(10), signal[:-10]])

        filtered_signal = np.zeros_like(signal)

        for i in range(len(signal)):
            if i < self.filter_order:
                self.input_buffer[i:] = reference_noise[: self.filter_order - i]
                if i > 0:
                    self.input_buffer[:i] = reference_noise[
                        max(0, i - self.filter_order) : i
                    ]
            else:
                self.input_buffer = reference_noise[i - self.filter_order : i]

            filter_output = np.dot(self.weights, self.input_buffer)

            error = signal[i] - filter_output
            filtered_signal[i] = error

            self.weights += self.step_size * error * self.input_buffer

        return filtered_signal

    def reset(self):
        """Reset filter state"""
        self.weights.fill(0.0)
        self.input_buffer.fill(0.0)


class RLSAdaptiveFilter:
    """
    Recursive Least Squares (RLS) Adaptive Filter
    Superior performance for rapidly changing non-stationary environments
    """

    def __init__(
        self,
        filter_order: int = 32,
        forgetting_factor: float = 0.99,
        regularization: float = 1e-6,
    ):
        self.filter_order = filter_order
        self.forgetting_factor = forgetting_factor
        self.regularization = regularization

        self.weights = np.zeros(filter_order)
        self.P = np.eye(filter_order) / regularization  # Inverse correlation matrix
        self.input_buffer = np.zeros(filter_order)

    def filter_signal(
        self,
        signal: npt.NDArray[np.float64],
        reference_noise: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Apply RLS adaptive filtering to remove non-stationary noise

        Args:
            signal: Input ECG signal
            reference_noise: Reference noise signal (if available)

        Returns:
            Filtered ECG signal
        """
        if reference_noise is None:
            reference_noise = np.concatenate([np.zeros(15), signal[:-15]])

        filtered_signal = np.zeros_like(signal)

        for i in range(len(signal)):
            if i < self.filter_order:
                self.input_buffer[i:] = reference_noise[: self.filter_order - i]
                if i > 0:
                    self.input_buffer[:i] = reference_noise[
                        max(0, i - self.filter_order) : i
                    ]
            else:
                self.input_buffer = reference_noise[i - self.filter_order : i]

            filter_output = np.dot(self.weights, self.input_buffer)

            error = signal[i] - filter_output
            filtered_signal[i] = error

            P_u = np.dot(self.P, self.input_buffer)
            denominator = self.forgetting_factor + np.dot(self.input_buffer, P_u)
            gain = P_u / denominator

            self.weights += gain * error

            self.P = (self.P - np.outer(gain, P_u)) / self.forgetting_factor

        return filtered_signal

    def reset(self):
        """Reset filter state"""
        self.weights.fill(0.0)
        self.P = np.eye(self.filter_order) / self.regularization
        self.input_buffer.fill(0.0)


class AdaptiveECGFilter:
    """
    Comprehensive adaptive filtering system for ECG signals
    Combines multiple adaptive techniques for optimal noise removal
    """

    def __init__(self, config: AdaptiveFilterConfig | None = None):
        self.config = config or AdaptiveFilterConfig()

        self.lms_filter = LMSAdaptiveFilter(
            filter_order=self.config.filter_order, step_size=self.config.lms_step_size
        )

        self.rls_filter = RLSAdaptiveFilter(
            filter_order=self.config.filter_order,
            forgetting_factor=self.config.rls_forgetting_factor,
            regularization=self.config.regularization,
        )

    def adaptive_filtering(
        self, signal: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Apply adaptive filtering for non-stationary noise removal

        Args:
            signal: Input ECG signal

        Returns:
            Adaptively filtered ECG signal
        """
        try:
            noise_characteristics = self._assess_noise_characteristics(signal)

            if (
                self.config.adaptation_mode == "rls"
                or noise_characteristics["rapidly_changing"]
            ):
                logger.info("Using RLS adaptive filter for rapidly changing noise")
                filtered_signal = self.rls_filter.filter_signal(signal)
            else:
                logger.info(
                    "Using LMS adaptive filter for stationary/slowly varying noise"
                )
                filtered_signal = self.lms_filter.filter_signal(signal)

            improvement_ratio = self._assess_filtering_quality(signal, filtered_signal)
            logger.info(
                f"Adaptive filtering improvement ratio: {improvement_ratio:.3f}"
            )

            return filtered_signal

        except Exception as e:
            logger.warning(f"Adaptive filtering failed: {e}, returning original signal")
            return signal

    def multi_channel_adaptive_filtering(
        self, signal: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Apply adaptive filtering to multi-channel ECG signals

        Args:
            signal: Multi-channel ECG signal (samples x channels)

        Returns:
            Adaptively filtered multi-channel ECG signal
        """
        if signal.ndim == 1:
            return self.adaptive_filtering(signal)

        filtered_signal = np.zeros_like(signal)

        for channel in range(signal.shape[1]):
            self.lms_filter.reset()
            self.rls_filter.reset()

            filtered_signal[:, channel] = self.adaptive_filtering(signal[:, channel])

        return filtered_signal

    def _assess_noise_characteristics(self, signal: npt.NDArray[np.float64]) -> dict:
        """
        Assess noise characteristics to determine optimal filtering strategy

        Args:
            signal: Input ECG signal

        Returns:
            Dictionary with noise characteristics
        """
        window_size = min(500, len(signal) // 10)  # 1-second windows at 500Hz
        num_windows = len(signal) // window_size

        if num_windows < 2:
            return {"rapidly_changing": False, "variance_ratio": 1.0}

        variances = []
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(signal))
            window_var = np.var(signal[start_idx:end_idx])
            variances.append(window_var)

        variance_std = np.std(variances)
        variance_mean = np.mean(variances)
        variance_ratio = variance_std / (variance_mean + 1e-10)

        rapidly_changing = variance_ratio > 0.5

        return {
            "rapidly_changing": rapidly_changing,
            "variance_ratio": variance_ratio,
            "mean_variance": variance_mean,
        }

    def _assess_filtering_quality(
        self, original: npt.NDArray[np.float64], filtered: npt.NDArray[np.float64]
    ) -> float:
        """
        Assess the quality improvement from adaptive filtering

        Args:
            original: Original signal
            filtered: Filtered signal

        Returns:
            Quality improvement ratio
        """
        try:
            original_hf_power = np.mean(np.abs(np.diff(original, n=2)) ** 2)
            filtered_hf_power = np.mean(np.abs(np.diff(filtered, n=2)) ** 2)

            if original_hf_power > 0:
                noise_reduction = filtered_hf_power / original_hf_power
                improvement_ratio = 1.0 / (noise_reduction + 1e-10)
            else:
                improvement_ratio = 1.0

            return float(min(improvement_ratio, 10.0))  # Cap at 10x improvement

        except Exception:
            return 1.0


def create_adaptive_filter(adaptation_mode: str = "lms", **kwargs) -> AdaptiveECGFilter:
    """
    Factory function to create adaptive ECG filter

    Args:
        adaptation_mode: "lms" or "rls"
        **kwargs: Additional configuration parameters

    Returns:
        Configured adaptive ECG filter
    """
    config = AdaptiveFilterConfig(adaptation_mode=adaptation_mode, **kwargs)
    return AdaptiveECGFilter(config)


if __name__ == "__main__":
    fs = 500  # Sampling frequency
    t = np.arange(0, 10, 1 / fs)  # 10 seconds

    ecg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)

    noise_freq = 50 + 10 * np.sin(2 * np.pi * 0.1 * t)  # Time-varying frequency
    noise = 0.3 * np.sin(2 * np.pi * noise_freq * t)

    noisy_signal = ecg_signal + noise

    adaptive_filter = create_adaptive_filter(adaptation_mode="rls")
    filtered_signal = adaptive_filter.adaptive_filtering(noisy_signal)

    print(f"Original signal power: {np.mean(ecg_signal**2):.4f}")
    print(f"Noise power: {np.mean(noise**2):.4f}")
    print(f"Filtered signal power: {np.mean(filtered_signal**2):.4f}")
    print(
        f"SNR improvement: {10*np.log10(np.mean((ecg_signal - filtered_signal)**2) / np.mean((ecg_signal - noisy_signal)**2)):.2f} dB"
    )
