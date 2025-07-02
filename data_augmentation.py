"""
Comprehensive Data Augmentation for ECG Analysis
Implements time domain and frequency domain augmentation techniques
Based on scientific recommendations for CardioAI Pro
"""

import logging
import random
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq, ifft

from app.preprocessing.advanced_pipeline import AdvancedECGPreprocessor

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for ECG data augmentation"""

    amplitude_scale_range: tuple[float, float] = (0.8, 1.2)
    time_shift_range: int = 50  # ±50ms
    gaussian_noise_snr_db: float = 20.0  # SNR > 20dB
    baseline_wander_amplitude: float = 0.1

    spectral_masking_ratio: float = 0.1  # 10% of frequency bins
    phase_rotation_range: tuple[float, float] = (-np.pi / 4, np.pi / 4)
    heart_rate_perturbation_bpm: float = 10.0  # ±10 bpm

    qrs_width_variation: float = 0.1  # ±10%
    st_segment_shift: float = 0.05  # ±50µV
    t_wave_inversion_prob: float = 0.05

    amplitude_scale_prob: float = 0.5
    time_shift_prob: float = 0.3
    noise_addition_prob: float = 0.4
    baseline_wander_prob: float = 0.2
    spectral_masking_prob: float = 0.3
    phase_rotation_prob: float = 0.2
    heart_rate_perturbation_prob: float = 0.3

    min_quality_score: float = 0.7
    preserve_clinical_features: bool = True

    target_balance_ratios: dict[str, float] | None = (
        None  # Will be set in __post_init__
    )

    def __post_init__(self):
        if self.target_balance_ratios is None:
            self.target_balance_ratios = {
                "STEMI": 0.05,  # 0.4% → 5%
                "NSTEMI": 0.03,  # Increase NSTEMI representation
                "AFIB": 0.08,  # Increase atrial fibrillation
                "VT": 0.02,  # Increase ventricular tachycardia
                "VF": 0.01,  # Increase ventricular fibrillation
                "COMPLETE_HEART_BLOCK": 0.015,  # Increase complete heart block
                "LONG_QT": 0.02,  # Increase long QT syndrome
                "BRUGADA": 0.005,  # Increase Brugada syndrome
            }


class TimeDomainAugmenter:
    """Time domain augmentation techniques for ECG signals"""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def amplitude_scaling(
        self, signal: np.ndarray, scale_factor: float | None = None
    ) -> np.ndarray:
        """
        Apply amplitude scaling to ECG signal
        Range: 0.8-1.2x as specified in recommendations
        """
        if scale_factor is None:
            scale_factor = np.random.uniform(*self.config.amplitude_scale_range)

        scaled_signal = signal * scale_factor

        max_amplitude = np.max(np.abs(scaled_signal))
        if max_amplitude > 5.0:  # 5mV maximum
            scaled_signal = scaled_signal * (5.0 / max_amplitude)

        return scaled_signal

    def temporal_shift(
        self, signal: np.ndarray, shift_samples: int | None = None, fs: int = 500
    ) -> np.ndarray:
        """
        Apply temporal shift to ECG signal
        Range: ±50ms as specified in recommendations
        """
        if shift_samples is None:
            max_shift_samples = int(
                self.config.time_shift_range * fs / 1000
            )  # Convert ms to samples
            shift_samples = np.random.randint(-max_shift_samples, max_shift_samples + 1)

        if shift_samples == 0:
            return signal

        shifted_signal = np.zeros_like(signal)

        if shift_samples > 0:
            shifted_signal[shift_samples:] = signal[:-shift_samples]
            shifted_signal[:shift_samples] = signal[0]
        else:
            shifted_signal[:shift_samples] = signal[-shift_samples:]
            shifted_signal[shift_samples:] = signal[-1]

        return shifted_signal

    def add_gaussian_noise(
        self, signal: np.ndarray, snr_db: float | None = None
    ) -> np.ndarray:
        """
        Add calibrated Gaussian noise to ECG signal
        SNR > 20dB as specified in recommendations
        """
        if snr_db is None:
            snr_db = self.config.gaussian_noise_snr_db

        signal_power = np.mean(signal**2)

        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

        return signal + noise

    def add_baseline_wander(self, signal: np.ndarray, fs: int = 500) -> np.ndarray:
        """Add realistic baseline wander to ECG signal"""
        duration = len(signal) / fs
        t = np.linspace(0, duration, len(signal))

        baseline_wander = (
            self.config.baseline_wander_amplitude
            * 0.5
            * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz
            + self.config.baseline_wander_amplitude
            * 0.3
            * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz
            + self.config.baseline_wander_amplitude
            * 0.2
            * np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz
        )

        if len(signal.shape) == 2:
            baseline_wander = baseline_wander[:, np.newaxis]

        return signal + baseline_wander

    def add_powerline_interference(
        self, signal: np.ndarray, fs: int = 500, frequency: float = 60.0
    ) -> np.ndarray:
        """Add powerline interference (50/60 Hz)"""
        duration = len(signal) / fs
        t = np.linspace(0, duration, len(signal))

        interference = (
            0.02 * np.sin(2 * np.pi * frequency * t)  # Fundamental
            + 0.01 * np.sin(2 * np.pi * 2 * frequency * t)  # 2nd harmonic
            + 0.005 * np.sin(2 * np.pi * 3 * frequency * t)  # 3rd harmonic
        )

        if len(signal.shape) == 2:
            interference = interference[:, np.newaxis]

        return signal + interference

    def muscle_artifact_simulation(
        self, signal: np.ndarray, fs: int = 500
    ) -> np.ndarray:
        """Simulate muscle artifacts (EMG contamination)"""
        duration = len(signal) / fs
        t = np.linspace(0, duration, len(signal))

        muscle_activity = np.zeros_like(t)

        num_bursts = np.random.randint(1, 5)
        for _ in range(num_bursts):
            burst_start = np.random.randint(0, len(t) // 2)
            burst_duration = np.random.randint(fs // 10, fs // 2)  # 0.1-0.5 seconds
            burst_end = min(burst_start + burst_duration, len(t))

            burst_signal = np.random.normal(0, 0.05, burst_end - burst_start)

            envelope = np.exp(-np.linspace(0, 3, burst_end - burst_start))
            muscle_activity[burst_start:burst_end] = burst_signal * envelope

        if len(signal.shape) == 2:
            muscle_activity = muscle_activity[:, np.newaxis]

        return signal + muscle_activity


class FrequencyDomainAugmenter:
    """Frequency domain augmentation techniques for ECG signals"""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def spectral_masking(self, signal: np.ndarray, fs: int = 500) -> np.ndarray:
        """
        Apply selective spectral masking
        Masks random frequency components while preserving clinical information
        """
        signal_fft = fft(signal, axis=0)
        frequencies = fftfreq(len(signal), 1 / fs)

        critical_freq_ranges = [
            (0.5, 3.0),  # QRS complex
            (3.0, 8.0),  # T wave
            (8.0, 15.0),  # High-frequency QRS components
        ]

        np.ones_like(signal_fft, dtype=bool)

        non_critical_indices = []
        for i, freq in enumerate(frequencies):
            freq_abs = abs(freq)
            is_critical = any(
                low <= freq_abs <= high for low, high in critical_freq_ranges
            )
            if not is_critical and freq_abs > 0.1:  # Avoid DC component
                non_critical_indices.append(i)

        num_to_mask = int(
            len(non_critical_indices) * self.config.spectral_masking_ratio
        )
        indices_to_mask = np.random.choice(
            non_critical_indices, num_to_mask, replace=False
        )

        signal_fft_masked = np.array(signal_fft)
        signal_fft_masked[indices_to_mask] = 0

        augmented_signal = np.real(ifft(signal_fft_masked))

        return np.asarray(augmented_signal)

    def phase_rotation(
        self, signal: np.ndarray, rotation_angle: float | None = None
    ) -> np.ndarray:
        """
        Apply controlled phase rotation
        Rotates phase components while preserving magnitude spectrum
        """
        if rotation_angle is None:
            rotation_angle = np.random.uniform(*self.config.phase_rotation_range)

        signal_fft = fft(signal, axis=0)

        phase_shift = np.exp(1j * rotation_angle)
        signal_fft_rotated = signal_fft * phase_shift

        augmented_signal = np.real(ifft(signal_fft_rotated))

        return np.asarray(augmented_signal)

    def heart_rate_perturbation(
        self, signal: np.ndarray, fs: int = 500, target_hr_change: float | None = None
    ) -> np.ndarray:
        """
        Apply heart rate perturbation (±10 bpm)
        Changes the temporal scaling to simulate different heart rates
        """
        if target_hr_change is None:
            target_hr_change = np.random.uniform(
                -self.config.heart_rate_perturbation_bpm,
                self.config.heart_rate_perturbation_bpm,
            )

        current_hr = self._estimate_heart_rate(signal, fs)

        if current_hr <= 0:
            return signal  # Cannot estimate heart rate

        new_hr = current_hr + target_hr_change
        new_hr = np.clip(new_hr, 40, 200)  # Physiological limits

        scaling_factor = current_hr / new_hr

        new_length = int(len(signal) * scaling_factor)

        if len(signal.shape) == 1:
            augmented_signal = scipy_signal.resample(signal, new_length)

            if len(augmented_signal) > len(signal):
                augmented_signal = augmented_signal[: len(signal)]
            else:
                padding = np.zeros(len(signal) - len(augmented_signal))
                augmented_signal = np.concatenate([augmented_signal, padding])
        else:
            augmented_signal = np.zeros_like(signal)
            for lead in range(signal.shape[1]):
                resampled_lead = scipy_signal.resample(signal[:, lead], new_length)

                if len(resampled_lead) > len(signal):
                    augmented_signal[:, lead] = resampled_lead[: len(signal)]
                else:
                    augmented_signal[: len(resampled_lead), lead] = resampled_lead

        return np.asarray(augmented_signal)

    def _estimate_heart_rate(self, signal: np.ndarray, fs: int = 500) -> float:
        """Estimate heart rate from ECG signal"""
        try:
            if len(signal.shape) == 2:
                lead_signal = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
            else:
                lead_signal = signal

            nyquist = fs / 2
            low_cutoff = 5 / nyquist
            high_cutoff = 15 / nyquist

            b, a = scipy_signal.butter(4, [low_cutoff, high_cutoff], btype="band")
            filtered_signal = scipy_signal.filtfilt(b, a, lead_signal)

            peaks, _ = scipy_signal.find_peaks(
                filtered_signal,
                height=np.std(filtered_signal) * 2,
                distance=int(0.3 * fs),  # Minimum 300ms between peaks
            )

            if len(peaks) < 2:
                return 70.0  # Default heart rate

            rr_intervals = np.diff(peaks) / fs  # Convert to seconds

            heart_rate = 60.0 / np.mean(rr_intervals)

            return float(heart_rate)

        except Exception:
            return 70.0  # Default heart rate


class MorphologicalAugmenter:
    """Morphological augmentation techniques for ECG signals"""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def qrs_width_variation(self, signal: np.ndarray, fs: int = 500) -> np.ndarray:
        """Vary QRS complex width while preserving morphology"""
        try:
            qrs_locations = self._detect_qrs_complexes(signal, fs)

            if len(qrs_locations) == 0:
                return signal

            augmented_signal = signal.copy()

            for qrs_start, qrs_end in qrs_locations:
                variation_factor = 1.0 + np.random.uniform(
                    -self.config.qrs_width_variation, self.config.qrs_width_variation
                )

                qrs_complex = signal[qrs_start:qrs_end]

                new_width = int(len(qrs_complex) * variation_factor)
                new_width = max(new_width, 10)  # Minimum width

                if len(qrs_complex.shape) == 1:
                    resampled_qrs = scipy_signal.resample(qrs_complex, new_width)
                else:
                    resampled_qrs = np.zeros((new_width, qrs_complex.shape[1]))
                    for lead in range(qrs_complex.shape[1]):
                        resampled_qrs[:, lead] = scipy_signal.resample(
                            qrs_complex[:, lead], new_width
                        )

                if qrs_start + new_width <= len(signal):
                    augmented_signal[qrs_start : qrs_start + new_width] = resampled_qrs

            return augmented_signal

        except Exception as e:
            logger.warning(f"QRS width variation failed: {e}")
            return signal

    def st_segment_shift(self, signal: np.ndarray, fs: int = 500) -> np.ndarray:
        """Apply ST segment elevation/depression"""
        try:
            st_segments = self._detect_st_segments(signal, fs)

            if len(st_segments) == 0:
                return signal

            augmented_signal = signal.copy()

            for st_start, st_end in st_segments:
                st_shift = np.random.uniform(
                    -self.config.st_segment_shift, self.config.st_segment_shift
                )

                augmented_signal[st_start:st_end] += st_shift

            return augmented_signal

        except Exception as e:
            logger.warning(f"ST segment shift failed: {e}")
            return signal

    def _detect_qrs_complexes(
        self, signal: np.ndarray, fs: int = 500
    ) -> list[tuple[int, int]]:
        """Detect QRS complex locations"""
        try:
            if len(signal.shape) == 2:
                lead_signal = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
            else:
                lead_signal = signal

            diff_signal = np.diff(lead_signal)

            peaks, _ = scipy_signal.find_peaks(
                np.abs(diff_signal),
                height=np.std(diff_signal) * 3,
                distance=int(0.3 * fs),
            )

            qrs_half_width = int(0.06 * fs)  # 60ms half-width

            qrs_locations = []
            for peak in peaks:
                qrs_start = max(0, peak - qrs_half_width)
                qrs_end = min(len(signal), peak + qrs_half_width)
                qrs_locations.append((qrs_start, qrs_end))

            return qrs_locations

        except Exception:
            return []

    def _detect_st_segments(
        self, signal: np.ndarray, fs: int = 500
    ) -> list[tuple[int, int]]:
        """Detect ST segment locations"""
        try:
            qrs_locations = self._detect_qrs_complexes(signal, fs)

            st_segments = []

            for _qrs_start, qrs_end in qrs_locations:
                st_start = qrs_end
                st_end = min(len(signal), st_start + int(0.1 * fs))  # 100ms

                if st_end > st_start:
                    st_segments.append((st_start, st_end))

            return st_segments

        except Exception:
            return []


class ECGDataAugmenter:
    """
    Comprehensive ECG data augmentation system
    Combines time domain, frequency domain, and morphological augmentation
    """

    def __init__(self, config: AugmentationConfig | None = None):
        self.config = config or AugmentationConfig()

        self.time_augmenter = TimeDomainAugmenter(self.config)
        self.freq_augmenter = FrequencyDomainAugmenter(self.config)
        self.morph_augmenter = MorphologicalAugmenter(self.config)

        self.preprocessor = AdvancedECGPreprocessor()

    def augment_signal(
        self,
        signal: np.ndarray,
        condition_code: str = "NORM",
        fs: int = 500,
        augmentation_methods: list[str] | None = None,
    ) -> np.ndarray:
        """
        Apply comprehensive augmentation to a single ECG signal

        Args:
            signal: ECG signal (samples x leads)
            condition_code: Condition code for preservation of clinical features
            fs: Sampling frequency
            augmentation_methods: List of methods to apply (None = random selection)

        Returns:
            Augmented ECG signal
        """
        augmented_signal = signal.copy()

        available_methods = {
            "amplitude_scaling": (
                self.time_augmenter.amplitude_scaling,
                self.config.amplitude_scale_prob,
            ),
            "temporal_shift": (
                self.time_augmenter.temporal_shift,
                self.config.time_shift_prob,
            ),
            "gaussian_noise": (
                self.time_augmenter.add_gaussian_noise,
                self.config.noise_addition_prob,
            ),
            "baseline_wander": (
                self.time_augmenter.add_baseline_wander,
                self.config.baseline_wander_prob,
            ),
            "powerline_interference": (
                self.time_augmenter.add_powerline_interference,
                0.1,
            ),
            "muscle_artifacts": (self.time_augmenter.muscle_artifact_simulation, 0.1),
            "spectral_masking": (
                self.freq_augmenter.spectral_masking,
                self.config.spectral_masking_prob,
            ),
            "phase_rotation": (
                self.freq_augmenter.phase_rotation,
                self.config.phase_rotation_prob,
            ),
            "heart_rate_perturbation": (
                self.freq_augmenter.heart_rate_perturbation,
                self.config.heart_rate_perturbation_prob,
            ),
            "qrs_width_variation": (self.morph_augmenter.qrs_width_variation, 0.1),
            "st_segment_shift": (self.morph_augmenter.st_segment_shift, 0.1),
        }

        if augmentation_methods is None:
            methods_to_apply = []
            for method_name, (method_func, prob) in available_methods.items():
                if np.random.random() < prob:
                    methods_to_apply.append((method_name, method_func))
        else:
            methods_to_apply = [
                (method_name, available_methods[method_name][0])
                for method_name in augmentation_methods
                if method_name in available_methods
            ]

        for method_name, method_func in methods_to_apply:
            try:
                if method_name in [
                    "baseline_wander",
                    "powerline_interference",
                    "muscle_artifacts",
                    "spectral_masking",
                    "heart_rate_perturbation",
                    "qrs_width_variation",
                    "st_segment_shift",
                ]:
                    augmented_signal = method_func(augmented_signal, fs)
                else:
                    augmented_signal = method_func(augmented_signal)

                if self.config.preserve_clinical_features:
                    quality_score = self._assess_signal_quality(augmented_signal)
                    if quality_score < self.config.min_quality_score:
                        logger.debug(
                            f"Augmentation {method_name} rejected due to low quality: {quality_score}"
                        )
                        augmented_signal = signal.copy()  # Revert
                        break

            except Exception as e:
                logger.warning(f"Augmentation method {method_name} failed: {e}")
                continue

        return augmented_signal

    def _assess_signal_quality(self, signal: np.ndarray) -> float:
        """Assess signal quality after augmentation"""
        try:
            processed_signal, quality_metrics = (
                self.preprocessor.advanced_preprocessing_pipeline(signal)
            )
            return quality_metrics.get("quality_score", 0.0)
        except Exception:
            return 0.0

    def augment_dataset(
        self,
        signals: list[np.ndarray],
        condition_codes: list[str],
        augmentation_factor: int = 2,
        fs: int = 500,
    ) -> tuple[list[np.ndarray], list[str]]:
        """
        Augment entire dataset

        Args:
            signals: List of ECG signals
            condition_codes: List of condition codes
            augmentation_factor: Number of augmented versions per original signal
            fs: Sampling frequency

        Returns:
            Tuple of (augmented_signals, augmented_condition_codes)
        """
        logger.info(f"Augmenting dataset with factor {augmentation_factor}...")

        augmented_signals = signals.copy()
        augmented_condition_codes = condition_codes.copy()

        for i, (signal, condition_code) in enumerate(
            zip(signals, condition_codes, strict=False)
        ):
            for _aug_idx in range(
                augmentation_factor - 1
            ):  # -1 because original is already included
                try:
                    augmented_signal = self.augment_signal(signal, condition_code, fs)
                    augmented_signals.append(augmented_signal)
                    augmented_condition_codes.append(condition_code)
                except Exception as e:
                    logger.warning(f"Failed to augment signal {i}: {e}")
                    continue

        logger.info(
            f"Dataset augmented: {len(signals)} → {len(augmented_signals)} samples"
        )

        return augmented_signals, augmented_condition_codes

    def balance_rare_conditions(
        self,
        signals: list[np.ndarray],
        condition_codes: list[str],
        target_ratios: dict[str, float] | None = None,
        fs: int = 500,
    ) -> tuple[list[np.ndarray], list[str]]:
        """
        Balance rare conditions using data augmentation

        Args:
            signals: List of ECG signals
            condition_codes: List of condition codes
            target_ratios: Target ratios for each condition (None = use config defaults)
            fs: Sampling frequency

        Returns:
            Tuple of (balanced_signals, balanced_condition_codes)
        """
        if target_ratios is None:
            target_ratios = self.config.target_balance_ratios

        logger.info("Balancing rare conditions using data augmentation...")

        condition_counts = {}
        condition_indices = {}

        for i, condition_code in enumerate(condition_codes):
            if condition_code not in condition_counts:
                condition_counts[condition_code] = 0
                condition_indices[condition_code] = []
            condition_counts[condition_code] += 1
            condition_indices[condition_code].append(i)

        total_samples = len(signals)

        balanced_signals = signals.copy()
        balanced_condition_codes = condition_codes.copy()

        for condition_code, target_ratio in target_ratios.items():
            if condition_code not in condition_counts:
                logger.warning(f"Condition {condition_code} not found in dataset")
                continue

            current_count = condition_counts[condition_code]
            current_ratio = current_count / total_samples

            logger.info(
                f"Condition {condition_code}: {current_ratio:.3f} → {target_ratio:.3f}"
            )

            if current_ratio >= target_ratio:
                continue  # Already balanced

            target_count = int(len(balanced_signals) * target_ratio)
            needed_samples = target_count - current_count

            if needed_samples <= 0:
                continue

            logger.info(
                f"Generating {needed_samples} synthetic samples for {condition_code}"
            )

            original_indices = condition_indices[condition_code]
            original_condition_signals = [signals[i] for i in original_indices]

            generated_count = 0
            max_attempts = needed_samples * 3  # Prevent infinite loops
            attempts = 0

            while generated_count < needed_samples and attempts < max_attempts:
                attempts += 1

                source_signal = random.choice(original_condition_signals)

                try:
                    augmentation_methods = random.sample(
                        [
                            "amplitude_scaling",
                            "temporal_shift",
                            "gaussian_noise",
                            "baseline_wander",
                            "spectral_masking",
                            "phase_rotation",
                        ],
                        k=random.randint(2, 4),
                    )

                    augmented_signal = self.augment_signal(
                        source_signal, condition_code, fs, augmentation_methods
                    )

                    quality_score = self._assess_signal_quality(augmented_signal)
                    if quality_score >= self.config.min_quality_score:
                        balanced_signals.append(augmented_signal)
                        balanced_condition_codes.append(condition_code)
                        generated_count += 1
                    else:
                        logger.debug(
                            f"Rejected augmented sample with quality {quality_score}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to generate sample for {condition_code}: {e}"
                    )
                    continue

            logger.info(
                f"Generated {generated_count}/{needed_samples} samples for {condition_code}"
            )

        final_condition_counts = {}
        for condition_code in balanced_condition_codes:
            final_condition_counts[condition_code] = (
                final_condition_counts.get(condition_code, 0) + 1
            )

        logger.info("Final condition distribution:")
        for condition_code, count in final_condition_counts.items():
            ratio = count / len(balanced_condition_codes)
            logger.info(f"  {condition_code}: {count} samples ({ratio:.3f})")

        return balanced_signals, balanced_condition_codes


def create_augmentation_config(**kwargs) -> AugmentationConfig:
    """Factory function to create augmentation configuration"""
    return AugmentationConfig(**kwargs)


def augment_ecg_dataset(
    signals: list[np.ndarray],
    condition_codes: list[str],
    config: AugmentationConfig | None = None,
    balance_rare_conditions: bool = True,
    augmentation_factor: int = 2,
    fs: int = 500,
) -> tuple[list[np.ndarray], list[str]]:
    """
    High-level function to augment ECG dataset

    Args:
        signals: List of ECG signals
        condition_codes: List of condition codes
        config: Augmentation configuration
        balance_rare_conditions: Whether to balance rare conditions
        augmentation_factor: General augmentation factor
        fs: Sampling frequency

    Returns:
        Tuple of (augmented_signals, augmented_condition_codes)
    """
    if config is None:
        config = create_augmentation_config()

    augmenter = ECGDataAugmenter(config)

    augmented_signals, augmented_condition_codes = augmenter.augment_dataset(
        signals, condition_codes, augmentation_factor, fs
    )

    if balance_rare_conditions:
        augmented_signals, augmented_condition_codes = (
            augmenter.balance_rare_conditions(
                augmented_signals, augmented_condition_codes, fs=fs
            )
        )

    return augmented_signals, augmented_condition_codes


def demonstrate_stemi_balancing():
    """Demonstrate STEMI condition balancing as specified in requirements"""

    config = create_augmentation_config(
        target_balance_ratios={
            "STEMI": 0.05,  # 5% target
            "NSTEMI": 0.03,
            "AFIB": 0.08,
        },
        min_quality_score=0.7,
        preserve_clinical_features=True,
    )

    logger.info("STEMI Balancing Configuration:")
    logger.info(f"Target STEMI ratio: {config.target_balance_ratios['STEMI']:.1%}")
    logger.info(f"Quality threshold: {config.min_quality_score}")
    logger.info(f"Clinical feature preservation: {config.preserve_clinical_features}")

    return config


if __name__ == "__main__":
    config = create_augmentation_config(
        amplitude_scale_range=(0.8, 1.2),
        time_shift_range=50,
        gaussian_noise_snr_db=20.0,
        spectral_masking_ratio=0.1,
        heart_rate_perturbation_bpm=10.0,
        min_quality_score=0.7,
    )

    logger.info("ECG Data Augmentation System initialized")
    logger.info(f"Configuration: {config}")
    logger.info("Ready for dataset augmentation and rare condition balancing")

    stemi_config = demonstrate_stemi_balancing()
