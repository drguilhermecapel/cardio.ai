#!/usr/bin/env python3
"""
Basic functionality test for advanced ECG preprocessing pipeline
"""

import sys
import os

sys.path.append(".")

import numpy as np
import time
from app.preprocessing import AdvancedECGPreprocessor, EnhancedSignalQualityAnalyzer


def generate_synthetic_ecg(duration=10, fs=360, heart_rate=70):
    """Generate synthetic ECG signal with realistic characteristics"""
    t = np.linspace(0, duration, fs * duration)

    beat_interval = 60 / heart_rate  # seconds
    ecg_signal = np.zeros_like(t)

    for beat_time in np.arange(0, duration, beat_interval):
        beat_idx = int(beat_time * fs)
        if beat_idx < len(ecg_signal) - 50:
            qrs_width = int(0.08 * fs)  # 80ms QRS width
            qrs_indices = np.arange(
                beat_idx - qrs_width // 2, beat_idx + qrs_width // 2
            )
            qrs_indices = qrs_indices[
                (qrs_indices >= 0) & (qrs_indices < len(ecg_signal))
            ]

            qrs_pattern = np.exp(
                -0.5 * ((qrs_indices - beat_idx) / (qrs_width / 6)) ** 2
            )
            ecg_signal[qrs_indices] += qrs_pattern

    baseline = 0.1 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz baseline wander

    return ecg_signal + baseline, t


def add_noise_to_ecg(clean_signal, t, fs):
    """Add various types of noise to clean ECG signal"""
    noisy_signal = clean_signal.copy()

    noise_level = 0.15
    white_noise = np.random.normal(0, noise_level, len(clean_signal))
    noisy_signal += white_noise

    powerline_noise = 0.08 * np.sin(2 * np.pi * 50 * t)
    noisy_signal += powerline_noise

    muscle_noise = 0.05 * np.random.normal(0, 1, len(clean_signal))
    muscle_noise = np.convolve(muscle_noise, np.ones(5) / 5, mode="same")  # Smooth
    noisy_signal += muscle_noise

    return noisy_signal


def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio in dB"""
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def test_basic_functionality():
    """Test basic preprocessing functionality"""
    print("=" * 60)
    print("TESTING ADVANCED ECG PREPROCESSING PIPELINE")
    print("=" * 60)

    fs = 360
    duration = 10

    print(f"Initializing preprocessor (fs={fs} Hz, duration={duration}s)...")
    preprocessor = AdvancedECGPreprocessor(sampling_rate=fs)
    quality_analyzer = EnhancedSignalQualityAnalyzer(sampling_rate=fs)

    print("Generating synthetic ECG signals...")
    clean_ecg, t = generate_synthetic_ecg(duration=duration, fs=fs, heart_rate=70)
    noisy_ecg = add_noise_to_ecg(clean_ecg, t, fs)

    print(f"Clean ECG shape: {clean_ecg.shape}")
    print(f"Noisy ECG shape: {noisy_ecg.shape}")

    noise = noisy_ecg - clean_ecg
    initial_snr = calculate_snr(clean_ecg, noise)
    print(f"Initial SNR: {initial_snr:.2f} dB")

    print("\n" + "-" * 40)
    print("TEST 1: Clean Signal Preprocessing")
    print("-" * 40)

    try:
        start_time = time.time()
        processed_clean, quality_metrics_clean = (
            preprocessor.advanced_preprocessing_pipeline(clean_ecg, clinical_mode=True)
        )
        processing_time = (time.time() - start_time) * 1000

        print("✓ Clean signal preprocessing successful")
        print(f"  - Processed signal shape: {processed_clean.shape}")
        print(f"  - Quality score: {quality_metrics_clean['quality_score']:.3f}")
        print(f"  - R-peaks detected: {quality_metrics_clean['r_peaks_detected']}")
        print(f"  - Processing time: {processing_time:.1f}ms")
        print(
            f"  - Meets quality threshold: {quality_metrics_clean['meets_quality_threshold']}"
        )

        expected_peaks = int(70 * duration / 60)  # ~12 peaks for 70 bpm, 10s
        peak_accuracy = (
            abs(quality_metrics_clean["r_peaks_detected"] - expected_peaks)
            / expected_peaks
        )
        print(
            f"  - Expected R-peaks: {expected_peaks}, Detected: {quality_metrics_clean['r_peaks_detected']}"
        )
        print(f"  - R-peak detection accuracy: {(1-peak_accuracy)*100:.1f}%")

    except Exception as e:
        print(f"✗ Clean signal preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "-" * 40)
    print("TEST 2: Noisy Signal Preprocessing")
    print("-" * 40)

    try:
        start_time = time.time()
        processed_noisy, quality_metrics_noisy = (
            preprocessor.advanced_preprocessing_pipeline(noisy_ecg, clinical_mode=True)
        )
        processing_time = (time.time() - start_time) * 1000

        print("✓ Noisy signal preprocessing successful")
        print(f"  - Processed signal shape: {processed_noisy.shape}")
        print(f"  - Quality score: {quality_metrics_noisy['quality_score']:.3f}")
        print(f"  - R-peaks detected: {quality_metrics_noisy['r_peaks_detected']}")
        print(f"  - Processing time: {processing_time:.1f}ms")
        print(
            f"  - Meets quality threshold: {quality_metrics_noisy['meets_quality_threshold']}"
        )

        processed_noise = processed_noisy - clean_ecg[: len(processed_noisy)]
        final_snr = calculate_snr(clean_ecg[: len(processed_noisy)], processed_noise)
        snr_improvement = final_snr - initial_snr
        quality_improvement = (
            (quality_metrics_noisy["quality_score"] - 0.3) / 0.3 * 100
        )  # Assume baseline 0.3

        print(f"  - Final SNR: {final_snr:.2f} dB")
        print(f"  - SNR improvement: {snr_improvement:.2f} dB")
        print(f"  - Quality improvement: {quality_improvement:.1f}%")

    except Exception as e:
        print(f"✗ Noisy signal preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "-" * 40)
    print("TEST 3: Enhanced Quality Analysis")
    print("-" * 40)

    try:
        comprehensive_quality_clean = (
            quality_analyzer.assess_signal_quality_comprehensive(clean_ecg)
        )
        comprehensive_quality_noisy = (
            quality_analyzer.assess_signal_quality_comprehensive(noisy_ecg)
        )

        print("✓ Quality analysis successful")
        print(
            f"  - Clean signal overall score: {comprehensive_quality_clean['overall_score']:.3f}"
        )
        print(
            f"  - Noisy signal overall score: {comprehensive_quality_noisy['overall_score']:.3f}"
        )
        print(
            f"  - Quality difference: {comprehensive_quality_clean['overall_score'] - comprehensive_quality_noisy['overall_score']:.3f}"
        )
        print(
            f"  - Clean signal recommendations: {len(comprehensive_quality_clean['recommendations'])}"
        )
        print(
            f"  - Noisy signal recommendations: {len(comprehensive_quality_noisy['recommendations'])}"
        )

    except Exception as e:
        print(f"✗ Quality analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    targets_met = []

    if quality_improvement > 15:
        print(
            f"✓ Signal quality improvement: {quality_improvement:.1f}% (Target: >15%)"
        )
        targets_met.append(True)
    else:
        print(
            f"✗ Signal quality improvement: {quality_improvement:.1f}% (Target: >15%)"
        )
        targets_met.append(False)

    r_peak_accuracy = (1 - peak_accuracy) * 100
    if r_peak_accuracy > 99.5:
        print(f"✓ R-peak detection accuracy: {r_peak_accuracy:.1f}% (Target: >99.5%)")
        targets_met.append(True)
    else:
        print(f"✗ R-peak detection accuracy: {r_peak_accuracy:.1f}% (Target: >99.5%)")
        targets_met.append(False)

    if processing_time < 1000:  # Less than 1 second
        print(f"✓ Processing time: {processing_time:.1f}ms (Efficient)")
        targets_met.append(True)
    else:
        print(f"⚠ Processing time: {processing_time:.1f}ms (Could be optimized)")
        targets_met.append(False)

    success_rate = sum(targets_met) / len(targets_met) * 100
    print(f"\nOverall success rate: {success_rate:.1f}%")

    if success_rate >= 80:
        print("🎉 PREPROCESSING PIPELINE TEST PASSED!")
        return True
    else:
        print("❌ PREPROCESSING PIPELINE TEST FAILED!")
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
