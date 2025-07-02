"""
ECG Document Scanner Service for digitizing ECG images.
"""

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ECGDocumentScanner:
    """Advanced ECG document scanner for digitizing ECG images."""

    def __init__(self) -> None:
        self.sampling_rate = 500  # Default sampling rate
        self.leads = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]

    async def digitize_ecg(
        self, filepath: str, sampling_rate: int = 500
    ) -> dict[str, Any]:
        """
        Digitize ECG from image file.

        Args:
            filepath: Path to ECG image file
            sampling_rate: Target sampling rate for digitized signal

        Returns:
            Dictionary containing digitized signal data and metadata
        """
        try:
            image = self._load_image(filepath)
            processed_image = self._preprocess_image(image)

            grid_info = self._detect_grid(processed_image)
            lead_regions = self._detect_lead_regions(processed_image, grid_info)

            signals = []
            confidence_scores = []

            for _, region in enumerate(lead_regions):
                signal, confidence = self._extract_signal_from_region(
                    region, sampling_rate
                )
                signals.append(signal)
                confidence_scores.append(confidence)

            while len(signals) < 12:
                signals.append(np.zeros(int(sampling_rate * 10)))  # 10 seconds of zeros
                confidence_scores.append(0.0)

            signal_matrix = np.column_stack(signals[:12])

            avg_confidence = np.mean(confidence_scores[:12])

            return {
                "signal": signal_matrix,
                "sampling_rate": sampling_rate,
                "labels": self.leads,
                "metadata": {
                    "source": "digitized_image",
                    "scanner_confidence": float(avg_confidence),
                    "document_detected": grid_info["grid_detected"],
                    "processing_method": "opencv_line_detection",
                    "grid_detected": grid_info["grid_detected"],
                    "leads_detected": len([c for c in confidence_scores if c > 0.3]),
                },
            }

        except Exception as e:
            logger.error(f"ECG image digitization failed for {filepath}: {str(e)}")
            mock_signal = np.random.randn(5000, 12) * 0.1
            return {
                "signal": mock_signal,
                "sampling_rate": sampling_rate,
                "labels": self.leads,
                "metadata": {
                    "source": "digitized_image_fallback",
                    "error": str(e),
                    "scanner_confidence": 0.0,
                    "processing_method": "fallback_random",
                },
            }

    def _load_image(self, filepath: str) -> np.ndarray[Any, Any]:
        """Load image using OpenCV."""
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Could not load image from {filepath}")
        return image

    def _preprocess_image(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess image for ECG detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        return thresh

    def _detect_grid(self, image: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Detect ECG grid pattern."""
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)

        grid_detected = (
            np.sum(horizontal_lines) > 1000 and np.sum(vertical_lines) > 1000
        )

        return {
            "grid_detected": grid_detected,
            "horizontal_lines": horizontal_lines,
            "vertical_lines": vertical_lines,
        }

    def _detect_lead_regions(
        self, image: np.ndarray[Any, Any], grid_info: dict[str, Any]
    ) -> list[np.ndarray[Any, Any]]:
        """Detect regions containing individual ECG leads."""
        height, width = image.shape

        regions = []
        rows, cols = 3, 4

        for row in range(rows):
            for col in range(cols):
                y1 = int(row * height / rows)
                y2 = int((row + 1) * height / rows)
                x1 = int(col * width / cols)
                x2 = int((col + 1) * width / cols)

                region = image[y1:y2, x1:x2]
                regions.append(region)

        return regions

    def _extract_signal_from_region(
        self, region: np.ndarray[Any, Any], sampling_rate: int
    ) -> tuple[np.ndarray[Any, Any], float]:
        """Extract ECG signal from a region using line detection."""
        try:
            contours, _ = cv2.findContours(
                region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return np.zeros(int(sampling_rate * 10)), 0.0

            longest_contour = max(contours, key=cv2.contourArea)

            height, width = region.shape
            signal_points = []

            for x in range(width):
                y_coords = []
                for point in longest_contour:
                    if point[0][0] == x:
                        y_coords.append(point[0][1])

                if y_coords:
                    avg_y = np.mean(y_coords)
                    signal_value = (height - avg_y) / height - 0.5
                    signal_points.append(signal_value)
                else:
                    if signal_points:
                        signal_points.append(signal_points[-1])
                    else:
                        signal_points.append(0.0)

            if signal_points:
                target_length = int(sampling_rate * 10)  # 10 seconds
                signal = np.interp(
                    np.linspace(0, len(signal_points) - 1, target_length),
                    np.arange(len(signal_points)),
                    signal_points,
                )

                confidence = min(
                    1.0, float(np.std(signal)) * 10
                )  # Heuristic confidence

                return signal, confidence
            else:
                return np.zeros(int(sampling_rate * 10)), 0.0

        except Exception as e:
            logger.warning(f"Signal extraction failed: {e}")
            return np.zeros(int(sampling_rate * 10)), 0.0
