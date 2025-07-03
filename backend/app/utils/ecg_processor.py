"""
ECG signal processing utilities.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

# import neurokit2 as nk  # Removed for standalone version
import numpy as np
from numpy.typing import NDArray

from app.core.exceptions import ECGProcessingException

logger = logging.getLogger(__name__)


class ECGProcessor:
    """ECG signal processing and file handling."""

    async def load_ecg_file(self, file_path: str) -> NDArray[np.float64]:
        """Load ECG data from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise ECGProcessingException(f"File not found: {file_path}")

            if path.suffix.lower() == ".csv":
                return await self._load_csv(file_path)
            elif path.suffix.lower() in [".txt", ".dat"]:
                return await self._load_text(file_path)
            elif path.suffix.lower() == ".xml":
                return await self._load_xml(file_path)
            else:
                raise ECGProcessingException(f"Unsupported file format: {path.suffix}")

        except Exception as e:
            logger.error("Failed to load ECG file %s: %s", file_path, str(e))
            raise ECGProcessingException(f"Failed to load ECG file: {str(e)}") from e

    async def _load_csv(self, file_path: str) -> NDArray[np.float64]:
        """Load ECG data from CSV file."""
        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return data
        except Exception as e:
            raise ECGProcessingException(f"Failed to load CSV file: {str(e)}") from e

    async def _load_text(self, file_path: str) -> NDArray[np.float64]:
        """Load ECG data from text file."""
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return data
        except Exception as e:
            raise ECGProcessingException(f"Failed to load text file: {str(e)}") from e

    async def _load_xml(self, file_path: str) -> NDArray[np.float64]:
        """Load ECG data from XML file (simplified implementation)."""
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            data_elements = root.findall(".//waveform/data")
            if not data_elements:
                raise ECGProcessingException("No waveform data found in XML")

            data_text = data_elements[0].text
            if data_text:
                values = [float(x) for x in data_text.split()]
                data = np.array(values).reshape(-1, 1)
                return data
            else:
                raise ECGProcessingException("Empty waveform data in XML")

        except Exception as e:
            raise ECGProcessingException(f"Failed to load XML file: {str(e)}") from e

    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from ECG file."""
        try:
            path = Path(file_path)
            metadata = {
                "acquisition_date": datetime.utcnow(),
                "sample_rate": 500,  # Default
                "duration_seconds": 10.0,  # Default
                "leads_count": 12,  # Default
                "leads_names": [
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
                ],
            }

            if path.suffix.lower() == ".xml":
                xml_metadata = await self._extract_xml_metadata(file_path)
                metadata.update(xml_metadata)

            data = await self.load_ecg_file(file_path)
            metadata["leads_count"] = data.shape[1]
            sample_rate = metadata.get("sample_rate", 500)
            if isinstance(sample_rate, int | float):
                calculated_duration = float(data.shape[0]) / float(sample_rate)
                metadata["duration_seconds"] = max(calculated_duration, 0.1)
            else:
                metadata["duration_seconds"] = 10.0

            return metadata

        except Exception as e:
            logger.error("Failed to extract metadata from %s: %s", file_path, str(e))
            return {
                "acquisition_date": datetime.utcnow(),
                "sample_rate": 500,
                "duration_seconds": 10.0,
                "leads_count": 12,
                "leads_names": [
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
                ],
            }

    async def _extract_xml_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from XML file."""
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            metadata: dict[str, Any] = {}

            sample_rate_elem = root.find(".//sampleRate")
            if sample_rate_elem is not None and sample_rate_elem.text is not None:
                metadata["sample_rate"] = int(sample_rate_elem.text)

            date_elem = root.find(".//acquisitionDate")
            if date_elem is not None:
                if date_elem.text is not None:
                    try:
                        metadata["acquisition_date"] = datetime.fromisoformat(
                            date_elem.text
                        )
                    except Exception:
                        pass

            device_elem = root.find(".//device")
            if device_elem is not None:
                manufacturer = device_elem.find("manufacturer")
                if manufacturer is not None and manufacturer.text is not None:
                    metadata["device_manufacturer"] = manufacturer.text

                model = device_elem.find("model")
                if model is not None and model.text is not None:
                    metadata["device_model"] = model.text

                serial = device_elem.find("serialNumber")
                if serial is not None and serial.text is not None:
                    metadata["device_serial"] = serial.text

            return metadata

        except Exception as e:
            logger.error("Failed to extract XML metadata: %s", str(e))
            return {}

    async def preprocess_signal(
        self, ecg_data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Preprocess ECG signal for analysis."""
        try:
            processed_data = ecg_data.copy()

            for i in range(processed_data.shape[1]):
                lead_data = processed_data[:, i]

                from scipy.signal import butter, filtfilt

                cleaned_signal = lead_data - np.mean(lead_data)

                nyquist = 500 / 2
                low = 0.5 / nyquist
                high = 40.0 / nyquist

                try:
                    b, a = butter(4, [low, high], btype="band")
                    cleaned_signal = filtfilt(b, a, cleaned_signal)
                except Exception:
                    pass

                processed_data[:, i] = cleaned_signal

            return processed_data

        except Exception as e:
            logger.error("Signal preprocessing failed: %s", str(e))
            return ecg_data

    async def process_file(self, file_path: str) -> dict[str, Any]:
        """Process ECG file and return processed data with metadata."""
        try:
            ecg_data = await self.load_ecg_file(file_path)
            metadata = await self.extract_metadata(file_path)
            preprocessed_data = await self.preprocess_signal(ecg_data)

            return {
                "raw_data": ecg_data,
                "preprocessed_data": preprocessed_data,
                "metadata": metadata,
                "file_path": file_path,
                "processing_success": True,
            }

        except Exception as e:
            logger.error("Failed to process ECG file %s: %s", file_path, str(e))
            raise ECGProcessingException(f"Failed to process file: {str(e)}") from e
