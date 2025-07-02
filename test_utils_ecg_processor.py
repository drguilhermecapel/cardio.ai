import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from app.utils.ecg_processor import ECGProcessor

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"


def test_ecg_processor_initialization():
    """Test ECG processor initialization."""
    processor = ECGProcessor()
    assert processor is not None


@pytest.mark.asyncio
async def test_preprocess_signal():
    """Test ECG signal preprocessing."""
    processor = ECGProcessor()
    signal = np.random.randn(1000, 1)

    processed = await processor.preprocess_signal(signal)

    assert isinstance(processed, np.ndarray)
    assert processed.shape == signal.shape


@pytest.mark.asyncio
async def test_extract_metadata():
    """Test ECG metadata extraction."""
    processor = ECGProcessor()

    with patch("pathlib.Path.exists", return_value=True):
        with patch.object(processor, "load_ecg_file") as mock_load:
            mock_load.return_value = np.random.randn(1000, 12)

            metadata = await processor.extract_metadata("test.csv")

            assert isinstance(metadata, dict)
            assert "leads_count" in metadata
            assert "duration_seconds" in metadata


@pytest.mark.asyncio
async def test_load_csv_file():
    """Test CSV file loading."""
    processor = ECGProcessor()

    with patch("numpy.loadtxt") as mock_loadtxt:
        mock_loadtxt.return_value = np.random.randn(1000)

        result = await processor._load_csv("test.csv")

        assert isinstance(result, np.ndarray)
        mock_loadtxt.assert_called_once()


@pytest.mark.asyncio
async def test_load_text_file():
    """Test text file loading."""
    processor = ECGProcessor()

    with patch("numpy.loadtxt") as mock_loadtxt:
        mock_loadtxt.return_value = np.random.randn(1000)

        result = await processor._load_text("test.txt")

        assert isinstance(result, np.ndarray)
        mock_loadtxt.assert_called_once()


@pytest.mark.asyncio
async def test_load_xml_file():
    """Test XML file loading."""
    processor = ECGProcessor()

    xml_content = """<?xml version="1.0"?>
    <ecg>
        <waveform>
            <data>1.0 2.0 3.0 4.0 5.0</data>
        </waveform>
    </ecg>"""

    with patch("xml.etree.ElementTree.parse") as mock_parse:
        mock_tree = MagicMock()
        mock_root = MagicMock()
        mock_data_elem = MagicMock()
        mock_data_elem.text = "1.0 2.0 3.0 4.0 5.0"
        mock_root.findall.return_value = [mock_data_elem]
        mock_tree.getroot.return_value = mock_root
        mock_parse.return_value = mock_tree

        result = await processor._load_xml("test.xml")

        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 1


@pytest.mark.asyncio
async def test_load_ecg_file_error():
    """Test ECG file loading with unsupported format."""
    processor = ECGProcessor()

    with patch("pathlib.Path.exists", return_value=True):
        with pytest.raises(Exception):
            await processor.load_ecg_file("test.unsupported")


@pytest.mark.asyncio
async def test_preprocess_signal_with_scipy():
    """Test ECG signal preprocessing with scipy filtering."""
    processor = ECGProcessor()
    signal = np.random.randn(1000, 2)

    with patch("scipy.signal.butter") as mock_butter:
        with patch("scipy.signal.filtfilt") as mock_filtfilt:
            mock_butter.return_value = ([1, 2], [3, 4])
            mock_filtfilt.return_value = signal[:, 0]

            result = await processor.preprocess_signal(signal)

            assert isinstance(result, np.ndarray)
            assert result.shape == signal.shape


@pytest.mark.asyncio
async def test_extract_metadata_with_xml():
    """Test metadata extraction from XML file."""
    processor = ECGProcessor()

    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.suffix", new_callable=lambda: ".xml"):
            with patch.object(processor, "_extract_xml_metadata") as mock_xml_meta:
                with patch.object(processor, "load_ecg_file") as mock_load:
                    mock_xml_meta.return_value = {"sample_rate": 250}
                    mock_load.return_value = np.random.randn(500, 6)

                    metadata = await processor.extract_metadata("test.xml")

                    assert isinstance(metadata, dict)
                    assert "leads_count" in metadata


@pytest.mark.asyncio
async def test_load_text_file_error():
    """Test text file loading with error."""
    processor = ECGProcessor()

    with patch("numpy.loadtxt") as mock_loadtxt:
        mock_loadtxt.side_effect = Exception("File read error")

        with pytest.raises(Exception):
            await processor._load_text("test.txt")


@pytest.mark.asyncio
async def test_load_csv_file_error():
    """Test CSV file loading with error."""
    processor = ECGProcessor()

    with patch("numpy.loadtxt") as mock_loadtxt:
        mock_loadtxt.side_effect = Exception("CSV read error")

        with pytest.raises(Exception):
            await processor._load_csv("test.csv")


@pytest.mark.asyncio
async def test_ecg_processor_error_handling():
    """Test ECG processor error handling."""
    processor = ECGProcessor()

    with pytest.raises(Exception):
        await processor.load_ecg_file("non_existent_file.csv")


@pytest.mark.asyncio
async def test_xml_metadata_extraction():
    """Test XML metadata extraction."""
    processor = ECGProcessor()

    with patch("xml.etree.ElementTree.parse") as mock_parse:
        mock_tree = MagicMock()
        mock_root = MagicMock()
        mock_sample_rate = MagicMock()
        mock_sample_rate.text = "500"
        mock_root.find.return_value = mock_sample_rate
        mock_tree.getroot.return_value = mock_root
        mock_parse.return_value = mock_tree

        metadata = await processor._extract_xml_metadata("test.xml")

        assert isinstance(metadata, dict)


@pytest.mark.asyncio
async def test_load_xml_file_error():
    """Test XML file loading with error."""
    processor = ECGProcessor()

    with patch("xml.etree.ElementTree.parse") as mock_parse:
        mock_parse.side_effect = Exception("XML parse error")

        with pytest.raises(Exception):
            await processor._load_xml("test.xml")
