#!/usr/bin/env python3
"""
Test script for ECG image upload functionality.
"""

import asyncio
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os

from app.services.ecg_document_scanner import ECGDocumentScanner
from app.services.hybrid_ecg_service import UniversalECGReader


async def create_test_ecg_image():
    """Create a simple test ECG image."""
    width, height = 800, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for x in range(0, width, 20):
        draw.line([(x, 0), (x, height)], fill="lightgray", width=1)
    for y in range(0, height, 20):
        draw.line([(0, y), (width, y)], fill="lightgray", width=1)

    for lead in range(3):
        y_offset = height // 4 + lead * (height // 4)
        points = []
        for x in range(0, width, 2):
            y = y_offset + 20 * np.sin(x * 0.02) + 5 * np.random.randn()
            points.append((x, int(y)))

        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill="black", width=2)

    return image


async def test_image_processing():
    """Test ECG image processing."""
    print("Creating test ECG image...")
    test_image = await create_test_ecg_image()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        test_image.save(tmp.name)

        print(f"Testing image processing on: {tmp.name}")

        scanner = ECGDocumentScanner()
        result = await scanner.digitize_ecg(tmp.name)

        print(f"Scanner confidence: {result['metadata']['scanner_confidence']}")
        print(f"Grid detected: {result['metadata']['grid_detected']}")
        print(f"Leads detected: {result['metadata']['leads_detected']}")
        print(f"Signal shape: {result['signal'].shape}")

        reader = UniversalECGReader()
        reader_result = reader.read_ecg(tmp.name)

        print(f"Reader result keys: {list(reader_result.keys())}")
        if "signal" in reader_result:
            print(f"Reader result shape: {reader_result['signal'].shape}")
        if "metadata" in reader_result:
            print(f"Reader metadata: {reader_result['metadata']}")
        else:
            print(f"Reader result: {reader_result}")

        os.unlink(tmp.name)


if __name__ == "__main__":
    asyncio.run(test_image_processing())
