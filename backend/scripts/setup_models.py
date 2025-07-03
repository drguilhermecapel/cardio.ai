"""
Script to download and setup ML models.
"""

import asyncio
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


async def download_models():
    """Download ML models for ECG analysis."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_files = [
        "ecg_classification_model.onnx",
        "rhythm_detection_model.onnx",
        "quality_assessment_model.onnx",
    ]

    for model_file in model_files:
        model_path = models_dir / model_file
        if not model_path.exists():
            with open(model_path, "w") as f:
                f.write("# Placeholder ONNX model file\n")
            logger.info(f"Created placeholder model: {model_path}")
        else:
            logger.info(f"Model already exists: {model_path}")


if __name__ == "__main__":
    asyncio.run(download_models())
