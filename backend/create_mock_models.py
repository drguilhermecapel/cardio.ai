"""
Create mock ONNX models for CardioAI Pro standalone version.
These are placeholder models for demonstration and testing purposes.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import os
from pathlib import Path


def create_mock_ecg_classifier():
    """Create a mock ECG classifier ONNX model."""

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 12, 5000]
    )

    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 15]  # 15 conditions
    )

    weights = np.random.randn(12 * 5000, 15).astype(np.float32) * 0.01
    bias = np.random.randn(15).astype(np.float32) * 0.1

    weight_tensor = helper.make_tensor(
        "weights", TensorProto.FLOAT, [12 * 5000, 15], weights.flatten()
    )
    bias_tensor = helper.make_tensor("bias", TensorProto.FLOAT, [15], bias.flatten())

    flatten_node = helper.make_node("Flatten", ["input"], ["flattened"], axis=1)

    matmul_node = helper.make_node("MatMul", ["flattened", "weights"], ["matmul_out"])

    add_node = helper.make_node("Add", ["matmul_out", "bias"], ["add_out"])

    softmax_node = helper.make_node("Softmax", ["add_out"], ["output"], axis=1)

    graph = helper.make_graph(
        [flatten_node, matmul_node, add_node, softmax_node],
        "ecg_classifier",
        [input_tensor],
        [output_tensor],
        [weight_tensor, bias_tensor],
    )

    model = helper.make_model(graph, producer_name="CardioAI-Mock")

    return model


def create_mock_rhythm_detector():
    """Create a mock rhythm detector ONNX model."""

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 12, 5000]
    )

    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 7]  # 7 rhythm types
    )

    weights = np.random.randn(12 * 5000, 7).astype(np.float32) * 0.01
    bias = np.random.randn(7).astype(np.float32) * 0.1

    weight_tensor = helper.make_tensor(
        "weights", TensorProto.FLOAT, [12 * 5000, 7], weights.flatten()
    )
    bias_tensor = helper.make_tensor("bias", TensorProto.FLOAT, [7], bias.flatten())

    flatten_node = helper.make_node("Flatten", ["input"], ["flattened"], axis=1)

    matmul_node = helper.make_node("MatMul", ["flattened", "weights"], ["matmul_out"])

    add_node = helper.make_node("Add", ["matmul_out", "bias"], ["add_out"])

    softmax_node = helper.make_node("Softmax", ["add_out"], ["output"], axis=1)

    graph = helper.make_graph(
        [flatten_node, matmul_node, add_node, softmax_node],
        "rhythm_detector",
        [input_tensor],
        [output_tensor],
        [weight_tensor, bias_tensor],
    )

    model = helper.make_model(graph, producer_name="CardioAI-Mock")

    return model


def create_mock_quality_assessor():
    """Create a mock quality assessor ONNX model."""

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 12, 5000]
    )

    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1]  # Quality score 0-1
    )

    weights = np.random.randn(12 * 5000, 1).astype(np.float32) * 0.01
    bias = np.array([0.5], dtype=np.float32)  # Default to medium quality

    weight_tensor = helper.make_tensor(
        "weights", TensorProto.FLOAT, [12 * 5000, 1], weights.flatten()
    )
    bias_tensor = helper.make_tensor("bias", TensorProto.FLOAT, [1], bias.flatten())

    flatten_node = helper.make_node("Flatten", ["input"], ["flattened"], axis=1)

    matmul_node = helper.make_node("MatMul", ["flattened", "weights"], ["matmul_out"])

    add_node = helper.make_node("Add", ["matmul_out", "bias"], ["add_out"])

    sigmoid_node = helper.make_node("Sigmoid", ["add_out"], ["output"])

    graph = helper.make_graph(
        [flatten_node, matmul_node, add_node, sigmoid_node],
        "quality_assessor",
        [input_tensor],
        [output_tensor],
        [weight_tensor, bias_tensor],
    )

    model = helper.make_model(graph, producer_name="CardioAI-Mock")

    return model


def main():
    """Create all mock models."""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    print("Creating mock ONNX models...")

    ecg_model = create_mock_ecg_classifier()
    onnx.save(ecg_model, models_dir / "ecg_classifier.onnx")
    print("✓ Created ecg_classifier.onnx")

    rhythm_model = create_mock_rhythm_detector()
    onnx.save(rhythm_model, models_dir / "rhythm_detector.onnx")
    print("✓ Created rhythm_detector.onnx")

    quality_model = create_mock_quality_assessor()
    onnx.save(quality_model, models_dir / "quality_assessor.onnx")
    print("✓ Created quality_assessor.onnx")

    print(f"\nMock models created in: {models_dir}")
    print("These are placeholder models for demonstration purposes.")


if __name__ == "__main__":
    main()
