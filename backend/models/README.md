# ML Models Directory

This directory contains the ONNX machine learning models used by CardioAI Pro for ECG analysis.

## Model Files (Mock/Placeholder)

For the standalone version, we include placeholder models that demonstrate the system architecture:

- `ecg_classifier.onnx` - Main ECG classification model
- `rhythm_detector.onnx` - Heart rhythm detection model  
- `quality_assessor.onnx` - Signal quality assessment model

## Production Deployment

In a production environment, these would be replaced with trained models specific to your clinical requirements and regulatory compliance needs.

## Model Information

- Input: 12-lead ECG data (5000 samples @ 500Hz)
- Output: Classification probabilities for various cardiac conditions
- Framework: ONNX Runtime for cross-platform inference
- Optimization: CPU-optimized for standalone deployment
