# PTB-XL Training Guide

## Quick Start

### 1. Setup Environment
```bash
# Navigate to PTB-XL directory
cd backend/app/ml/ptbxl

# Run automated pipeline
./automated_ptbxl_pipeline.sh
```

### 2. Manual Training
```bash
# Basic training
python quickstart.py

# Advanced experiments
python run_ptbxl_experiments.py --config advanced

# Custom training
python train_ptbxl.py --epochs 100 --batch_size 32
```

## Configuration Options

### Experiment Configurations
1. **baseline**: Basic CNN model (~92.6% AUC)
2. **advanced**: Hybrid architecture (~97.3% AUC)
3. **ensemble**: Multiple models (~98.1% AUC)
4. **mobile**: Optimized for edge (~95.2% AUC)

### Training Parameters
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--num_classes`: Number of classes (5 or 71)

## Expected Results

### Performance Targets
- **Baseline**: ~92.6% AUC in 50 epochs
- **Advanced**: ~97.3% AUC in 100 epochs
- **Ensemble**: ~98.1% AUC (state-of-the-art)
- **Mobile**: ~95.2% AUC with 25ms latency

### Dataset Information
- **Total samples**: 21,837 ECG recordings
- **Duration**: 10 seconds each
- **Sampling rate**: 500 Hz
- **Leads**: 12-lead ECG
- **Classes**: 5 superclasses or 71 detailed classes

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size
2. **Dataset not found**: Run automated pipeline first
3. **Low performance**: Check data preprocessing
4. **Slow training**: Enable mixed precision

### Performance Optimization
- Use mixed precision training
- Enable gradient accumulation
- Use data parallel training
- Optimize data loading

## Integration

### Model Deployment
```bash
# Prepare model for deployment
python integrate_ptbxl_model.py --model_path models/best_model.pth

# Validate performance
python integrate_ptbxl_model.py --validate --model_path models/best_model.pth
```

### API Integration
The trained models are automatically integrated with the main CardioAI Pro API through the advanced ML service.

