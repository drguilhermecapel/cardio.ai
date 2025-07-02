# ECG Dataset Integration Guide

## Overview

The CardioAI Pro system now includes comprehensive integration with major public ECG datasets, enabling training and validation of AI models using clinically validated data. This integration supports advanced preprocessing pipelines and provides seamless access to standardized ECG datasets for diagnostic pattern recognition.

## Supported Datasets

### 1. MIT-BIH Arrhythmia Database
- **Description**: Gold standard for arrhythmia detection research
- **Records**: 48 half-hour excerpts of two-channel ambulatory ECG recordings
- **Sampling Rate**: 360 Hz
- **Annotations**: Beat-by-beat arrhythmia annotations
- **Use Cases**: Arrhythmia detection, beat classification, algorithm validation

### 2. PTB-XL Database
- **Description**: Large publicly available electrocardiography dataset
- **Records**: 21,837 clinical 12-lead ECGs from 18,885 patients
- **Duration**: 10-second ECGs
- **Sampling Rate**: 500 Hz
- **Annotations**: Diagnostic statements, demographic data
- **Use Cases**: Multi-label classification, demographic analysis, clinical validation

### 3. CPSC-2018 Database
- **Description**: China Physiological Signal Challenge 2018 dataset
- **Records**: 6,877 single-lead ECG recordings
- **Duration**: 6-60 seconds
- **Sampling Rate**: 500 Hz
- **Annotations**: 9 types of cardiac abnormalities
- **Use Cases**: Abnormality detection, single-lead analysis

## Architecture Components

### ECGDatasetDownloader
Handles automatic downloading and caching of public ECG datasets.

```python
from app.datasets.ecg_public_datasets import ECGDatasetDownloader

downloader = ECGDatasetDownloader()

# Download MIT-BIH dataset (specific records)
mit_path = downloader.download_mit_bih(
    records_to_download=['100', '101', '102', '103', '104']
)

# Download PTB-XL dataset
ptb_path = downloader.download_ptb_xl()

# Download CPSC-2018 dataset
cpsc_path = downloader.download_cpsc_2018()
```

### ECGDatasetLoader
Provides unified interface for loading and preprocessing ECG data from different datasets.

```python
from app.datasets.ecg_public_datasets import ECGDatasetLoader

loader = ECGDatasetLoader()

# Load MIT-BIH records with preprocessing
records = loader.load_mit_bih(
    dataset_path='ecg_datasets/mit-bih',
    preprocess=True
)

# Load PTB-XL records
ptb_records = loader.load_ptb_xl(
    dataset_path='ecg_datasets/ptb-xl',
    preprocess=True
)
```

### ECGDatasetAnalyzer
Provides comprehensive analysis and visualization capabilities for loaded datasets.

```python
from app.datasets.ecg_public_datasets import ECGDatasetAnalyzer

analyzer = ECGDatasetAnalyzer()

# Analyze dataset characteristics
analyzer.analyze_dataset(records, "MIT-BIH Sample")

# Generate statistical reports
stats = analyzer.generate_statistics(records)
```

## Integration with Advanced Preprocessing Pipeline

The dataset integration seamlessly works with the advanced preprocessing pipeline implemented in `app.preprocessing.advanced_pipeline`:

```python
from app.preprocessing.advanced_pipeline import AdvancedECGPreprocessor
from app.datasets.ecg_public_datasets import ECGDatasetLoader

# Load dataset
loader = ECGDatasetLoader()
records = loader.load_mit_bih('ecg_datasets/mit-bih')

# Apply advanced preprocessing
preprocessor = AdvancedECGPreprocessor()

for record in records:
    processed_signal, quality_metrics = preprocessor.advanced_preprocessing_pipeline(
        record.signal, 
        clinical_mode=True
    )
    
    print(f"Quality Score: {quality_metrics['quality_score']:.3f}")
    print(f"R-peaks detected: {quality_metrics['r_peaks_detected']}")
```

## HybridECGAnalysisService Integration

The dataset service is automatically integrated into the main analysis service:

```python
from app.services.hybrid_ecg_service import HybridECGAnalysisService

# Service automatically initializes with dataset integration
service = HybridECGAnalysisService(db=db, validation_service=validation_service)

# Dataset service is available as an attribute
if service.dataset_service_available:
    print("✓ Public ECG datasets are available for training and validation")
```

## Quick Start Examples

### Scenario 1: Download and Explore MIT-BIH
```python
from app.datasets.ecg_datasets_quickguide import scenario_1_download_and_explore

# Downloads MIT-BIH sample and generates visualizations
records = scenario_1_download_and_explore()
```

### Scenario 2: Prepare ML Dataset
```python
from app.datasets.ecg_datasets_quickguide import scenario_2_prepare_ml_dataset

# Prepares dataset for machine learning with train/test split
X_train, X_test, y_train, y_test, labels = scenario_2_prepare_ml_dataset(records)
```

### Scenario 3: Quick Start
```python
from app.datasets.ecg_datasets_quickguide import quick_start_mit_bih

# One-line setup for MIT-BIH dataset
X, y = quick_start_mit_bih()
```

## Data Structures

### ECGRecord
Standardized data structure for ECG records across all datasets:

```python
@dataclass
class ECGRecord:
    patient_id: str
    signal: np.ndarray
    sampling_rate: int
    duration: float
    leads: List[str]
    labels: List[str]
    metadata: Dict[str, Any]
    annotations: Optional[Dict[str, Any]] = None
```

## Machine Learning Integration

### Dataset Preparation
```python
from app.datasets.ecg_public_datasets import prepare_ml_dataset

# Prepare windowed dataset for ML training
X, y = prepare_ml_dataset(
    records,
    window_size=3600,  # 10 seconds @ 360Hz
    target_labels=['normal', 'pvc', 'pac', 'arrhythmia']
)
```

### Batch Processing
```python
from app.datasets.ecg_public_datasets import load_and_preprocess_all

# Load and preprocess multiple datasets
paths = {
    'mit-bih': 'ecg_datasets/mit-bih',
    'ptb-xl': 'ecg_datasets/ptb-xl'
}

datasets = load_and_preprocess_all(paths)
```

## Quality Assessment Integration

The dataset integration includes comprehensive quality assessment using the enhanced signal quality analyzer:

```python
from app.preprocessing.enhanced_quality_analyzer import EnhancedSignalQualityAnalyzer

analyzer = EnhancedSignalQualityAnalyzer()

for record in records:
    quality_metrics = analyzer.assess_signal_quality_comprehensive(record.signal)
    
    if quality_metrics['overall_score'] > 0.8:
        # Use high-quality signal for training
        process_for_training(record)
```

## Performance Considerations

### Memory Management
- Datasets are loaded incrementally to manage memory usage
- Batch processing capabilities for large datasets
- Automatic cleanup of temporary files

### Processing Speed
- Parallel processing support for multiple records
- Optimized preprocessing pipelines
- Caching mechanisms for repeated operations

### Storage Requirements
- MIT-BIH: ~23 MB
- PTB-XL: ~828 MB
- CPSC-2018: ~434 MB

## Error Handling and Logging

The dataset integration includes comprehensive error handling and logging:

```python
import logging

# Configure logging for dataset operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ecg_datasets')

# All dataset operations are logged with detailed information
# Example log output:
# INFO:ecg_datasets:✓ MIT-BIH dataset downloaded successfully
# INFO:ecg_datasets:✓ Loaded 5 records from MIT-BIH database
# INFO:ecg_datasets:✓ Advanced preprocessing completed: quality=0.553, r_peaks=6
```

## Dependencies

The dataset integration requires the following additional dependencies:

```toml
# Core dependencies
h5py = "^3.14.0"          # HDF5 file format support
wfdb = "^4.3.0"           # WFDB format support
pyedflib = "^0.1.40"      # EDF format support
pandas = "^2.0.0"         # Data manipulation
matplotlib = "^3.10.3"    # Visualization
tqdm = "^4.65.0"          # Progress bars
```

## Testing and Validation

The dataset integration includes comprehensive testing:

```bash
# Run dataset integration tests
poetry run python -m pytest tests/test_dataset_integration.py -v

# Test specific dataset loading
poetry run python -c "
from app.datasets.ecg_public_datasets import ECGDatasetLoader
loader = ECGDatasetLoader()
records = loader.load_mit_bih('ecg_datasets/mit-bih', preprocess=True)
print(f'✓ Loaded {len(records)} records successfully')
"
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   poetry add h5py wfdb pyedflib
   ```

2. **Dataset Download Failures**
   - Check internet connectivity
   - Verify dataset URLs are accessible
   - Check available disk space

3. **Memory Issues with Large Datasets**
   - Use batch processing
   - Reduce window sizes
   - Process datasets incrementally

4. **Quality Score Too Low**
   - Adjust quality thresholds in ECGConfig
   - Review signal preprocessing parameters
   - Check for dataset-specific artifacts

### Debug Mode
```python
from app.datasets.ecg_public_datasets import ECGDatasetLoader

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

loader = ECGDatasetLoader(debug=True)
```

## Future Enhancements

- Support for additional public ECG datasets
- Real-time dataset streaming capabilities
- Advanced data augmentation techniques
- Integration with cloud storage providers
- Automated dataset quality assessment
- Cross-dataset validation frameworks

## Contributing

When adding support for new datasets:

1. Extend `ECGDatasetDownloader` with new download methods
2. Add loading logic to `ECGDatasetLoader`
3. Update `ECGRecord` structure if needed
4. Add comprehensive tests
5. Update this documentation

## References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/
- PTB-XL Database: https://physionet.org/content/ptb-xl/
- CPSC-2018 Challenge: http://2018.icbeb.org/Challenge.html
- WFDB Python Package: https://github.com/MIT-LCP/wfdb-python
