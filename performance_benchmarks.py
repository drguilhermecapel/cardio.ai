#!/usr/bin/env python3
"""
Performance Benchmarks for CardioAI Pro
Validates performance metrics for all model configurations
"""

import time
import torch
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def benchmark_model(self, model_path: str, test_data: torch.Tensor) -> Dict:
        """Benchmark a single model"""
        logger.info(f"Benchmarking model: {model_path}")
        
        # Load model
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_data[:1])
        
        # Benchmark inference time
        times = []
        with torch.no_grad():
            for i in range(100):
                start_time = time.time()
                _ = model(test_data[i:i+1])
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate metrics
        avg_latency = np.mean(times)
        p95_latency = np.percentile(times, 95)
        p99_latency = np.percentile(times, 99)
        
        # Model size
        model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
        
        return {
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'model_size_mb': model_size,
            'device': str(self.device)
        }
    
    def run_all_benchmarks(self) -> Dict:
        """Run benchmarks for all model configurations"""
        
        # Generate test data
        test_data = torch.randn(100, 12, 5000).to(self.device)
        
        models = {
            'baseline': 'models/baseline_model.pth',
            'advanced': 'models/advanced_model.pth',
            'ensemble': 'models/ensemble_model.pth',
            'mobile': 'models/mobile_model.pth'
        }
        
        results = {}
        
        for model_name, model_path in models.items():
            if Path(model_path).exists():
                try:
                    results[model_name] = self.benchmark_model(model_path, test_data)
                    logger.info(f"✓ {model_name} benchmarked successfully")
                except Exception as e:
                    logger.error(f"✗ Failed to benchmark {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
            else:
                logger.warning(f"Model not found: {model_path}")
                results[model_name] = {'error': 'Model file not found'}
        
        return results
    
    def validate_performance_targets(self, results: Dict) -> Dict:
        """Validate against performance targets"""
        
        targets = {
            'baseline': {'accuracy': 92.6, 'latency_ms': 150},
            'advanced': {'accuracy': 97.3, 'latency_ms': 100},
            'ensemble': {'accuracy': 98.1, 'latency_ms': 200},
            'mobile': {'accuracy': 95.2, 'latency_ms': 25}
        }
        
        validation_results = {}
        
        for model_name, target in targets.items():
            if model_name in results and 'error' not in results[model_name]:
                model_results = results[model_name]
                
                latency_ok = model_results['avg_latency_ms'] <= target['latency_ms']
                size_ok = True  # Will be validated separately
                
                if model_name == 'mobile':
                    size_ok = model_results['model_size_mb'] <= 5.0  # 5MB target
                
                validation_results[model_name] = {
                    'latency_target_met': latency_ok,
                    'size_target_met': size_ok,
                    'overall_pass': latency_ok and size_ok
                }
            else:
                validation_results[model_name] = {
                    'latency_target_met': False,
                    'size_target_met': False,
                    'overall_pass': False
                }
        
        return validation_results
    
    def generate_report(self, results: Dict, validation: Dict) -> str:
        """Generate comprehensive performance report"""
        
        report = []
        report.append("# CardioAI Pro Performance Benchmark Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Device: {self.device}")
        report.append("")
        
        report.append("## Performance Results")
        report.append("")
        
        for model_name, model_results in results.items():
            report.append(f"### {model_name.title()} Model")
            
            if 'error' in model_results:
                report.append(f"❌ **Error**: {model_results['error']}")
            else:
                report.append(f"- **Average Latency**: {model_results['avg_latency_ms']:.2f} ms")
                report.append(f"- **P95 Latency**: {model_results['p95_latency_ms']:.2f} ms")
                report.append(f"- **P99 Latency**: {model_results['p99_latency_ms']:.2f} ms")
                report.append(f"- **Model Size**: {model_results['model_size_mb']:.2f} MB")
                
                # Validation status
                if model_name in validation:
                    val = validation[model_name]
                    status = "✅ PASS" if val['overall_pass'] else "❌ FAIL"
                    report.append(f"- **Validation Status**: {status}")
            
            report.append("")
        
        report.append("## Target Validation")
        report.append("")
        
        for model_name, val in validation.items():
            status_icon = "✅" if val['overall_pass'] else "❌"
            report.append(f"- **{model_name.title()}**: {status_icon}")
        
        report.append("")
        report.append("## Performance Targets")
        report.append("")
        report.append("| Model | Accuracy Target | Latency Target | Size Target |")
        report.append("|-------|----------------|----------------|-------------|")
        report.append("| Baseline | 92.6% AUC | <150ms | N/A |")
        report.append("| Advanced | 97.3% AUC | <100ms | N/A |")
        report.append("| Ensemble | 98.1% AUC | <200ms | N/A |")
        report.append("| Mobile | 95.2% AUC | <25ms | <5MB |")
        
        return "\\n".join(report)

def main():
    """Main benchmark execution"""
    benchmark = PerformanceBenchmark()
    
    logger.info("Starting performance benchmarks...")
    
    # Run benchmarks
    results = benchmark.run_all_benchmarks()
    
    # Validate against targets
    validation = benchmark.validate_performance_targets(results)
    
    # Generate report
    report = benchmark.generate_report(results, validation)
    
    # Save results
    with open('performance_results.json', 'w') as f:
        json.dump({
            'results': results,
            'validation': validation,
            'timestamp': time.time()
        }, f, indent=2)
    
    # Save report
    with open('PERFORMANCE_REPORT.md', 'w') as f:
        f.write(report)
    
    logger.info("Benchmarks completed. Results saved to performance_results.json and PERFORMANCE_REPORT.md")
    
    # Print summary
    print("\\n" + "="*50)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*50)
    
    for model_name, val in validation.items():
        status = "PASS" if val['overall_pass'] else "FAIL"
        print(f"{model_name.upper()}: {status}")
    
    print("="*50)

if __name__ == "__main__":
    main()

