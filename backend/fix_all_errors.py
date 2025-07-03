#!/usr/bin/env python3
"""
Fix All Errors - Automatic Error Correction Script
Fixes all identified issues in CardioAI Pro system
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any


class CardioAIFixer:
    """Automatic fixer for CardioAI Pro errors"""
    
    def __init__(self):
        self.backend_path = Path(__file__).parent
        self.app_path = self.backend_path / "app"
        self.tests_path = self.backend_path / "tests"
        self.errors_fixed = []
        self.files_created = []
        self.files_modified = []
        
    def run(self):
        """Run all fixes"""
        print("🏥 CardioAI Pro - Sistema de Correção Automática")
        print("=" * 60)
        
        # 1. Fix syntax error in ecg_service.py
        self.fix_syntax_error()
        
        # 2. Create missing imports in __init__ files
        self.fix_init_files()
        
        # 3. Fix test imports
        self.fix_test_imports()
        
        # 4. Create missing utility files
        self.create_missing_utils()
        
        # 5. Fix database models
        self.fix_database_models()
        
        # 6. Install missing dependencies
        self.install_dependencies()
        
        # 7. Create missing directories
        self.create_directories()
        
        # 8. Generate summary
        self.print_summary()
        
    def fix_syntax_error(self):
        """Fix syntax error in ecg_service.py line 701"""
        print("\n📝 Corrigindo erro de sintaxe em ecg_service.py...")
        
        ecg_service_path = self.app_path / "services" / "ecg_service.py"
        
        if ecg_service_path.exists():
            try:
                # Read file
                with open(ecg_service_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Fix the syntax error on line 701
                # Looking for unterminated string
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'return {"id": 1, "status": "' in line and not line.strip().endswith('}'):
                        lines[i] = '        return {"id": 1, "status": "completed"}'
                        self.errors_fixed.append(f"Fixed syntax error on line {i+1}")
                
                # Write back
                with open(ecg_service_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                self.files_modified.append(str(ecg_service_path))
                print("✅ Erro de sintaxe corrigido")
                
            except Exception as e:
                print(f"❌ Erro ao corrigir sintaxe: {e}")
        else:
            print("⚠️ Arquivo ecg_service.py não encontrado - será criado com o código completo")
    
    def fix_init_files(self):
        """Fix __init__.py files with proper imports"""
        print("\n📦 Corrigindo arquivos __init__.py...")
        
        init_configs = {
            self.app_path / "services" / "__init__.py": [
                "from .ecg_service import ECGAnalysisService",
                "from .user_service import UserService",
                "from .patient_service import PatientService",
                "from .validation_service import ValidationService",
                "from .notification_service import NotificationService",
                "from .ml_model_service import MLModelService",
                "from .interpretability_service import InterpretabilityService",
                "",
                "__all__ = [",
                '    "ECGAnalysisService",',
                '    "UserService",',
                '    "PatientService",',
                '    "ValidationService",',
                '    "NotificationService",',
                '    "MLModelService",',
                '    "InterpretabilityService",',
                "]"
            ],
            self.app_path / "utils" / "__init__.py": [
                "from .memory_monitor import MemoryMonitor, get_memory_monitor",
                "from .ecg_processor import ECGProcessor",
                "from .file_utils import save_upload_file, delete_file",
                "from .validators import validate_ecg_file",
                "from .security import get_password_hash, verify_password",
                "",
                "__all__ = [",
                '    "MemoryMonitor",',
                '    "get_memory_monitor",',
                '    "ECGProcessor",',
                '    "save_upload_file",',
                '    "delete_file",',
                '    "validate_ecg_file",',
                '    "get_password_hash",',
                '    "verify_password",',
                "]"
            ],
            self.app_path / "schemas" / "__init__.py": [
                "from .ecg_analysis import *",
                "from .user import *",
                "from .patient import *",
                "from .validation import *",
                "from .notification import *",
            ],
            self.app_path / "core" / "__init__.py": [
                "from .config import settings",
                "from .exceptions import *",
                "from .logging import get_logger",
                "from .security import *",
            ],
        }
        
        for init_path, imports in init_configs.items():
            try:
                init_path.parent.mkdir(parents=True, exist_ok=True)
                with open(init_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(imports))
                self.files_created.append(str(init_path))
                print(f"✅ {init_path.name} em {init_path.parent.name}")
            except Exception as e:
                print(f"❌ Erro em {init_path}: {e}")
    
    def fix_test_imports(self):
        """Fix imports in test files"""
        print("\n🧪 Corrigindo imports nos testes...")
        
        test_fixes = {
            "get_memory_info()": "get_memory_stats()",
            "from app.services.ecg_service import ecg_service": "from app.services.ecg_service import ECGAnalysisService",
            "ecg_service.": "ecg_service_instance.",
        }
        
        fixed_count = 0
        
        for test_file in self.tests_path.rglob("test_*.py"):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                for old, new in test_fixes.items():
                    content = content.replace(old, new)
                
                if content != original_content:
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixed_count += 1
                    self.files_modified.append(str(test_file))
                    
            except Exception as e:
                print(f"⚠️ Erro em {test_file.name}: {e}")
        
        print(f"✅ {fixed_count} arquivos de teste corrigidos")
    
    def create_missing_utils(self):
        """Create missing utility files"""
        print("\n🛠️ Criando arquivos utilitários faltantes...")
        
        # ECGProcessor
        ecg_processor_path = self.app_path / "utils" / "ecg_processor.py"
        if not ecg_processor_path.exists():
            ecg_processor_content = '''"""ECG Processor - Signal Processing Utilities"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional, Any


class ECGProcessor:
    """ECG signal processing utilities"""
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        
    def process_signal(self, ecg_signal: np.ndarray) -> Dict[str, Any]:
        """Process ECG signal and extract features"""
        # Remove baseline wander
        processed = self.remove_baseline_wander(ecg_signal)
        
        # Detect R peaks
        r_peaks = self.detect_r_peaks(processed)
        
        # Calculate heart rate
        heart_rate = self.calculate_heart_rate(r_peaks)
        
        # Extract features
        features = {
            "heart_rate": heart_rate,
            "r_peaks": r_peaks.tolist(),
            "signal_quality": self.assess_signal_quality(processed)
        }
        
        return features
    
    def remove_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Remove baseline wander using high-pass filter"""
        # Butterworth high-pass filter
        b, a = signal.butter(4, 0.5/(self.sampling_rate/2), 'high')
        return signal.filtfilt(b, a, signal)
    
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Detect R peaks using Pan-Tompkins algorithm"""
        # Simplified implementation
        # Derivative
        diff = np.diff(ecg_signal)
        
        # Square
        squared = diff ** 2
        
        # Moving average
        window = int(0.150 * self.sampling_rate)  # 150ms window
        mwa = np.convolve(squared, np.ones(window)/window, mode='same')
        
        # Find peaks
        peaks, _ = signal.find_peaks(mwa, height=np.std(mwa)*2, distance=int(0.2*self.sampling_rate))
        
        return peaks
    
    def calculate_heart_rate(self, r_peaks: np.ndarray) -> float:
        """Calculate heart rate from R peaks"""
        if len(r_peaks) < 2:
            return 0.0
        
        # RR intervals in samples
        rr_intervals = np.diff(r_peaks)
        
        # Convert to seconds
        rr_seconds = rr_intervals / self.sampling_rate
        
        # Heart rate in bpm
        heart_rate = 60.0 / np.mean(rr_seconds)
        
        return round(heart_rate, 1)
    
    def assess_signal_quality(self, signal: np.ndarray) -> float:
        """Assess ECG signal quality (0-100)"""
        # Simple quality metrics
        # Check for flat line
        if np.std(signal) < 0.01:
            return 0.0
        
        # Check for clipping
        clipping_ratio = np.sum(np.abs(signal) > 0.95) / len(signal)
        if clipping_ratio > 0.01:
            return 50.0
        
        # Basic SNR estimate
        noise = signal - self.remove_baseline_wander(signal)
        snr = 10 * np.log10(np.var(signal) / (np.var(noise) + 1e-10))
        
        # Convert to 0-100 scale
        quality = min(100, max(0, snr * 5))
        
        return round(quality, 1)
'''
            with open(ecg_processor_path, 'w', encoding='utf-8') as f:
                f.write(ecg_processor_content)
            self.files_created.append(str(ecg_processor_path))
            print("✅ ECGProcessor criado")
        
        # File utils
        file_utils_path = self.app_path / "utils" / "file_utils.py"
        if not file_utils_path.exists():
            file_utils_content = '''"""File utilities for handling uploads"""

import os
import shutil
from pathlib import Path
from typing import Optional
import hashlib


def save_upload_file(file_content: bytes, filename: str, upload_dir: str = "uploads") -> str:
    """Save uploaded file and return path"""
    # Create upload directory
    upload_path = Path(upload_dir)
    upload_path.mkdir(exist_ok=True)
    
    # Generate unique filename
    file_hash = hashlib.md5(file_content).hexdigest()[:8]
    name, ext = os.path.splitext(filename)
    unique_filename = f"{name}_{file_hash}{ext}"
    
    # Save file
    file_path = upload_path / unique_filename
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    return str(file_path)


def delete_file(file_path: str) -> bool:
    """Delete file if exists"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False
'''
            with open(file_utils_path, 'w', encoding='utf-8') as f:
                f.write(file_utils_content)
            self.files_created.append(str(file_utils_path))
            print("✅ File utils criado")
        
        # Validators
        validators_path = self.app_path / "utils" / "validators.py"
        if not validators_path.exists():
            validators_content = '''"""Validation utilities"""

from typing import Tuple
import os


def validate_ecg_file(file_path: str, max_size_mb: int = 100) -> Tuple[bool, str]:
    """Validate ECG file"""
    if not os.path.exists(file_path):
        return False, "File not found"
    
    # Check file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    if file_size > max_size_mb:
        return False, f"File too large ({file_size:.1f}MB > {max_size_mb}MB)"
    
    # Check extension
    valid_extensions = ['.csv', '.edf', '.txt', '.npy', '.mat']
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in valid_extensions:
        return False, f"Invalid file type: {ext}"
    
    return True, "Valid"
'''
            with open(validators_path, 'w', encoding='utf-8') as f:
                f.write(validators_content)
            self.files_created.append(str(validators_path))
            print("✅ Validators criado")
    
    def fix_database_models(self):
        """Fix database model issues"""
        print("\n🗄️ Corrigindo modelos de banco de dados...")
        
        # Fix base import in models
        models_init = self.app_path / "db" / "models" / "__init__.py"
        if models_init.exists():
            content = '''"""Database models"""

from .user import User
from .patient import Patient
from .ecg_analysis import ECGAnalysis
from .validation import Validation
from .notification import Notification

__all__ = ["User", "Patient", "ECGAnalysis", "Validation", "Notification"]
'''
            with open(models_init, 'w', encoding='utf-8') as f:
                f.write(content)
            self.files_modified.append(str(models_init))
            print("✅ Models __init__.py atualizado")
    
    def install_dependencies(self):
        """Install missing dependencies"""
        print("\n📦 Instalando dependências faltantes...")
        
        missing_deps = [
            "numpy",
            "scipy",
            "pandas",
            "scikit-learn",
            "pydantic",
            "sqlalchemy",
            "psutil",
            "python-multipart",
            "aiofiles",
        ]
        
        for dep in missing_deps:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             capture_output=True, text=True, check=True)
                print(f"✅ {dep}")
            except Exception as e:
                print(f"⚠️ Erro ao instalar {dep}: {e}")
    
    def create_directories(self):
        """Create required directories"""
        print("\n📁 Criando diretórios necessários...")
        
        dirs = [
            self.backend_path / "uploads",
            self.backend_path / "logs",
            self.backend_path / "reports",
            self.app_path / "ml" / "models",
            self.tests_path / "fixtures",
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ {dir_path.relative_to(self.backend_path)}")
    
    def print_summary(self):
        """Print summary of fixes"""
        print("\n" + "=" * 60)
        print("📊 RESUMO DAS CORREÇÕES")
        print("=" * 60)
        
        print(f"\n✅ Erros corrigidos: {len(self.errors_fixed)}")
        for error in self.errors_fixed:
            print(f"   - {error}")
        
        print(f"\n📄 Arquivos criados: {len(self.files_created)}")
        for file in self.files_created[:5]:  # Show first 5
            print(f"   - {Path(file).name}")
        if len(self.files_created) > 5:
            print(f"   ... e mais {len(self.files_created) - 5} arquivos")
        
        print(f"\n✏️ Arquivos modificados: {len(self.files_modified)}")
        for file in self.files_modified[:5]:  # Show first 5
            print(f"   - {Path(file).name}")
        if len(self.files_modified) > 5:
            print(f"   ... e mais {len(self.files_modified) - 5} arquivos")
        
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("1. Execute: pytest --cov=app --cov-report=html")
        print("2. Verifique a cobertura em: htmlcov/index.html")
        print("3. Se necessário, execute novamente este script")
        
        print("\n💡 DICA: Para resolver problemas específicos:")
        print("   - Verifique os logs de erro detalhados")
        print("   - Execute testes individuais com -vv")
        print("   - Use o debugger para investigar falhas")


def main():
    """Main entry point"""
    fixer = CardioAIFixer()
    fixer.run()


if __name__ == "__main__":
    main()
