#!/usr/bin/env python3
"""
Script de teste simplificado
"""

import subprocess
import sys

print("\n=== Testando importações ===\n")

# Testar importações principais
try:
    from app.services.ecg_service import ECGAnalysisService
    print("✓ ECGAnalysisService importado com sucesso")
except Exception as e:
    print(f"✗ Erro ao importar ECGAnalysisService: {e}")

try:
    from app.models.ecg import FileType, ProcessingStatus, ClinicalUrgency, RhythmType
    print("✓ Models ECG importados com sucesso")
except Exception as e:
    print(f"✗ Erro ao importar models ECG: {e}")

try:
    from app.schemas.ecg import ECGReportResponse
    print("✓ ECGReportResponse importado com sucesso")
except Exception as e:
    print(f"✗ Erro ao importar ECGReportResponse: {e}")

print("\n=== Executando teste específico ===\n")

# Executar teste
result = subprocess.run([
    sys.executable, "-m", "pytest",
    "tests/test_ecg_service_critical_coverage.py",
    "-v", "-s"
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("\nERROS:")
    print(result.stderr)
