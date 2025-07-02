#!/usr/bin/env python3
"""
Script final para corrigir TODOS os problemas de importação
"""

import os
from pathlib import Path
import re

# Cores
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_step(msg):
    print(f"\n{BLUE}[STEP]{RESET} {msg}")

def print_success(msg):
    print(f"{GREEN}[SUCCESS]{RESET} {msg}")

def print_error(msg):
    print(f"{RED}[ERROR]{RESET} {msg}")

def print_warning(msg):
    print(f"{YELLOW}[WARNING]{RESET} {msg}")

print(f"{BLUE}{'='*60}{RESET}")
print(f"{BLUE}CardioAI Pro - Fix All Imports Final{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

# 1. Criar app/models/ecg.py se não existir
print_step("1. Criando app/models/ecg.py...")

ecg_models_file = Path("app/models/ecg.py")
if not ecg_models_file.exists():
    ecg_models_content = '''"""
ECG models - importado de constants para compatibilidade
"""

# Re-exportar de constants para manter compatibilidade
from app.core.constants import (
    FileType,
    ClinicalUrgency,
    AnalysisStatus as ProcessingStatus,  # Alias para compatibilidade
    DiagnosisCategory as RhythmType,     # Alias para compatibilidade
)

# Adicionar valores extras se necessário
class ECGLeadType:
    """ECG lead types"""
    I = "I"
    II = "II"
    III = "III"
    AVR = "aVR"
    AVL = "aVL"
    AVF = "aVF"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"

__all__ = ["FileType", "ProcessingStatus", "ClinicalUrgency", "RhythmType", "ECGLeadType"]
'''
    ecg_models_file.parent.mkdir(parents=True, exist_ok=True)
    with open(ecg_models_file, 'w', encoding='utf-8') as f:
        f.write(ecg_models_content)
    print_success("app/models/ecg.py criado")
else:
    print_warning("app/models/ecg.py já existe")

# 2. Verificar e criar ECGReportResponse se necessário
print_step("2. Verificando schemas...")

ecg_schemas_file = Path("app/schemas/ecg.py")
if not ecg_schemas_file.exists():
    ecg_schemas_content = '''"""
ECG schemas
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

class ECGReportResponse(BaseModel):
    """ECG report response schema"""
    report_id: str
    analysis_id: str
    format: str = "pdf"
    content: bytes = b""
    filename: str
    generated_at: datetime = None
    
    class Config:
        orm_mode = True
        
    def __init__(self, **data):
        if 'generated_at' not in data or data['generated_at'] is None:
            data['generated_at'] = datetime.utcnow()
        super().__init__(**data)

class ECGAnalysisRequest(BaseModel):
    """ECG analysis request schema"""
    patient_id: int
    file_data: bytes
    file_name: str
    analysis_type: str = "standard"
    
class ECGAnalysisResponse(BaseModel):
    """ECG analysis response schema"""
    analysis_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        orm_mode = True
'''
    ecg_schemas_file.parent.mkdir(parents=True, exist_ok=True)
    with open(ecg_schemas_file, 'w', encoding='utf-8') as f:
        f.write(ecg_schemas_content)
    print_success("app/schemas/ecg.py criado")
else:
    # Verificar se ECGReportResponse existe no arquivo
    with open(ecg_schemas_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "ECGReportResponse" not in content:
        print_warning("ECGReportResponse não encontrado, adicionando...")
        
        # Adicionar ECGReportResponse
        ecg_report_class = '''

class ECGReportResponse(BaseModel):
    """ECG report response schema"""
    report_id: str
    analysis_id: str
    format: str = "pdf"
    content: bytes = b""
    filename: str
    generated_at: datetime = None
    
    class Config:
        orm_mode = True
        
    def __init__(self, **data):
        if 'generated_at' not in data or data['generated_at'] is None:
            data['generated_at'] = datetime.utcnow()
        super().__init__(**data)
'''
        
        # Adicionar imports necessários se não existirem
        if "from datetime import datetime" not in content:
            content = "from datetime import datetime\n" + content
        
        content += ecg_report_class
        
        with open(ecg_schemas_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print_success("ECGReportResponse adicionado")
    else:
        print_success("ECGReportResponse já existe")

# 3. Corrigir imports no ecg_service.py
print_step("3. Corrigindo imports no ecg_service.py...")

ecg_service_file = Path("app/services/ecg_service.py")
if ecg_service_file.exists():
    with open(ecg_service_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Atualizar import problemático
    old_import = "from app.models.ecg import FileType, ProcessingStatus, ClinicalUrgency, RhythmType"
    new_import = "from app.core.constants import FileType, AnalysisStatus as ProcessingStatus, ClinicalUrgency, DiagnosisCategory as RhythmType"
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        with open(ecg_service_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print_success("Import corrigido em ecg_service.py")
    else:
        print_warning("Import já estava correto ou não encontrado")

# 4. Garantir que todas as constantes existam
print_step("4. Verificando constantes...")

constants_file = Path("app/core/constants.py")
if constants_file.exists():
    with open(constants_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Lista de constantes necessárias
    required_constants = {
        "FileType": ["CSV", "PDF", "XML", "TXT"],
        "AnalysisStatus": ["PENDING", "PROCESSING", "COMPLETED", "FAILED"],
        "ClinicalUrgency": ["ROUTINE", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "DiagnosisCategory": ["NORMAL", "ARRHYTHMIA", "ISCHEMIA", "OTHER"]
    }
    
    modified = False
    
    for enum_name, values in required_constants.items():
        if f"class {enum_name}" not in content:
            print_warning(f"{enum_name} não encontrado, criando...")
            # Criar enum
            enum_def = f"\n\nclass {enum_name}(str, Enum):\n    \"\"\"{enum_name}\"\"\"\n"
            for value in values:
                enum_def += f"    {value} = \"{value.lower()}\"\n"
            content += enum_def
            modified = True
        else:
            # Verificar valores individuais
            for value in values:
                pattern = f"{value}\\s*=\\s*[\"']"
                if not re.search(pattern, content):
                    print_warning(f"{enum_name}.{value} não encontrado, adicionando...")
                    # Adicionar valor ao enum existente
                    enum_pattern = f"class {enum_name}.*?(?=class|\\Z)"
                    match = re.search(enum_pattern, content, re.DOTALL)
                    if match:
                        enum_content = match.group(0)
                        # Adicionar antes do próximo class ou no final
                        insert_pos = match.end() - 5 if match.group(0).endswith("class") else match.end()
                        new_value = f"\n    {value} = \"{value.lower()}\""
                        content = content[:insert_pos] + new_value + content[insert_pos:]
                        modified = True
    
    if modified:
        with open(constants_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print_success("Constantes atualizadas")
    else:
        print_success("Todas as constantes já existem")

# 5. Criar arquivos __init__.py necessários
print_step("5. Verificando arquivos __init__.py...")

init_dirs = [
    "app",
    "app/models",
    "app/schemas",
    "app/services",
    "app/core",
    "app/utils",
    "app/api",
    "app/api/v1",
    "app/api/v1/endpoints"
]

for dir_path in init_dirs:
    init_file = Path(dir_path) / "__init__.py"
    if not init_file.exists():
        init_file.parent.mkdir(parents=True, exist_ok=True)
        init_file.touch()
        print_success(f"Criado {init_file}")

# 6. Criar script de teste simplificado
print_step("6. Criando script de teste simplificado...")

test_script = '''#!/usr/bin/env python3
"""
Script de teste simplificado
"""

import subprocess
import sys

print("\\n=== Testando importações ===\\n")

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

print("\\n=== Executando teste específico ===\\n")

# Executar teste
result = subprocess.run([
    sys.executable, "-m", "pytest",
    "tests/test_ecg_service_critical_coverage.py",
    "-v", "-s"
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("\\nERROS:")
    print(result.stderr)
'''

with open("test_imports.py", 'w', encoding='utf-8') as f:
    f.write(test_script)

print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{GREEN}CORREÇÕES APLICADAS COM SUCESSO!{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

print(f"\n{YELLOW}PRÓXIMOS PASSOS:{RESET}")
print("1. Execute o teste de importações:")
print("   python test_imports.py")
print("\n2. Se tudo estiver OK, execute o teste completo:")
print("   pytest tests/test_ecg_service_critical_coverage.py -v")
print("\n3. Para executar todos os testes com cobertura:")
print("   pytest --cov=app --cov-report=term-missing --cov-report=html")

# Verificar sintaxe de arquivos críticos
print(f"\n{YELLOW}Verificando sintaxe dos arquivos modificados...{RESET}")

import ast

files_to_check = [
    ("app/services/ecg_service.py", "ECGAnalysisService"),
    ("app/models/ecg.py", "ECG Models"),
    ("app/schemas/ecg.py", "ECG Schemas"),
    ("app/core/constants.py", "Constants")
]

all_valid = True
for file_path, name in files_to_check:
    if Path(file_path).exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            print_success(f"{name} - sintaxe válida")
        except SyntaxError as e:
            print_error(f"{name} - erro de sintaxe: {e}")
            all_valid = False

if all_valid:
    print(f"\n{GREEN}✓ Todos os arquivos têm sintaxe válida!{RESET}")
else:
    print(f"\n{RED}✗ Alguns arquivos têm erros de sintaxe. Verifique os erros acima.{RESET}")
