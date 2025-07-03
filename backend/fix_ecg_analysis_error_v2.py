#!/usr/bin/env python3
"""
Script para corrigir o erro NameError: name 'ECGAnalysis' is not defined
Versão corrigida com print_warning definido
"""

import re
from pathlib import Path

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

def print_info(msg):
    print(f"{YELLOW}[INFO]{RESET} {msg}")

def print_warning(msg):
    print(f"{YELLOW}[WARNING]{RESET} {msg}")

print(f"{BLUE}{'='*60}{RESET}")
print(f"{BLUE}Fix ECGAnalysis NameError (v2){RESET}")
print(f"{BLUE}{'='*60}{RESET}")

# 1. Verificar o problema no ecg_service.py
print_step("1. Analisando ecg_service.py...")

ecg_service_file = Path("app/services/ecg_service.py")
if not ecg_service_file.exists():
    print_error("ecg_service.py não encontrado!")
    exit(1)

# Backup
backup_file = ecg_service_file.with_suffix('.py.backup_ecganalysis_v2')
with open(ecg_service_file, 'r', encoding='utf-8') as f:
    content = f.read()
with open(backup_file, 'w', encoding='utf-8') as f:
    f.write(content)
print_success(f"Backup criado: {backup_file}")

# 2. Verificar imports
print_step("2. Verificando imports...")

# Verificar se ECGAnalysis está importado
if "from app.models.ecg_analysis import ECGAnalysis" not in content and \
   "from app.models import ECGAnalysis" not in content:
    print_warning("ECGAnalysis não está importado")
    
    # Adicionar import
    import_line = "from app.models.ecg_analysis import ECGAnalysis"
    
    # Encontrar onde adicionar o import (após outros imports de models)
    import_section = re.search(r'(from app\.models.*?\n)+', content)
    if import_section:
        # Adicionar após outros imports de models
        insert_pos = import_section.end()
        content = content[:insert_pos] + import_line + "\n" + content[insert_pos:]
    else:
        # Adicionar após a linha de import do BaseModel
        basemodel_import = re.search(r'from pydantic import.*BaseModel.*\n', content)
        if basemodel_import:
            insert_pos = basemodel_import.end()
            content = content[:insert_pos] + "\n" + import_line + "\n" + content[insert_pos:]
        else:
            # Adicionar após os primeiros imports
            first_import = re.search(r'^import.*\n|^from.*\n', content, re.MULTILINE)
            if first_import:
                # Encontrar o fim da seção de imports
                lines = content.split('\n')
                import_end = 0
                for i, line in enumerate(lines):
                    if line.strip() and not (line.startswith('import ') or line.startswith('from ')):
                        if i > 0 and not lines[i-1].strip():  # linha vazia antes
                            import_end = i
                            break
                
                if import_end > 0:
                    lines.insert(import_end, import_line)
                    content = '\n'.join(lines)
                else:
                    content = import_line + "\n" + content
            else:
                # Adicionar no início
                content = import_line + "\n" + content
    
    print_success("Import de ECGAnalysis adicionado")
else:
    print_info("ECGAnalysis já está importado")

# 3. Verificar classe ECGStatistics e corrigir problemas estruturais
print_step("3. Analisando estrutura de classes e métodos...")

# Procurar por métodos que estão dentro de classes quando deveriam estar fora
lines = content.split('\n')
fixed_lines = []
inside_class = False
class_indent = 0
i = 0

while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Detectar início de classe
    if re.match(r'^class\s+\w+.*:', line):
        inside_class = True
        class_indent = len(line) - len(line.lstrip())
        fixed_lines.append(line)
        i += 1
        continue
    
    # Detectar fim de classe (linha não indentada ou outra classe/função)
    if inside_class and line and not line.startswith(' '):
        inside_class = False
    
    # Se estamos dentro de uma classe
    if inside_class:
        line_indent = len(line) - len(line.lstrip()) if line.strip() else 1000
        
        # Se encontrar uma definição de método com tipo de retorno ECGAnalysis
        if re.match(r'^\s+.*def\s+\w+.*->\s*ECGAnalysis\s*:', line):
            # Este método provavelmente deveria estar fora da classe
            print_warning(f"Método encontrado dentro de classe na linha {i+1}: {stripped[:50]}...")
            
            # Verificar se é um método de classe ou deveria estar no nível do módulo
            # Se tem 'self' como parâmetro, é método de classe
            if 'self' in line:
                fixed_lines.append(line)
            else:
                # Mover para fora da classe (remover indentação extra)
                new_line = line[class_indent+4:] if len(line) > class_indent+4 else line.lstrip()
                
                # Adicionar linha vazia antes se necessário
                if fixed_lines and fixed_lines[-1].strip():
                    fixed_lines.append('')
                
                # Marcar que saímos da classe
                inside_class = False
                fixed_lines.append(new_line)
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)
    
    i += 1

content = '\n'.join(fixed_lines)

# 4. Criar modelo ECGAnalysis se não existir
print_step("4. Verificando modelo ECGAnalysis...")

ecg_analysis_model = Path("app/models/ecg_analysis.py")
if not ecg_analysis_model.exists():
    print_warning("Modelo ECGAnalysis não existe, criando...")
    
    ecg_analysis_content = '''"""
ECG Analysis model.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float, Text
from sqlalchemy.orm import relationship

from app.models.base import Base
from app.core.constants import AnalysisStatus, ClinicalUrgency, DiagnosisCategory


class ECGAnalysis(Base):
    """ECG Analysis model."""
    
    __tablename__ = "ecg_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    analysis_id = Column(String, unique=True, index=True)
    file_path = Column(String, nullable=False)
    original_filename = Column(String)
    file_hash = Column(String)
    file_size_bytes = Column(Integer)
    
    status = Column(String, default=AnalysisStatus.PENDING)
    clinical_urgency = Column(String, default=ClinicalUrgency.ROUTINE)
    diagnosis_category = Column(String, default=DiagnosisCategory.NORMAL)
    
    signal_quality_score = Column(Float)
    confidence_score = Column(Float)
    
    measurements = Column(JSON)
    diagnosis = Column(Text)
    findings = Column(JSON)
    recommendations = Column(JSON)
    
    processing_time_seconds = Column(Float)
    ml_model_version = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    patient = relationship("Patient", back_populates="ecg_analyses")
    validations = relationship("Validation", back_populates="ecg_analysis")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "analysis_id": self.analysis_id,
            "status": self.status,
            "clinical_urgency": self.clinical_urgency,
            "diagnosis_category": self.diagnosis_category,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "measurements": self.measurements,
            "diagnosis": self.diagnosis,
            "findings": self.findings,
            "confidence_score": self.confidence_score
        }
'''
    
    ecg_analysis_model.parent.mkdir(parents=True, exist_ok=True)
    with open(ecg_analysis_model, 'w', encoding='utf-8') as f:
        f.write(ecg_analysis_content)
    print_success("Modelo ECGAnalysis criado")
else:
    print_info("Modelo ECGAnalysis já existe")

# 5. Criar base model se não existir
base_model_file = Path("app/models/base.py")
if not base_model_file.exists():
    print_warning("Base model não existe, criando...")
    
    base_content = '''"""
Base model for SQLAlchemy.
"""

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
'''
    
    base_model_file.parent.mkdir(parents=True, exist_ok=True)
    with open(base_model_file, 'w', encoding='utf-8') as f:
        f.write(base_content)
    print_success("Base model criado")

# 6. Adicionar __init__.py no models se não existir
models_init = Path("app/models/__init__.py")
if not models_init.exists():
    print_warning("__init__.py não existe em models, criando...")
    
    init_content = '''"""
Database models.
"""

from app.models.base import Base
from app.models.ecg_analysis import ECGAnalysis

__all__ = ["Base", "ECGAnalysis"]
'''
    
    with open(models_init, 'w', encoding='utf-8') as f:
        f.write(init_content)
    print_success("models/__init__.py criado")

# 7. Salvar arquivo corrigido
with open(ecg_service_file, 'w', encoding='utf-8') as f:
    f.write(content)
print_success("ecg_service.py atualizado")

# 8. Verificar sintaxe
print_step("5. Verificando sintaxe...")

import ast
try:
    ast.parse(content)
    print_success("Sintaxe válida!")
except SyntaxError as e:
    print_error(f"Erro de sintaxe: {e}")
    print_info(f"Linha {e.lineno}: {e.text}")
    
    # Mostrar contexto
    lines = content.split('\n')
    start = max(0, e.lineno - 5)
    end = min(len(lines), e.lineno + 5)
    
    print("\nContexto do erro:")
    for i in range(start, end):
        prefix = ">>>" if i == e.lineno - 1 else "   "
        print(f"{prefix} {i+1:4d}: {lines[i]}")

print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{GREEN}Correções aplicadas!{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

print(f"\n{YELLOW}PRÓXIMOS PASSOS:{RESET}")
print("1. Execute o script fix_all_imports_final.py se ainda não executou:")
print("   python fix_all_imports_final.py")
print("\n2. Teste novamente:")
print("   pytest tests/test_ecg_service_critical_coverage.py -v")
print("\n3. Se ainda houver erros, execute todos os testes para ver o progresso:")
print("   pytest --cov=app --cov-report=term-missing -x")

# Verificar se os arquivos foram criados
print(f"\n{YELLOW}Verificando arquivos criados:{RESET}")
files_to_check = [
    "app/models/ecg_analysis.py",
    "app/models/base.py",
    "app/models/__init__.py"
]

for file_path in files_to_check:
    if Path(file_path).exists():
        print_success(f"✓ {file_path}")
    else:
        print_error(f"✗ {file_path}")
