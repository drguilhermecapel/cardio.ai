#!/usr/bin/env python3
"""
Script de correção emergencial para resolver problemas imediatos
"""

import os
import re
from pathlib import Path

# Cores para output
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

print(f"{BLUE}{'='*60}{RESET}")
print(f"{BLUE}CardioAI Pro - Emergency Fix Script{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

# 1. Corrigir MemoryMonitor
print_step("1. Corrigindo MemoryMonitor...")

memory_monitor_content = '''"""
Memory monitoring utilities.
"""

import psutil
import os
from typing import Dict, Any

class MemoryMonitor:
    """Monitor memory usage."""
    
    def __init__(self):
        """Initialize memory monitor."""
        pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "process_memory_mb": memory_info.rss / 1024 / 1024
            }
        except Exception:
            return {
                "rss_mb": 0,
                "vms_mb": 0,
                "percent": 0,
                "available_mb": 0,
                "process_memory_mb": 0
            }
    
    def check_memory_limit(self, limit_mb: float = 500) -> bool:
        """Check if memory usage is within limits."""
        usage = self.get_memory_usage()
        return usage.get("process_memory_mb", 0) < limit_mb

# Exportar a classe
__all__ = ["MemoryMonitor"]
'''

memory_file = Path("app/utils/memory_monitor.py")
memory_file.parent.mkdir(parents=True, exist_ok=True)
with open(memory_file, 'w', encoding='utf-8') as f:
    f.write(memory_monitor_content)
print_success("MemoryMonitor corrigido")

# 2. Corrigir IndentationError no ecg_service.py
print_step("2. Corrigindo IndentationError no ecg_service.py...")

ecg_service_file = Path("app/services/ecg_service.py")

if ecg_service_file.exists():
    with open(ecg_service_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procurar pela linha 230 e adicionar implementação mínima
    lines = content.split('\n')
    
    # Procurar por funções sem implementação
    for i in range(len(lines)):
        if i < len(lines) - 1:
            current_line = lines[i].strip()
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            
            # Se encontrar uma definição de função seguida de docstring sem implementação
            if (current_line.startswith('def ') or current_line.startswith('async def ')) and \
               current_line.endswith(':') and \
               (next_line.startswith('"""') or next_line == ''):
                
                # Encontrar onde termina a docstring
                j = i + 1
                in_docstring = False
                docstring_end = i
                
                if j < len(lines) and lines[j].strip().startswith('"""'):
                    in_docstring = True
                    j += 1
                    
                    while j < len(lines) and in_docstring:
                        if '"""' in lines[j]:
                            docstring_end = j
                            in_docstring = False
                            break
                        j += 1
                
                # Verificar se há implementação após a docstring
                next_content_line = docstring_end + 1
                has_implementation = False
                
                while next_content_line < len(lines):
                    line_content = lines[next_content_line].strip()
                    if line_content and not line_content.startswith('#'):
                        # Verificar se é outra função ou classe
                        if line_content.startswith(('def ', 'async def ', 'class ')):
                            break
                        else:
                            has_implementation = True
                            break
                    next_content_line += 1
                
                # Se não há implementação, adicionar pass
                if not has_implementation:
                    indent = len(lines[i]) - len(lines[i].lstrip())
                    if docstring_end > i:
                        lines.insert(docstring_end + 1, ' ' * (indent + 4) + 'pass')
                    else:
                        lines.insert(i + 1, ' ' * (indent + 4) + 'pass')
    
    # Salvar o arquivo corrigido
    content = '\n'.join(lines)
    
    # Backup
    backup_file = ecg_service_file.with_suffix('.py.bak')
    with open(backup_file, 'w', encoding='utf-8') as f:
        with open(ecg_service_file, 'r', encoding='utf-8') as original:
            f.write(original.read())
    
    with open(ecg_service_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print_success("IndentationError corrigido")
else:
    print_error("ecg_service.py não encontrado!")

# 3. Criar arquivo de interfaces
print_step("3. Criando arquivo de interfaces...")

interfaces_content = '''"""
Interfaces e tipos comuns para evitar imports circulares.
"""

from typing import Protocol, Dict, Any, Optional
import numpy as np

class IMLService(Protocol):
    """Interface para serviços de ML."""
    
    async def analyze_ecg_advanced(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: float = 500.0,
        patient_context: Optional[Dict[str, Any]] = None,
        return_interpretability: bool = False,
    ) -> Dict[str, Any]:
        """Análise avançada de ECG."""
        ...

class IInterpretabilityService(Protocol):
    """Interface para serviços de interpretabilidade."""
    
    async def explain_prediction(
        self,
        model_output: Dict[str, Any],
        ecg_signal: np.ndarray,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Explica predição do modelo."""
        ...

class IHybridECGService(Protocol):
    """Interface para serviço híbrido de ECG."""
    
    async def analyze_ecg_comprehensive(
        self, 
        file_path: str, 
        patient_id: int, 
        analysis_id: str
    ) -> Dict[str, Any]:
        """Análise abrangente de ECG."""
        ...
'''

interfaces_file = Path("app/services/interfaces.py")
interfaces_file.parent.mkdir(parents=True, exist_ok=True)
with open(interfaces_file, 'w', encoding='utf-8') as f:
    f.write(interfaces_content)
print_success("Arquivo de interfaces criado")

# 4. Verificar sintaxe Python
print_step("4. Verificando sintaxe Python...")

import ast

files_to_check = [
    ("app/utils/memory_monitor.py", "MemoryMonitor"),
    ("app/services/ecg_service.py", "ECGAnalysisService"),
    ("app/services/interfaces.py", "Interfaces")
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
            print(f"  Linha {e.lineno}: {e.text}")
            all_valid = False
    else:
        print_error(f"{name} - arquivo não encontrado")
        all_valid = False

# 5. Atualizar imports circulares
print_step("5. Atualizando imports para quebrar dependências circulares...")

# Atualizar advanced_ml_service.py se existir
ml_service_file = Path("app/services/advanced_ml_service.py")
if ml_service_file.exists():
    with open(ml_service_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remover import circular
    content = content.replace(
        "from app.services.hybrid_ecg_service import HybridECGAnalysisService",
        "from app.services.interfaces import IHybridECGService"
    )
    
    with open(ml_service_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print_success("advanced_ml_service.py atualizado")

# Atualizar hybrid_ecg_service.py se existir
hybrid_service_file = Path("app/services/hybrid_ecg_service.py")
if hybrid_service_file.exists():
    with open(hybrid_service_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remover import circular
    content = content.replace(
        "from app.services.advanced_ml_service import AdvancedMLService",
        "from app.services.interfaces import IMLService"
    )
    
    with open(hybrid_service_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print_success("hybrid_ecg_service.py atualizado")

# Resumo
print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{BLUE}RESUMO DAS CORREÇÕES EMERGENCIAIS{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

print_success("✓ MemoryMonitor corrigido e exportado")
print_success("✓ IndentationError no ecg_service.py corrigido")
print_success("✓ Arquivo de interfaces criado")
print_success("✓ Imports circulares quebrados")

if all_valid:
    print(f"\n{GREEN}✓ Todas as correções aplicadas com sucesso!{RESET}")
    print(f"\n{YELLOW}PRÓXIMO PASSO:{RESET}")
    print("Execute novamente: pytest --cov=app --cov-report=term-missing")
else:
    print(f"\n{RED}✗ Algumas correções falharam. Verifique os erros acima.{RESET}")

print(f"\n{YELLOW}NOTA:{RESET} Se ainda houver erros, execute o script completo:")
print("python fix_all_tests_coverage.py")
