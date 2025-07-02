#!/usr/bin/env python3
"""
Script simples e direto para adicionar o import de Path
"""

from pathlib import Path

# Cores
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

print(f"{BLUE}{'='*60}{RESET}")
print(f"{BLUE}Fix Path Import - Solução Simples{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

# Arquivo a ser corrigido
ecg_service_file = Path("app/services/ecg_service.py")

if not ecg_service_file.exists():
    print(f"{RED}[ERROR]{RESET} ecg_service.py não encontrado!")
    exit(1)

# Fazer backup
backup_file = ecg_service_file.with_suffix('.py.backup_path_simple')
with open(ecg_service_file, 'r', encoding='utf-8') as f:
    content = f.read()
with open(backup_file, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"{GREEN}[SUCCESS]{RESET} Backup criado: {backup_file}")

# Verificar se Path já está importado
if "from pathlib import Path" in content:
    print(f"{YELLOW}[INFO]{RESET} Path já está importado")
else:
    print(f"{YELLOW}[INFO]{RESET} Adicionando import de Path...")
    
    # Adicionar o import após os primeiros imports
    lines = content.split('\n')
    
    # Encontrar onde inserir
    inserted = False
    for i, line in enumerate(lines):
        # Inserir após o primeiro bloco de imports
        if (line.startswith('import ') or line.startswith('from ')) and i < len(lines) - 1:
            # Verificar se a próxima linha não é um import
            j = i + 1
            while j < len(lines) and (lines[j].startswith('import ') or 
                                      lines[j].startswith('from ') or 
                                      lines[j].strip() == ''):
                j += 1
            
            # Inserir antes da primeira linha não-import
            lines.insert(j, 'from pathlib import Path')
            inserted = True
            break
    
    if not inserted:
        # Se não encontrou, adicionar no início
        lines.insert(0, 'from pathlib import Path')
        lines.insert(1, '')
    
    content = '\n'.join(lines)
    print(f"{GREEN}[SUCCESS]{RESET} Import de Path adicionado")

# Verificar outros imports essenciais
essential_imports = {
    "numpy": "import numpy as np",
    "typing": "from typing import Dict, List, Optional, Any, Union, Tuple",
    "datetime": "from datetime import datetime",
    "logging": "import logging"
}

for name, import_line in essential_imports.items():
    if name == "typing":
        # Verificar se algum import de typing existe
        if "from typing import" not in content:
            print(f"{YELLOW}[INFO]{RESET} Adicionando import de typing...")
            lines = content.split('\n')
            
            # Encontrar onde está o import de Path
            for i, line in enumerate(lines):
                if "from pathlib import Path" in line:
                    lines.insert(i + 1, import_line)
                    break
            
            content = '\n'.join(lines)
    else:
        # Outros imports
        if import_line not in content and f"import {name}" not in content:
            print(f"{YELLOW}[INFO]{RESET} Adicionando import de {name}...")
            lines = content.split('\n')
            
            # Adicionar após pathlib
            for i, line in enumerate(lines):
                if "from pathlib import Path" in line:
                    lines.insert(i + 1, import_line)
                    break
            
            content = '\n'.join(lines)

# Salvar arquivo
with open(ecg_service_file, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"{GREEN}[SUCCESS]{RESET} Arquivo salvo")

# Verificar sintaxe
print(f"\n{YELLOW}[INFO]{RESET} Verificando sintaxe...")
import ast
try:
    ast.parse(content)
    print(f"{GREEN}[SUCCESS]{RESET} Sintaxe válida!")
except SyntaxError as e:
    print(f"{RED}[ERROR]{RESET} Erro de sintaxe na linha {e.lineno}")
    print(f"  {e.text}")

print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{GREEN}Correção concluída!{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

print(f"\n{YELLOW}Agora execute:{RESET}")
print("pytest tests/test_ecg_service_critical_coverage.py -v")
print("\nou para ver todos os testes:")
print("pytest --cov=app --cov-report=term-missing --cov-report=html")
