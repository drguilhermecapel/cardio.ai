#!/usr/bin/env python3
"""
Script para corrigir o erro final: NameError: name 'np' is not defined
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

print(f"{BLUE}{'='*60}{RESET}")
print(f"{BLUE}Fix Numpy Error Final{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

# 1. Analisar ecg_service.py
print_step("1. Analisando ecg_service.py...")

ecg_service_file = Path("app/services/ecg_service.py")
if not ecg_service_file.exists():
    print_error("ecg_service.py não encontrado!")
    exit(1)

# Backup
backup_file = ecg_service_file.with_suffix('.py.backup_numpy')
with open(ecg_service_file, 'r', encoding='utf-8') as f:
    content = f.read()
with open(backup_file, 'w', encoding='utf-8') as f:
    f.write(content)
print_success(f"Backup criado: {backup_file}")

# 2. Verificar se numpy está importado
print_step("2. Verificando import do numpy...")

if "import numpy as np" not in content:
    print_info("Numpy não está importado, adicionando...")
    
    # Adicionar import do numpy no início, após outros imports
    lines = content.split('\n')
    import_added = False
    
    for i, line in enumerate(lines):
        # Adicionar após outros imports padrão
        if line.startswith('import ') or line.startswith('from '):
            # Procurar onde termina a seção de imports
            j = i
            while j < len(lines) and (lines[j].startswith('import ') or 
                                      lines[j].startswith('from ') or 
                                      lines[j].strip() == ''):
                j += 1
            
            # Inserir numpy import
            if not import_added:
                lines.insert(j, 'import numpy as np')
                lines.insert(j + 1, '')  # Linha em branco
                import_added = True
                break
    
    if not import_added:
        # Se não encontrou imports, adicionar no início
        lines.insert(0, 'import numpy as np')
        lines.insert(1, '')
    
    content = '\n'.join(lines)
    print_success("Import do numpy adicionado")
else:
    print_info("Numpy já está importado")

# 3. Corrigir uso de np.ndarray em anotações de tipo dentro de classes
print_step("3. Corrigindo anotações de tipo com numpy...")

# Procurar por padrões como "campo: np.ndarray" dentro de classes
# e substituir por "campo: 'np.ndarray'" ou usar Any
lines = content.split('\n')
fixed_lines = []
inside_class = False
class_indent = 0

# Adicionar imports necessários para typing
if "from typing import" in content:
    # Verificar se Any está importado
    typing_import_pattern = r'from typing import (.+)'
    match = re.search(typing_import_pattern, content)
    if match and "Any" not in match.group(1):
        content = content.replace(
            match.group(0),
            match.group(0).rstrip() + ", Any"
        )

for i, line in enumerate(lines):
    # Detectar início de classe
    if re.match(r'^class\s+\w+.*:', line):
        inside_class = True
        class_indent = len(line) - len(line.lstrip())
        fixed_lines.append(line)
        continue
    
    # Detectar fim de classe
    if inside_class and line and not line.startswith(' '):
        inside_class = False
    
    # Se estamos dentro de uma classe e encontramos np.ndarray
    if inside_class and 'np.ndarray' in line and ':' in line:
        # Verificar se é uma anotação de tipo de campo
        if re.match(r'^\s+\w+\s*:\s*.*np\.ndarray', line):
            print_info(f"Corrigindo anotação na linha {i+1}: {line.strip()}")
            
            # Opção 1: Usar string literal
            # line = line.replace('np.ndarray', '"np.ndarray"')
            
            # Opção 2: Usar Any (mais seguro)
            line = re.sub(r':\s*np\.ndarray', ': Any', line)
            line = re.sub(r':\s*Optional\[np\.ndarray\]', ': Optional[Any]', line)
            line = re.sub(r':\s*List\[np\.ndarray\]', ': List[Any]', line)
    
    fixed_lines.append(line)

content = '\n'.join(fixed_lines)

# 4. Procurar especificamente pela linha 420 mencionada no erro
print_step("4. Verificando linha específica do erro (linha 420)...")

lines = content.split('\n')
if len(lines) >= 420:
    error_line = lines[419]  # Linha 420 (índice 419)
    print_info(f"Linha 420: {error_line.strip()}")
    
    # Se for um método ou campo com np.ndarray
    if 'np.ndarray' in error_line:
        # Se for dentro de uma definição de função/método, está OK
        # Se for um campo de classe, precisa ser corrigido
        
        # Verificar contexto - olhar linhas anteriores para ver se estamos em uma classe
        in_class_context = False
        for j in range(max(0, 419-20), 419):
            if re.match(r'^class\s+\w+.*:', lines[j]):
                in_class_context = True
                break
            elif re.match(r'^def\s+\w+.*:', lines[j]) or re.match(r'^async\s+def\s+\w+.*:', lines[j]):
                in_class_context = False
                break
        
        if in_class_context:
            print_info("Linha está dentro de uma classe, corrigindo...")
            lines[419] = lines[419].replace('np.ndarray', 'Any')
            content = '\n'.join(lines)

# 5. Alternativa: mover campos problemáticos para fora da classe
print_step("5. Verificando estrutura da classe ECGStatistics...")

# Procurar pela classe ECGStatistics e analisar seu conteúdo
ecg_stats_match = re.search(r'(class ECGStatistics\(BaseModel\):.*?)(?=\nclass|\ndef|\nasync def|\Z)', content, re.DOTALL)

if ecg_stats_match:
    class_content = ecg_stats_match.group(0)
    print_info("Classe ECGStatistics encontrada")
    
    # Se houver métodos dentro da classe que não deveriam estar lá
    if re.search(r'\n\s+def\s+\w+.*\(.*\).*:', class_content):
        print_info("Métodos encontrados dentro da classe, movendo para fora...")
        
        # Extrair apenas os campos da classe
        lines = class_content.split('\n')
        class_fields = []
        class_methods = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if i == 0:  # Linha de definição da classe
                class_fields.append(line)
            elif line.strip().startswith('"""') and i < 5:  # Docstring
                # Capturar toda a docstring
                class_fields.append(line)
                i += 1
                while i < len(lines) and '"""' not in lines[i]:
                    class_fields.append(lines[i])
                    i += 1
                if i < len(lines):
                    class_fields.append(lines[i])
            elif re.match(r'^\s+\w+\s*:', line):  # Campo
                class_fields.append(line)
            elif re.match(r'^\s+def\s+', line) or re.match(r'^\s+async\s+def\s+', line):  # Método
                # Capturar todo o método
                method_indent = len(line) - len(line.lstrip())
                method_lines = [line[4:]]  # Remover indentação extra
                i += 1
                while i < len(lines) and (not lines[i].strip() or 
                                          (lines[i].strip() and 
                                           len(lines[i]) - len(lines[i].lstrip()) > method_indent)):
                    method_lines.append(lines[i][4:] if len(lines[i]) > 4 else lines[i])
                    i += 1
                i -= 1  # Voltar uma linha
                class_methods.extend(method_lines)
            else:
                if line.strip():  # Ignorar linhas vazias
                    class_fields.append(line)
            i += 1
        
        # Reconstruir: classe com apenas campos + métodos fora
        new_class = '\n'.join(class_fields)
        if class_methods:
            new_class += '\n\n\n' + '\n'.join(class_methods)
        
        content = content.replace(class_content, new_class)

# 6. Salvar arquivo corrigido
with open(ecg_service_file, 'w', encoding='utf-8') as f:
    f.write(content)
print_success("ecg_service.py corrigido")

# 7. Verificar sintaxe
print_step("6. Verificando sintaxe...")

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
print("1. Execute o teste:")
print("   pytest tests/test_ecg_service_critical_coverage.py -v")
print("\n2. Se funcionar, execute todos os testes com cobertura:")
print("   pytest --cov=app --cov-report=term-missing --cov-report=html")

# Verificação final
print(f"\n{YELLOW}Verificação final:{RESET}")

# Verificar se numpy está importado
if "import numpy as np" in content:
    print_success("✓ Numpy importado corretamente")
else:
    print_error("✗ Numpy não foi importado")

# Verificar se ainda há np.ndarray problemático
lines = content.split('\n')
problems_found = False
for i, line in enumerate(lines):
    if re.match(r'^\s+\w+\s*:\s*np\.ndarray', line):
        print_error(f"✗ Ainda há problema na linha {i+1}: {line.strip()}")
        problems_found = True

if not problems_found:
    print_success("✓ Nenhum uso problemático de np.ndarray encontrado em campos de classe")

print(f"\n{GREEN}Script concluído!{RESET}")
