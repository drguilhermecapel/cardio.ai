#!/usr/bin/env python3
"""
Script para corrigir a indentação específica da função generate_report
"""

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

print(f"{BLUE}{'='*60}{RESET}")
print(f"{BLUE}Fix generate_report Indentation{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

# Caminho do arquivo
ecg_service_file = Path("app/services/ecg_service.py")

if not ecg_service_file.exists():
    print_error(f"Arquivo {ecg_service_file} não encontrado!")
    exit(1)

# Backup
backup_file = ecg_service_file.with_suffix('.py.backup_generate_report')
with open(ecg_service_file, 'r', encoding='utf-8') as f:
    content = f.read()
with open(backup_file, 'w', encoding='utf-8') as f:
    f.write(content)
print_success(f"Backup criado: {backup_file}")

# Ler o arquivo linha por linha
with open(ecg_service_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Procurar e corrigir a função generate_report
print_step("Procurando função generate_report...")

# Encontrar a linha com "async def generate_report("
for i in range(len(lines)):
    if "async def generate_report(" in lines[i]:
        print_success(f"Encontrada na linha {i+1}")
        
        # Verificar a indentação atual
        current_indent = len(lines[i]) - len(lines[i].lstrip())
        print(f"Indentação atual: {current_indent} espaços")
        
        # Corrigir para 4 espaços (indentação de método de classe)
        correct_indent = 4
        
        # Corrigir a definição da função (linhas 230-235)
        j = i
        while j < len(lines) and not lines[j].strip().endswith(':'):
            lines[j] = ' ' * correct_indent + lines[j].lstrip()
            j += 1
        
        # Corrigir a linha que termina com ':'
        if j < len(lines):
            lines[j] = ' ' * correct_indent + lines[j].lstrip()
        
        # Encontrar onde termina a docstring
        j += 1  # Pular para após a definição
        docstring_start = None
        docstring_end = None
        
        # Procurar início da docstring
        while j < len(lines):
            if '"""' in lines[j]:
                docstring_start = j
                # Verificar se fecha na mesma linha
                if lines[j].count('"""') >= 2:
                    docstring_end = j
                break
            j += 1
        
        # Se não fechou na mesma linha, procurar o fim
        if docstring_start is not None and docstring_end is None:
            j = docstring_start + 1
            while j < len(lines):
                if '"""' in lines[j]:
                    docstring_end = j
                    break
                j += 1
        
        # Adicionar implementação após a docstring
        if docstring_end is not None:
            # Verificar se já existe implementação
            k = docstring_end + 1
            has_implementation = False
            
            # Pular linhas em branco
            while k < len(lines) and lines[k].strip() == '':
                k += 1
            
            # Verificar se há código ou é outra função
            if k < len(lines):
                next_line = lines[k]
                next_indent = len(next_line) - len(next_line.lstrip())
                
                # Se a próxima linha tem indentação menor ou igual, não há implementação
                if (next_indent <= correct_indent or 
                    'def ' in next_line or 
                    'class ' in next_line):
                    has_implementation = False
                else:
                    has_implementation = True
            
            if not has_implementation:
                # Adicionar implementação stub
                print_step("Adicionando implementação stub...")
                implementation = f'''{' ' * (correct_indent + 4)}# TODO: Implementar geração de relatório
{' ' * (correct_indent + 4)}from app.schemas.ecg import ECGReportResponse
{' ' * (correct_indent + 4)}
{' ' * (correct_indent + 4)}return ECGReportResponse(
{' ' * (correct_indent + 8)}report_id="temp_id",
{' ' * (correct_indent + 8)}analysis_id=analysis_id,
{' ' * (correct_indent + 8)}format=report_format,
{' ' * (correct_indent + 8)}content=b"",
{' ' * (correct_indent + 8)}filename=f"report_{{analysis_id}}.{{report_format}}"
{' ' * (correct_indent + 4)})
'''
                # Inserir após a docstring
                lines.insert(docstring_end + 1, implementation)
                print_success("Implementação stub adicionada")
        
        break

# Salvar o arquivo corrigido
with open(ecg_service_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print_success("Arquivo corrigido e salvo!")

# Verificar sintaxe
print_step("Verificando sintaxe Python...")
import ast
try:
    with open(ecg_service_file, 'r', encoding='utf-8') as f:
        content = f.read()
    ast.parse(content)
    print_success("Sintaxe Python válida!")
    
    print(f"\n{GREEN}✓ Correção concluída com sucesso!{RESET}")
    print(f"\nAgora execute:")
    print(f"  pytest tests/test_ecg_service_critical_coverage.py -v")
    
except SyntaxError as e:
    print_error(f"Ainda há erro de sintaxe: {e}")
    print_error(f"Linha {e.lineno}: {e.text}")
    
    # Criar correção manual
    print(f"\n{YELLOW}Correção manual necessária:{RESET}")
    print(f"1. Abra o arquivo: app\\services\\ecg_service.py")
    print(f"2. Vá para a linha {e.lineno}")
    print(f"3. Verifique a indentação (deve ser 4 espaços para métodos de classe)")
    print(f"4. Adicione uma implementação mínima após a docstring:")
    print(f"       pass  # ou return None")
