#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para corrigir erro de indentação no ECGAnalysisService
"""

import re
from pathlib import Path

def fix_ecg_service_indentation():
    """Corrige erro de indentação no ECGAnalysisService."""
    
    service_file = Path("app/services/ecg_service.py")
    
    if not service_file.exists():
        print("[ERRO] Arquivo ecg_service.py não encontrado!")
        return False
    
    print(f"[INFO] Corrigindo indentação em {service_file}...")
    
    try:
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Backup
        with open(service_file.with_suffix('.py.backup'), 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Corrigir indentação do método __init__
        # Procurar por padrões problemáticos de indentação
        lines = content.split('\n')
        fixed_lines = []
        in_class = False
        class_indent = 0
        
        for i, line in enumerate(lines):
            # Detectar início de classe
            if line.strip().startswith('class ECGAnalysisService'):
                in_class = True
                class_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line)
                continue
            
            # Se estamos dentro da classe
            if in_class:
                # Se é uma definição de método
                if line.strip().startswith('def '):
                    # Garantir indentação correta (4 espaços após a classe)
                    method_line = line.strip()
                    correct_indent = ' ' * (class_indent + 4)
                    fixed_lines.append(correct_indent + method_line)
                elif line.strip() and not line[0].isspace() and not line.strip().startswith('#'):
                    # Se não tem indentação e não é comentário, provavelmente saímos da classe
                    in_class = False
                    fixed_lines.append(line)
                else:
                    # Manter outras linhas como estão
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # Juntar linhas corrigidas
        fixed_content = '\n'.join(fixed_lines)
        
        # Garantir que o __init__ está correto
        init_pattern = r'(\s*)def __init__\('
        init_match = re.search(init_pattern, fixed_content)
        
        if init_match:
            # Verificar se a indentação está correta
            current_indent = len(init_match.group(1))
            if current_indent % 4 != 0 or current_indent == 0:
                print(f"[INFO] Corrigindo indentação do __init__ (atual: {current_indent} espaços)")
                # Encontrar a classe para determinar a indentação correta
                class_match = re.search(r'^(class ECGAnalysisService.*?:)$', fixed_content, re.MULTILINE)
                if class_match:
                    # __init__ deve ter 4 espaços a mais que a classe
                    correct_indent = '    '  # 4 espaços
                    fixed_content = re.sub(
                        r'^\s*def __init__\(',
                        correct_indent + 'def __init__(',
                        fixed_content,
                        flags=re.MULTILINE
                    )
        
        # Salvar conteúdo corrigido
        with open(service_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print("[OK] Indentação corrigida!")
        return True
        
    except Exception as e:
        print(f"[ERRO] Falha ao corrigir indentação: {e}")
        
        # Tentar restaurar backup
        backup_file = service_file.with_suffix('.py.backup')
        if backup_file.exists():
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_content = f.read()
                with open(service_file, 'w', encoding='utf-8') as f:
                    f.write(backup_content)
                print("[INFO] Backup restaurado")
            except:
                pass
        
        return False


def verify_syntax():
    """Verifica se o arquivo está sintaticamente correto."""
    import ast
    
    service_file = Path("app/services/ecg_service.py")
    
    try:
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Tentar fazer parse do arquivo
        ast.parse(content)
        print("[OK] Sintaxe verificada com sucesso!")
        return True
    except SyntaxError as e:
        print(f"[ERRO] Erro de sintaxe na linha {e.lineno}: {e.msg}")
        print(f"       Texto: {e.text}")
        return False
    except Exception as e:
        print(f"[ERRO] Erro ao verificar sintaxe: {e}")
        return False


def main():
    """Função principal."""
    print("="*60)
    print("CORRIGINDO ERRO DE INDENTAÇÃO - ECGAnalysisService")
    print("="*60)
    print()
    
    # Corrigir indentação
    if fix_ecg_service_indentation():
        # Verificar sintaxe
        if verify_syntax():
            print("\n[SUCESSO] Arquivo corrigido e sintaxe verificada!")
            print("\nAgora execute novamente os testes:")
            print("  pytest tests/test_ecg_service_critical_coverage.py -v")
        else:
            print("\n[AVISO] Arquivo corrigido mas ainda há erros de sintaxe")
            print("Verifique manualmente o arquivo app/services/ecg_service.py")
    else:
        print("\n[ERRO] Não foi possível corrigir o arquivo")
        print("Verifique manualmente o arquivo app/services/ecg_service.py")


if __name__ == "__main__":
    main()
