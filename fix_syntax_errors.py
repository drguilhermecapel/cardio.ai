#!/usr/bin/env python3
"""
Script para diagnosticar e corrigir erros de sintaxe em train_ecg.py
"""

import ast
import re
from pathlib import Path
import sys

def check_parentheses_balance(content):
    """Verifica o balanceamento de parênteses, colchetes e chaves"""
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    line_number = 1
    char_position = 0
    
    issues = []
    
    for i, char in enumerate(content):
        if char == '\n':
            line_number += 1
            char_position = 0
        else:
            char_position += 1
        
        if char in pairs:
            stack.append((char, line_number, char_position))
        elif char in pairs.values():
            if not stack:
                issues.append(f"Linha {line_number}, posição {char_position}: '{char}' sem abertura correspondente")
            else:
                opening, open_line, open_pos = stack.pop()
                expected = pairs[opening]
                if char != expected:
                    issues.append(f"Linha {line_number}: Esperado '{expected}' mas encontrado '{char}'")
    
    # Verificar aberturas não fechadas
    for opening, line, pos in stack:
        issues.append(f"Linha {line}: '{opening}' não foi fechado")
    
    return issues

def find_syntax_error_context(file_path, error_line):
    """Mostra o contexto ao redor da linha com erro"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Mostrar 10 linhas antes e depois do erro
        start = max(0, error_line - 10)
        end = min(len(lines), error_line + 5)
        
        print("\n📄 Contexto do erro (linhas {} a {}):".format(start + 1, end))
        print("-" * 60)
        
        for i in range(start, end):
            line_num = i + 1
            prefix = ">>> " if line_num == error_line else "    "
            print(f"{prefix}{line_num:4d}: {lines[i].rstrip()}")
        
        print("-" * 60)
        
        # Analisar a linha específica
        if error_line <= len(lines):
            error_line_content = lines[error_line - 1]
            
            # Verificar problemas comuns
            if error_line_content.strip() == ')':
                print("\n⚠️  Problema detectado: Parêntese de fechamento sozinho na linha")
                print("💡 Solução: Verifique se há um parêntese de abertura correspondente nas linhas anteriores")
                
                # Procurar por parênteses não fechados antes
                open_parens = 0
                for i in range(error_line - 2, max(0, error_line - 50), -1):
                    line = lines[i]
                    open_parens += line.count('(') - line.count(')')
                    if open_parens < 0:
                        print(f"\n🔍 Possível parêntese extra encontrado por volta da linha {i + 1}")
                        break
            
            # Verificar vírgulas extras
            if ',' in lines[error_line - 2] and error_line_content.strip() == ')':
                print("\n⚠️  Possível vírgula extra antes do parêntese de fechamento")
                print("💡 Solução: Remova a vírgula no final da linha anterior")
        
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")

def try_fix_common_issues(file_path):
    """Tenta corrigir problemas comuns automaticamente"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed = False
        
        # Backup
        backup_path = Path(file_path).with_suffix('.py.syntax_backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ Backup criado: {backup_path}")
        
        # Procurar por linhas com apenas ')' 
        for i, line in enumerate(lines):
            if line.strip() == ')':
                # Verificar se a linha anterior termina com vírgula
                if i > 0 and lines[i-1].rstrip().endswith(','):
                    print(f"\n🔧 Corrigindo: Removendo vírgula extra na linha {i}")
                    lines[i-1] = lines[i-1].rstrip()[:-1] + '\n'
                    fixed = True
        
        # Procurar por parênteses desbalanceados em definições de função/classe
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Se é uma definição de função ou classe com múltiplas linhas
            if (line.startswith('def ') or line.startswith('class ')) and line.endswith(':'):
                # Contar parênteses
                j = i
                paren_count = 0
                while j >= 0 and j >= i - 10:  # Verificar até 10 linhas antes
                    paren_count += lines[j].count('(') - lines[j].count(')')
                    j -= 1
                
                if paren_count != 0:
                    print(f"\n⚠️  Parênteses desbalanceados detectados por volta da linha {i + 1}")
            
            i += 1
        
        if fixed:
            # Salvar arquivo corrigido
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print("\n✅ Arquivo corrigido!")
            return True
        else:
            print("\n❌ Não foi possível corrigir automaticamente")
            return False
            
    except Exception as e:
        print(f"Erro ao processar arquivo: {e}")
        return False

def validate_python_syntax(file_path):
    """Valida a sintaxe Python do arquivo"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Tentar compilar o código
        compile(content, file_path, 'exec')
        print("\n✅ Sintaxe Python válida!")
        return True
    except SyntaxError as e:
        print(f"\n❌ Erro de sintaxe encontrado:")
        print(f"   Arquivo: {e.filename}")
        print(f"   Linha: {e.lineno}")
        print(f"   Posição: {e.offset}")
        print(f"   Erro: {e.msg}")
        
        if e.text:
            print(f"   Código: {e.text.strip()}")
            if e.offset:
                print(f"           {' ' * (e.offset - 1)}^")
        
        return False

def main():
    """Função principal"""
    print("="*60)
    print("🔍 DIAGNÓSTICO DE ERROS DE SINTAXE")
    print("="*60)
    
    file_path = "train_ecg.py"
    if not Path(file_path).exists():
        print(f"❌ Arquivo '{file_path}' não encontrado!")
        file_path = input("Digite o caminho completo: ").strip()
    
    # Ler o arquivo
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return
    
    # 1. Verificar balanceamento de parênteses
    print("\n🔍 Verificando balanceamento de parênteses...")
    issues = check_parentheses_balance(content)
    if issues:
        print("❌ Problemas encontrados:")
        for issue in issues[:10]:  # Mostrar no máximo 10 problemas
            print(f"   - {issue}")
    else:
        print("✅ Parênteses balanceados!")
    
    # 2. Validar sintaxe Python
    print("\n🔍 Validando sintaxe Python...")
    if not validate_python_syntax(file_path):
        # Mostrar contexto do erro
        try:
            compile(content, file_path, 'exec')
        except SyntaxError as e:
            if e.lineno:
                find_syntax_error_context(file_path, e.lineno)
        
        # Perguntar se deseja tentar correção automática
        response = input("\n🔧 Deseja tentar correção automática? (s/n): ").strip().lower()
        if response == 's':
            if try_fix_common_issues(file_path):
                # Validar novamente
                print("\n🔍 Validando arquivo corrigido...")
                validate_python_syntax(file_path)
    
    print("\n" + "="*60)
    print("✅ Diagnóstico concluído!")
    
    # Sugestão manual
    print("\n💡 Se o erro persistir, procure por:")
    print("   1. Vírgulas extras antes de ')' ou ']' ou '}'")
    print("   2. Parênteses, colchetes ou chaves não fechados")
    print("   3. Indentação incorreta")
    print("   4. Aspas não fechadas em strings")
    print("\n📝 Dica: Use um editor com destaque de sintaxe (VS Code, PyCharm)")

if __name__ == "__main__":
    main()