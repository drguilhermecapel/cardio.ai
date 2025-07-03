#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para diagnosticar o problema de indentação no ECGAnalysisService
"""

from pathlib import Path

def diagnose_indentation_issue():
    """Mostra exatamente onde está o problema de indentação."""
    
    service_file = Path("app/services/ecg_service.py")
    
    if not service_file.exists():
        print("[ERRO] Arquivo ecg_service.py não encontrado!")
        return
    
    print(f"[INFO] Analisando {service_file}...")
    print("="*60)
    
    try:
        with open(service_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Mostrar linhas ao redor da linha 32
        print("\n[CONTEXTO] Linhas 25-40 do arquivo:")
        print("-"*60)
        
        for i in range(max(0, 24), min(len(lines), 40)):
            line_num = i + 1
            line = lines[i].rstrip('\n')
            
            # Mostrar espaços como pontos para visualizar melhor
            visual_line = line.replace(' ', '·')
            
            # Marcar linha problemática
            marker = " <-- ERRO AQUI" if line_num == 32 else ""
            
            print(f"{line_num:3d}: {visual_line}{marker}")
        
        print("-"*60)
        
        # Análise detalhada da linha 32
        if len(lines) > 31:
            line_32 = lines[31]
            print(f"\n[ANÁLISE] Linha 32:")
            print(f"  Conteúdo: {repr(line_32)}")
            print(f"  Espaços no início: {len(line_32) - len(line_32.lstrip())}")
            print(f"  Primeiro caractere não-espaço: {line_32.lstrip()[0] if line_32.strip() else 'VAZIA'}")
            
            # Verificar indentação esperada
            print("\n[INDENTAÇÃO ESPERADA]:")
            print("  - Métodos dentro de classe: 4 espaços")
            print("  - Conteúdo dentro de método: 8 espaços")
            print("  - Continuação de linha: alinhado com parêntese de abertura")
        
        # Procurar padrões problemáticos
        print("\n[PADRÕES ENCONTRADOS]:")
        
        in_class = False
        for i, line in enumerate(lines):
            if 'class ECGAnalysisService' in line:
                in_class = True
                print(f"  Classe encontrada na linha {i+1}")
            
            if in_class and line.strip().startswith('def __init__'):
                indent = len(line) - len(line.lstrip())
                print(f"  __init__ encontrado na linha {i+1} com {indent} espaços de indentação")
                
                # Verificar linhas seguintes
                for j in range(i+1, min(i+5, len(lines))):
                    next_line = lines[j]
                    if next_line.strip():
                        next_indent = len(next_line) - len(next_line.lstrip())
                        print(f"    Linha {j+1}: {next_indent} espaços - {next_line.strip()[:40]}...")
        
    except Exception as e:
        print(f"[ERRO] Falha ao analisar arquivo: {e}")


def suggest_fix():
    """Sugere correção específica."""
    print("\n[SUGESTÃO DE CORREÇÃO]:")
    print("="*60)
    print("""
Se o erro está na definição do __init__, a estrutura correta deve ser:

class ECGAnalysisService:
····def __init__(
········self,
········db: AsyncSession = None,
········ml_service: MLModelService = None,
········# ... outros parâmetros
····) -> None:
········# Código do método

Onde:
- '·' representa um espaço
- Métodos devem ter 4 espaços de indentação
- Parâmetros devem ter 8 espaços (ou alinhados com o parêntese)
- Corpo do método deve ter 8 espaços
""")


def main():
    """Função principal."""
    print("DIAGNÓSTICO DE ERRO DE INDENTAÇÃO")
    print("="*60)
    
    diagnose_indentation_issue()
    suggest_fix()
    
    print("\n[PRÓXIMOS PASSOS]:")
    print("1. Execute: python fix-indentation-error.py")
    print("2. Se não funcionar, edite manualmente app/services/ecg_service.py")
    print("3. Corrija a indentação na linha 32 e arredores")
    print("4. Execute os testes novamente")


if __name__ == "__main__":
    main()
