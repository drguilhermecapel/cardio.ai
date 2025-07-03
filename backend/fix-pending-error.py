#!/usr/bin/env python3
"""
Corrige especificamente o erro pending"}" no ecg_service.py
"""

import os
from pathlib import Path
import shutil

def fix_pending_error():
    """Corrige o erro específico de sintaxe"""
    print("🔧 Corrigindo erro pending\"}\" no ecg_service.py...")
    
    ecg_file = Path("app/services/ecg_service.py")
    
    if not ecg_file.exists():
        print("❌ Arquivo não encontrado!")
        return False
        
    # Fazer backup
    backup_file = ecg_file.with_suffix('.py.backup_pending')
    
    try:
        # Backup
        shutil.copy2(ecg_file, backup_file)
        print(f"📁 Backup criado: {backup_file}")
        
        # Ler arquivo
        with open(ecg_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Procurar linha 1199
        if len(lines) >= 1199:
            line_1199 = lines[1198]  # índice 0-based
            print(f"\n📍 Linha 1199 atual:")
            print(f"   {line_1199.rstrip()}")
            
            if 'pending"}' in line_1199:
                print("\n🔍 Erro encontrado! Corrigindo...")
                
                # Analisar contexto da linha
                stripped = line_1199.strip()
                
                # Diferentes correções baseadas no contexto
                if stripped == 'pending"}':
                    # Linha sozinha - provavelmente fim de dicionário
                    lines[1198] = '        "status": "pending"}\n'
                elif stripped.endswith('pending"}'):
                    # Final de linha - adicionar aspas
                    lines[1198] = line_1199.replace('pending"}', '"pending"}"')
                elif ': pending"}' in line_1199:
                    # Valor de dicionário - corrigir formato
                    lines[1198] = line_1199.replace(': pending"}', ': "pending"}')
                else:
                    # Correção genérica
                    lines[1198] = line_1199.replace('pending"}', '"pending"}')
                
                print(f"\n✅ Linha corrigida para:")
                print(f"   {lines[1198].rstrip()}")
                
                # Salvar
                with open(ecg_file, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                    
                print("\n✅ Arquivo salvo com sucesso!")
                return True
            else:
                print("\n✅ Erro 'pending\"}' não encontrado na linha 1199")
                
                # Procurar em outras linhas
                for i, line in enumerate(lines):
                    if 'pending"}' in line and 'pending"}"' not in line:
                        print(f"\n⚠️ Encontrado na linha {i+1}: {line.strip()}")
                        
                        # Corrigir
                        if ': pending"}' in line:
                            lines[i] = line.replace(': pending"}', ': "pending"}')
                        else:
                            lines[i] = line.replace('pending"}', '"pending"}')
                            
                        # Salvar
                        with open(ecg_file, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                            
                        print(f"✅ Corrigido!")
                        return True
                        
        else:
            print(f"\n⚠️ Arquivo tem apenas {len(lines)} linhas")
            
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return False
        
    return True


def verify_fix():
    """Verifica se a correção funcionou"""
    print("\n🔍 Verificando correção...")
    
    try:
        # Tentar importar
        from app.services.ecg_service import ECGAnalysisService
        print("✅ ECGAnalysisService importado com sucesso!")
        return True
    except SyntaxError as e:
        print(f"❌ Ainda há erro de sintaxe:")
        print(f"   Arquivo: {e.filename}")
        print(f"   Linha: {e.lineno}")
        print(f"   Erro: {e.msg}")
        return False
    except Exception as e:
        print(f"⚠️ Outro erro: {e}")
        return False


def main():
    print("=" * 60)
    print("🔧 CARDIOAI PRO - CORREÇÃO DO ERRO PENDING")
    print("=" * 60)
    
    # Aplicar correção
    if fix_pending_error():
        # Verificar
        if verify_fix():
            print("\n🎉 SUCESSO! O erro foi corrigido!")
            print("\n📝 Próximo passo:")
            print("   python -m pytest --cov=app --cov-report=html")
        else:
            print("\n⚠️ A correção foi aplicada mas ainda há problemas.")
            print("Execute: python cardioai_defibrillator.py")
    else:
        print("\n❌ Não foi possível aplicar a correção.")
        

if __name__ == "__main__":
    main()
