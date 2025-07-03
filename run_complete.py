#!/usr/bin/env python3
"""
Script para executar o CardioAI Pro - Sistema Completo
Interface Web + Backend + APIs integrados
"""

import sys
import os
import uvicorn
from pathlib import Path

# Adicionar o diretório backend ao path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def main():
    """Função principal para executar o sistema completo."""
    print("🚀 Iniciando CardioAI Pro - Sistema Completo")
    print("=" * 60)
    print("📍 Diretório:", os.getcwd())
    print("🔧 Modo: Completo (Interface Web + Backend + APIs)")
    print("🌐 Frontend: Integrado")
    print("🔬 Backend: FastAPI")
    print("🧠 IA: Modelos simplificados")
    print("🏥 FHIR: R4 Compatível")
    
    try:
        # Configurações do servidor
        config = {
            "app": "app.main_full:app",
            "host": "0.0.0.0",  # Permitir acesso externo
            "port": 8000,
            "reload": False,
            "log_level": "info",
            "access_log": True
        }
        
        print("\n🌐 URLs de Acesso:")
        print(f"  • Interface Principal: http://{config['host']}:{config['port']}/")
        print(f"  • Documentação API: http://{config['host']}:{config['port']}/docs")
        print(f"  • ReDoc: http://{config['host']}:{config['port']}/redoc")
        print(f"  • Health Check: http://{config['host']}:{config['port']}/health")
        print(f"  • Informações: http://{config['host']}:{config['port']}/info")
        
        print("\n🔬 Funcionalidades Disponíveis:")
        print("  • ✅ Interface Web Interativa")
        print("  • ✅ Análise de ECG em Tempo Real")
        print("  • ✅ Upload de Arquivos (CSV, TXT, NPY)")
        print("  • ✅ Modelos de IA Integrados")
        print("  • ✅ Compatibilidade FHIR R4")
        print("  • ✅ APIs RESTful Completas")
        print("  • ✅ Documentação Interativa")
        print("  • ✅ Dashboard de Monitoramento")
        
        print("\n🎯 Como Usar:")
        print("  1. Acesse a interface principal no navegador")
        print("  2. Use os cards para navegar pelas funcionalidades")
        print("  3. Teste análise de ECG com dados de exemplo")
        print("  4. Explore a documentação da API")
        print("  5. Integre com sistemas externos via APIs")
        
        print("\n" + "=" * 60)
        print("🚀 Iniciando servidor...")
        
        # Executar servidor
        uvicorn.run(**config)
        
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido pelo usuário")
        print("👋 CardioAI Pro encerrado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao executar servidor: {str(e)}")
        print("💡 Verifique se todas as dependências estão instaladas")
        sys.exit(1)

if __name__ == "__main__":
    main()

