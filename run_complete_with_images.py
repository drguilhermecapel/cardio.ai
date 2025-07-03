#!/usr/bin/env python3
"""
Script para executar o CardioAI Pro - Sistema Completo com Análise de Imagens
Interface Web + Backend + APIs + Digitalização de ECG
"""

import sys
import os
import uvicorn
from pathlib import Path

# Adicionar o diretório backend ao path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def main():
    """Função principal para executar o sistema completo com imagens."""
    print("🚀 Iniciando CardioAI Pro - Sistema Completo com Análise de Imagens")
    print("=" * 70)
    print("📍 Diretório:", os.getcwd())
    print("🔧 Modo: Completo (Interface + Backend + APIs + Digitalização)")
    print("🌐 Frontend: Integrado e Aprimorado")
    print("🔬 Backend: FastAPI com Análise de Imagens")
    print("🧠 IA: Modelos Aprimorados + Suporte .h5")
    print("🖼️ Imagens: Digitalização de ECG Automática")
    print("🏥 FHIR: R4 Compatível")
    
    try:
        # Configurações do servidor
        config = {
            "app": "app.main_complete:app",
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
        
        print("\n🖼️ Funcionalidades de Imagem:")
        print("  • ✅ Upload de Imagens ECG (JPG, PNG, PDF)")
        print("  • ✅ Digitalização Automática de Traçados")
        print("  • ✅ Extração de Dados Numéricos")
        print("  • ✅ Detecção de Grade e Calibração")
        print("  • ✅ Análise de Qualidade da Digitalização")
        print("  • ✅ Análise em Lote de Múltiplas Imagens")
        
        print("\n🔬 Funcionalidades de Análise:")
        print("  • ✅ Modelos de IA Aprimorados")
        print("  • ✅ Suporte a Modelos .h5 Pré-treinados")
        print("  • ✅ Diagnóstico Automático Preciso")
        print("  • ✅ Recomendações Clínicas")
        print("  • ✅ Sistema de Confiança e Qualidade")
        print("  • ✅ Compatibilidade FHIR R4")
        
        print("\n🎯 Endpoints Principais:")
        print("  • POST /api/v1/ecg/image/analyze - Análise completa de imagem")
        print("  • POST /api/v1/ecg/image/digitize-only - Apenas digitalização")
        print("  • POST /api/v1/ecg/image/batch-analyze - Análise em lote")
        print("  • GET /api/v1/ecg/image/supported-formats - Formatos suportados")
        print("  • POST /api/v1/ecg/analyze - Análise de dados numéricos")
        print("  • GET /api/v1/ecg/models - Modelos disponíveis")
        
        print("\n📋 Formatos Suportados:")
        print("  • Imagens: JPG, JPEG, PNG, BMP, TIFF, PDF")
        print("  • Dados: CSV, TXT, NPY")
        print("  • Modelos: .h5 (TensorFlow/Keras), .pkl, .joblib")
        
        print("\n🎯 Como Usar:")
        print("  1. Acesse a interface principal no navegador")
        print("  2. Use 'Análise de Imagem ECG' para upload de imagens")
        print("  3. Configure parâmetros de qualidade e modelo")
        print("  4. Visualize resultados com diagnóstico e recomendações")
        print("  5. Use análise em lote para múltiplas imagens")
        print("  6. Explore APIs para integração com sistemas externos")
        
        print("\n" + "=" * 70)
        print("🚀 Iniciando servidor...")
        
        # Executar servidor
        uvicorn.run(**config)
        
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido pelo usuário")
        print("👋 CardioAI Pro encerrado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao executar servidor: {str(e)}")
        print("💡 Verifique se todas as dependências estão instaladas:")
        print("   pip3 install opencv-python scikit-image scipy matplotlib pillow")
        sys.exit(1)

if __name__ == "__main__":
    main()

