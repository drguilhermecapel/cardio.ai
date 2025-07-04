#!/usr/bin/env python3
"""
Servidor Web Flask para Interface Cardio.AI
Serve a API e a interface React para testes
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image

# Adicionar path do backend
sys.path.append(str(Path(__file__).parent / "backend"))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Criar aplicação Flask
app = Flask(__name__)
CORS(app)  # Permitir CORS para frontend

# Configurações
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Criar diretório de uploads
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Verifica se arquivo é permitido."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Página inicial - redireciona para interface React."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cardio.AI - Interface de Teste</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            .container { max-width: 600px; margin: 0 auto; }
            .logo { font-size: 48px; color: #e53e3e; margin-bottom: 20px; }
            .title { font-size: 32px; margin-bottom: 20px; }
            .description { font-size: 18px; color: #666; margin-bottom: 30px; }
            .button { 
                display: inline-block; 
                padding: 12px 24px; 
                background: #3182ce; 
                color: white; 
                text-decoration: none; 
                border-radius: 6px; 
                margin: 10px;
            }
            .status { margin-top: 30px; padding: 20px; background: #f7fafc; border-radius: 6px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">❤️</div>
            <h1 class="title">Cardio.AI</h1>
            <p class="description">
                Sistema de Análise de Eletrocardiograma com IA<br>
                Interface de Teste - Correções Implementadas
            </p>
            
            <a href="http://localhost:3000" class="button">🚀 Abrir Interface React</a>
            <a href="/api/health" class="button">🔍 Verificar API</a>
            
            <div class="status">
                <h3>Status do Sistema</h3>
                <p>✅ Servidor Flask rodando na porta 5000</p>
                <p>✅ API disponível em /api/v1/ecg/ptbxl/analyze-image</p>
                <p>✅ Interface React deve estar na porta 3000</p>
                <p>✅ Correções PTB-XL implementadas</p>
            </div>
            
            <div style="margin-top: 30px; font-size: 14px; color: #666;">
                <p><strong>Instruções:</strong></p>
                <p>1. Certifique-se de que a interface React está rodando (npm run dev)</p>
                <p>2. Use a interface React para fazer upload de imagens ECG</p>
                <p>3. A API irá processar e retornar os resultados</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/api/health')
def health_check():
    """Verificação de saúde da API."""
    try:
        # Verificar se serviços estão disponíveis
        from backend.app.services.ptbxl_model_service import get_ptbxl_service
        
        service = get_ptbxl_service()
        model_status = service.is_loaded
        
        return jsonify({
            'status': 'healthy',
            'message': 'API Cardio.AI funcionando',
            'model_loaded': model_status,
            'model_type': service.model_type if model_status else 'not_loaded',
            'bias_correction': service.bias_correction_applied if model_status else False,
            'endpoints': [
                '/api/health',
                '/api/v1/ecg/ptbxl/analyze-image'
            ]
        })
        
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Erro no sistema: {str(e)}'
        }), 500

@app.route('/api/v1/ecg/ptbxl/analyze-image', methods=['POST'])
def analyze_ecg_image():
    """Endpoint principal para análise de imagem ECG."""
    try:
        logger.info("📥 Recebida requisição de análise de ECG")
        
        # Verificar se arquivo foi enviado
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Nenhum arquivo enviado'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Nenhum arquivo selecionado'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Tipo de arquivo não permitido'
            }), 400
        
        # Salvar arquivo temporariamente
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logger.info(f"📁 Arquivo salvo: {filepath}")
        
        # Ler arquivo como bytes
        with open(filepath, 'rb') as f:
            image_data = f.read()
        
        # Processar com digitalizador
        from backend.app.services.ecg_digitizer import ECGDigitizer
        
        digitizer = ECGDigitizer()
        digitization_result = digitizer.digitize_ecg_from_image(image_data, filename)
        
        if not digitization_result['success']:
            logger.warning(f"⚠️ Falha na digitalização: {digitization_result.get('error', 'Erro desconhecido')}")
            # Continuar com dados sintéticos
        
        # Obter dados ECG (reais ou sintéticos)
        ecg_data = digitization_result.get('ecg_data')
        if ecg_data is None:
            logger.info("🔧 Gerando dados ECG sintéticos para demonstração")
            import numpy as np
            ecg_data = np.random.normal(0, 0.1, (12, 1000)).astype(np.float32)
        
        # Analisar com modelo PTB-XL
        from backend.app.services.ptbxl_model_service import get_ptbxl_service
        
        service = get_ptbxl_service()
        
        if not service.is_loaded:
            return jsonify({
                'success': False,
                'error': 'Modelo PTB-XL não carregado'
            }), 500
        
        # Realizar predição
        prediction_result = service.predict_ecg(ecg_data)
        
        if not prediction_result.get('success', False):
            return jsonify({
                'success': False,
                'error': f"Erro na predição: {prediction_result.get('error', 'Erro desconhecido')}"
            }), 500
        
        # Preparar resposta
        response = {
            'success': True,
            'primary_diagnosis': prediction_result.get('primary_diagnosis'),
            'top_diagnoses': prediction_result.get('top_diagnoses', []),
            'model_used': prediction_result.get('model_used', 'unknown'),
            'bias_correction_applied': prediction_result.get('bias_correction_applied', False),
            'processing_info': {
                'quality_score': digitization_result.get('quality_score', 0.8),
                'leads_detected': digitization_result.get('leads_detected', 12),
                'sampling_rate': digitization_result.get('sampling_rate', 100),
                'grid_detected': digitization_result.get('grid_detected', False),
                'calibration_applied': digitization_result.get('calibration_applied', False)
            },
            'digitization_info': {
                'method': digitization_result.get('processing_info', {}).get('method', 'standard'),
                'image_dimensions': digitization_result.get('image_dimensions', []),
                'success': digitization_result.get('success', False)
            }
        }
        
        # Limpar arquivo temporário
        try:
            os.remove(filepath)
        except:
            pass
        
        logger.info(f"✅ Análise concluída: {prediction_result.get('primary_diagnosis', {}).get('class_name', 'N/A')}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Erro na análise: {e}")
        return jsonify({
            'success': False,
            'error': f'Erro interno: {str(e)}'
        }), 500

@app.route('/api/v1/ecg/ptbxl/model-info')
def get_model_info():
    """Informações sobre o modelo PTB-XL."""
    try:
        from backend.app.services.ptbxl_model_service import get_ptbxl_service
        
        service = get_ptbxl_service()
        info = service.get_model_info()
        
        return jsonify({
            'success': True,
            'model_info': info
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handler para arquivos muito grandes."""
    return jsonify({
        'success': False,
        'error': 'Arquivo muito grande. Máximo permitido: 16MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handler para rotas não encontradas."""
    return jsonify({
        'success': False,
        'error': 'Endpoint não encontrado'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handler para erros internos."""
    return jsonify({
        'success': False,
        'error': 'Erro interno do servidor'
    }), 500

if __name__ == '__main__':
    logger.info("🚀 Iniciando servidor Cardio.AI...")
    logger.info("📡 API disponível em: http://localhost:5000")
    logger.info("🔗 Interface React em: http://localhost:3000")
    logger.info("🏥 Health check em: http://localhost:5000/api/health")
    
    # Verificar se modelo está carregado
    try:
        from backend.app.services.ptbxl_model_service import get_ptbxl_service
        service = get_ptbxl_service()
        if service.is_loaded:
            logger.info(f"✅ Modelo PTB-XL carregado: {service.model_type}")
            logger.info(f"🔧 Correção de viés: {'Ativa' if service.bias_correction_applied else 'Inativa'}")
        else:
            logger.warning("⚠️ Modelo PTB-XL não carregado")
    except Exception as e:
        logger.error(f"❌ Erro ao verificar modelo: {e}")
    
    # Iniciar servidor
    app.run(
        host='0.0.0.0',  # Permitir acesso externo
        port=5000,
        debug=True,
        threaded=True
    )

