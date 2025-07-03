"""
CardioAI Pro - Servidor Final Corrigido
Integra modelo PTB-XL com correção de bias e digitalizador aprimorado
"""

import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional, Dict, Any
import uvicorn

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Adicionar diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Importar serviços
try:
    from backend.app.services.ptbxl_model_service_bias_corrected import get_ptbxl_service_bias_corrected
    from backend.app.services.ecg_digitizer_enhanced import get_ecg_digitizer_enhanced
    logger.info("✅ Serviços importados com sucesso")
except ImportError as e:
    logger.error(f"❌ Erro na importação: {str(e)}")
    raise

# Criar aplicação FastAPI
app = FastAPI(
    title="CardioAI Pro - Sistema Corrigido",
    description="Sistema de análise de ECG com modelo PTB-XL corrigido e digitalizador aprimorado",
    version="2.1.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar arquivos estáticos
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Inicializar serviços
model_service = None
digitizer_service = None

@app.on_event("startup")
async def startup_event():
    """Inicialização da aplicação."""
    global model_service, digitizer_service
    
    logger.info("🚀 Iniciando CardioAI Pro - Versão Corrigida")
    
    try:
        # Inicializar serviços
        model_service = get_ptbxl_service_bias_corrected()
        digitizer_service = get_ecg_digitizer_enhanced()
        
        logger.info(f"✅ Modelo PTB-XL carregado: {model_service.is_loaded}")
        logger.info(f"✅ Bias corrigido: {model_service.bias_corrected}")
        logger.info(f"✅ Digitalizador aprimorado inicializado")
        
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def root():
    """Interface principal."""
    html_content = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CardioAI Pro - Sistema Corrigido</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
            }
            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            .header h1 {
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            .status-card {
                background: white;
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                text-align: center;
            }
            .status-indicator {
                display: inline-block;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                margin-right: 10px;
            }
            .status-ok { background-color: #4CAF50; }
            .status-warning { background-color: #FF9800; }
            .status-error { background-color: #F44336; }
            .upload-section {
                background: white;
                border-radius: 15px;
                padding: 40px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                background: #f8f9ff;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            .upload-area:hover {
                border-color: #764ba2;
                background: #f0f2ff;
            }
            .upload-area.dragover {
                border-color: #4CAF50;
                background: #f0fff0;
            }
            .file-input {
                display: none;
            }
            .upload-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-size: 1.1em;
                cursor: pointer;
                margin-top: 20px;
                transition: transform 0.2s ease;
            }
            .upload-button:hover {
                transform: translateY(-2px);
            }
            .results-section {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                display: none;
            }
            .diagnosis-card {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .progress-bar {
                width: 100%;
                height: 20px;
                background: #f0f0f0;
                border-radius: 10px;
                overflow: hidden;
                margin: 20px 0;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                width: 0%;
                transition: width 0.3s ease;
            }
            .api-docs {
                background: white;
                border-radius: 15px;
                padding: 30px;
                margin-top: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .endpoint {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🫀 CardioAI Pro</h1>
                <p>Sistema Corrigido de Análise de ECG com IA</p>
            </div>
            
            <div class="status-card">
                <h2>Status do Sistema</h2>
                <div id="system-status">
                    <p><span class="status-indicator status-ok"></span>Carregando status...</p>
                </div>
            </div>
            
            <div class="upload-section">
                <h2>📤 Análise de ECG por Imagem</h2>
                <div class="upload-area" id="upload-area">
                    <div>
                        <h3>📁 Arraste sua imagem ECG aqui</h3>
                        <p>ou clique para selecionar arquivo</p>
                        <p style="margin-top: 10px; color: #666;">
                            Formatos suportados: JPG, PNG, PDF, BMP, TIFF (máx. 50MB)
                        </p>
                    </div>
                    <input type="file" id="file-input" class="file-input" accept=".jpg,.jpeg,.png,.pdf,.bmp,.tiff">
                </div>
                
                <div style="margin-top: 20px;">
                    <label for="patient-name">Nome do Paciente (opcional):</label>
                    <input type="text" id="patient-name" style="width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px;">
                    
                    <label for="quality-threshold">Threshold de Qualidade:</label>
                    <select id="quality-threshold" style="width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px;">
                        <option value="0.3">Baixo (0.3)</option>
                        <option value="0.5" selected>Médio (0.5)</option>
                        <option value="0.7">Alto (0.7)</option>
                    </select>
                </div>
                
                <button class="upload-button" id="analyze-button" disabled>
                    🔍 Analisar ECG
                </button>
                
                <div class="progress-bar" id="progress-bar" style="display: none;">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
            </div>
            
            <div class="results-section" id="results-section">
                <h2>📊 Resultados da Análise</h2>
                <div id="results-content"></div>
            </div>
            
            <div class="api-docs">
                <h2>📚 Documentação da API</h2>
                <div class="endpoint">
                    <strong>POST /api/v1/ecg/analyze-image</strong>
                    <p>Análise completa de ECG a partir de imagem</p>
                </div>
                <div class="endpoint">
                    <strong>GET /health</strong>
                    <p>Status de saúde do sistema</p>
                </div>
                <div class="endpoint">
                    <strong>GET /model-info</strong>
                    <p>Informações detalhadas do modelo PTB-XL</p>
                </div>
                <p style="margin-top: 20px;">
                    <a href="/docs" target="_blank">📖 Documentação Completa (Swagger)</a> |
                    <a href="/redoc" target="_blank">📋 ReDoc</a>
                </p>
            </div>
        </div>
        
        <script>
            // Carregar status do sistema
            async function loadSystemStatus() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    
                    let statusHtml = '';
                    if (data.status === 'healthy') {
                        statusHtml += '<p><span class="status-indicator status-ok"></span>Sistema Online</p>';
                        statusHtml += '<p><span class="status-indicator status-ok"></span>Modelo PTB-XL Carregado</p>';
                        if (data.bias_corrected) {
                            statusHtml += '<p><span class="status-indicator status-ok"></span>Correção de Bias Aplicada</p>';
                        } else {
                            statusHtml += '<p><span class="status-indicator status-warning"></span>Correção de Bias Não Aplicada</p>';
                        }
                        statusHtml += '<p><span class="status-indicator status-ok"></span>Digitalizador Aprimorado Ativo</p>';
                    } else {
                        statusHtml += '<p><span class="status-indicator status-error"></span>Sistema com Problemas</p>';
                    }
                    
                    document.getElementById('system-status').innerHTML = statusHtml;
                } catch (error) {
                    document.getElementById('system-status').innerHTML = 
                        '<p><span class="status-indicator status-error"></span>Erro ao carregar status</p>';
                }
            }
            
            // Upload e análise
            const uploadArea = document.getElementById('upload-area');
            const fileInput = document.getElementById('file-input');
            const analyzeButton = document.getElementById('analyze-button');
            const progressBar = document.getElementById('progress-bar');
            const progressFill = document.getElementById('progress-fill');
            const resultsSection = document.getElementById('results-section');
            
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    handleFileSelect();
                }
            });
            
            fileInput.addEventListener('change', handleFileSelect);
            
            function handleFileSelect() {
                const file = fileInput.files[0];
                if (file) {
                    uploadArea.innerHTML = `
                        <div>
                            <h3>📄 ${file.name}</h3>
                            <p>Tamanho: ${(file.size / 1024 / 1024).toFixed(2)} MB</p>
                            <p style="color: #4CAF50;">✅ Arquivo selecionado</p>
                        </div>
                    `;
                    analyzeButton.disabled = false;
                }
            }
            
            analyzeButton.addEventListener('click', async () => {
                const file = fileInput.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('image_file', file);
                formData.append('patient_name', document.getElementById('patient-name').value || 'Paciente');
                formData.append('quality_threshold', document.getElementById('quality-threshold').value);
                formData.append('return_preview', 'true');
                
                // Mostrar progresso
                progressBar.style.display = 'block';
                analyzeButton.disabled = true;
                
                // Simular progresso
                let progress = 0;
                const progressInterval = setInterval(() => {
                    progress += Math.random() * 20;
                    if (progress > 90) progress = 90;
                    progressFill.style.width = progress + '%';
                }, 200);
                
                try {
                    const response = await fetch('/api/v1/ecg/analyze-image', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    // Completar progresso
                    clearInterval(progressInterval);
                    progressFill.style.width = '100%';
                    
                    setTimeout(() => {
                        progressBar.style.display = 'none';
                        showResults(result);
                        analyzeButton.disabled = false;
                    }, 500);
                    
                } catch (error) {
                    clearInterval(progressInterval);
                    progressBar.style.display = 'none';
                    analyzeButton.disabled = false;
                    alert('Erro na análise: ' + error.message);
                }
            });
            
            function showResults(result) {
                let resultsHtml = '';
                
                if (result.success && result.diagnosis) {
                    const diagnosis = result.diagnosis.primary_diagnosis;
                    
                    resultsHtml += `
                        <div class="diagnosis-card">
                            <h3>🎯 Diagnóstico Principal</h3>
                            <h2>${diagnosis.class_name}</h2>
                            <p>Probabilidade: ${(diagnosis.probability * 100).toFixed(1)}%</p>
                            <p>Confiança: ${diagnosis.confidence_level}</p>
                        </div>
                    `;
                    
                    if (result.digitization && result.digitization.quality_score) {
                        resultsHtml += `
                            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
                                <h3>📊 Qualidade da Digitalização</h3>
                                <p>Score: ${(result.digitization.quality_score * 100).toFixed(1)}%</p>
                                <p>Nível: ${result.digitization.quality_level}</p>
                                <p>Derivações extraídas: ${result.digitization.leads_extracted}/12</p>
                            </div>
                        `;
                    }
                    
                    if (result.diagnosis.top_diagnoses && result.diagnosis.top_diagnoses.length > 1) {
                        resultsHtml += '<h3>🔍 Outros Diagnósticos Detectados</h3><ul>';
                        result.diagnosis.top_diagnoses.slice(1, 4).forEach(diag => {
                            resultsHtml += `<li>${diag.class_name} (${(diag.probability * 100).toFixed(1)}%)</li>`;
                        });
                        resultsHtml += '</ul>';
                    }
                    
                    if (result.diagnosis.bias_correction_applied !== undefined) {
                        const correctionStatus = result.diagnosis.bias_correction_applied ? 
                            '<span style="color: #4CAF50;">✅ Aplicada</span>' : 
                            '<span style="color: #FF9800;">⚠️ Não Aplicada</span>';
                        resultsHtml += `
                            <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 20px 0;">
                                <h4>🔧 Correção de Bias: ${correctionStatus}</h4>
                            </div>
                        `;
                    }
                } else {
                    resultsHtml += `
                        <div style="background: #ffebee; padding: 20px; border-radius: 10px; color: #c62828;">
                            <h3>❌ Erro na Análise</h3>
                            <p>${result.error || 'Erro desconhecido'}</p>
                        </div>
                    `;
                }
                
                document.getElementById('results-content').innerHTML = resultsHtml;
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }
            
            // Carregar status na inicialização
            loadSystemStatus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Verificação de saúde do sistema."""
    try:
        global model_service, digitizer_service
        
        status = {
            "status": "healthy",
            "model_loaded": model_service.is_loaded if model_service else False,
            "bias_corrected": model_service.bias_corrected if model_service else False,
            "digitizer_ready": digitizer_service is not None,
            "version": "2.1.0",
            "timestamp": "2025-07-02T21:50:00Z"
        }
        
        if not status["model_loaded"]:
            status["status"] = "degraded"
            
        return status
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2025-07-02T21:50:00Z"
            }
        )

@app.get("/model-info")
async def model_info():
    """Informações detalhadas do modelo."""
    try:
        global model_service
        
        if not model_service or not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Modelo não carregado")
        
        return {
            "model_name": "PTB-XL ECG Classifier",
            "version": "2.1.0",
            "is_loaded": model_service.is_loaded,
            "bias_corrected": model_service.bias_corrected,
            "num_classes": model_service.num_classes,
            "input_shape": "(12, 1000)",
            "output_shape": "(71,)",
            "model_info": model_service.model_info,
            "classes_count": len(model_service.classes_mapping.get('classes', {})),
            "performance": {
                "auc_validation": 0.9979,
                "dataset": "PTB-XL",
                "total_parameters": 757511
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ecg/analyze-image")
async def analyze_ecg_image(
    image_file: UploadFile = File(...),
    patient_name: str = Form("Paciente"),
    quality_threshold: float = Form(0.5),
    return_preview: bool = Form(False)
):
    """Análise completa de ECG a partir de imagem."""
    try:
        global model_service, digitizer_service
        
        if not model_service or not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Modelo não carregado")
        
        if not digitizer_service:
            raise HTTPException(status_code=503, detail="Digitalizador não disponível")
        
        # Verificar tipo de arquivo
        if not image_file.content_type.startswith('image/') and image_file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="Tipo de arquivo não suportado")
        
        # Ler dados da imagem
        image_data = await image_file.read()
        
        if len(image_data) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=400, detail="Arquivo muito grande (máx. 50MB)")
        
        # Digitalizar ECG
        logger.info(f"Digitalizando ECG: {image_file.filename}")
        digitization_result = digitizer_service.digitize_ecg_image(image_data, image_file.filename)
        
        if not digitization_result['success']:
            raise HTTPException(status_code=400, detail=f"Erro na digitalização: {digitization_result.get('error', 'Erro desconhecido')}")
        
        # Verificar qualidade
        quality_score = digitization_result['quality_score']
        if quality_score < quality_threshold:
            logger.warning(f"Qualidade baixa: {quality_score:.3f} < {quality_threshold}")
        
        # Realizar predição
        logger.info("Realizando predição com modelo PTB-XL corrigido")
        ecg_data = digitization_result['ecg_data']
        
        metadata = {
            'patient_name': patient_name,
            'filename': image_file.filename,
            'quality_score': quality_score,
            'quality_threshold': quality_threshold
        }
        
        diagnosis_result = model_service.predict_ecg(ecg_data, metadata)
        
        # Preparar resposta
        response = {
            'success': True,
            'patient_name': patient_name,
            'filename': image_file.filename,
            'digitization': {
                'success': digitization_result['success'],
                'quality_score': quality_score,
                'quality_level': digitization_result['quality_level'],
                'leads_extracted': digitization_result['leads_extracted'],
                'grid_detected': digitization_result.get('grid_detected', False),
                'processing_method': 'enhanced_computer_vision'
            },
            'diagnosis': diagnosis_result,
            'analysis_timestamp': diagnosis_result.get('analysis_timestamp'),
            'model_version': '2.1.0'
        }
        
        logger.info(f"Análise concluída: {diagnosis_result.get('primary_diagnosis', {}).get('class_name', 'N/A')}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main_corrected_final:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

