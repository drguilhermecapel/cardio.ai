"""
Aplicação CardioAI Pro - Versão Completa com Análise de Imagens
Sistema completo de análise de ECG com digitalização de imagens
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import logging
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador de ciclo de vida da aplicação."""
    # Startup
    logger.info("Iniciando CardioAI Pro (Versão Completa com Imagens)...")
    
    try:
        # Inicializar serviços
        from app.services.model_service_enhanced import get_enhanced_model_service
        model_service = get_enhanced_model_service()
        logger.info("Serviço de modelos aprimorado inicializado")
        
        # Criar diretórios necessários
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
        
        # Criar frontend estático se não existir
        create_enhanced_frontend()
        
        logger.info("CardioAI Pro (Completo com Imagens) iniciado com sucesso")
        
    except Exception as e:
        logger.error(f"Erro na inicialização: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Encerrando CardioAI Pro (Completo com Imagens)...")


def create_enhanced_frontend():
    """Cria frontend estático aprimorado com suporte a imagens."""
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    # Criar index.html aprimorado
    index_html = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CardioAI Pro - Sistema Avançado de Análise de ECG</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-hover { transition: all 0.3s ease; }
        .card-hover:hover { transform: translateY(-5px); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .upload-area { border: 2px dashed #cbd5e0; transition: all 0.3s ease; }
        .upload-area:hover { border-color: #4299e1; background-color: #f7fafc; }
        .upload-area.dragover { border-color: #3182ce; background-color: #ebf8ff; }
    </style>
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <header class="gradient-bg text-white shadow-lg">
        <div class="container mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-4">
                    <i class="fas fa-heartbeat text-3xl pulse"></i>
                    <div>
                        <h1 class="text-2xl font-bold">CardioAI Pro</h1>
                        <p class="text-sm opacity-90">Sistema Avançado de Análise de ECG com IA</p>
                    </div>
                </div>
                <div class="flex items-center space-x-4">
                    <span class="bg-green-500 px-3 py-1 rounded-full text-sm">v2.0.0</span>
                    <span class="bg-blue-500 px-3 py-1 rounded-full text-sm">
                        <i class="fas fa-image mr-1"></i>Imagens
                    </span>
                    <button onclick="checkHealth()" class="bg-white bg-opacity-20 px-4 py-2 rounded-lg hover:bg-opacity-30 transition">
                        <i class="fas fa-stethoscope mr-2"></i>Status
                    </button>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container mx-auto px-6 py-8">
        <!-- Status Card -->
        <div id="statusCard" class="bg-white rounded-xl shadow-lg p-6 mb-8 card-hover">
            <div class="flex items-center justify-between">
                <div>
                    <h2 class="text-xl font-semibold text-gray-800">Status do Sistema</h2>
                    <p class="text-gray-600">Verificando conectividade...</p>
                </div>
                <div id="statusIndicator" class="w-4 h-4 bg-yellow-500 rounded-full pulse"></div>
            </div>
        </div>

        <!-- Features Grid -->
        <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            <!-- ECG Image Analysis -->
            <div class="bg-white rounded-xl shadow-lg p-6 card-hover border-l-4 border-blue-500">
                <div class="flex items-center mb-4">
                    <div class="bg-blue-100 p-3 rounded-lg">
                        <i class="fas fa-image text-blue-600 text-xl"></i>
                    </div>
                    <h3 class="text-lg font-semibold ml-4">Análise de Imagem ECG</h3>
                </div>
                <p class="text-gray-600 mb-4">Upload de imagens ECG (JPG, PNG, PDF) para análise automática</p>
                <button onclick="showImageAnalysisForm()" class="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition">
                    <i class="fas fa-upload mr-2"></i>Analisar Imagem
                </button>
            </div>

            <!-- ECG Data Analysis -->
            <div class="bg-white rounded-xl shadow-lg p-6 card-hover border-l-4 border-green-500">
                <div class="flex items-center mb-4">
                    <div class="bg-green-100 p-3 rounded-lg">
                        <i class="fas fa-chart-line text-green-600 text-xl"></i>
                    </div>
                    <h3 class="text-lg font-semibold ml-4">Análise de Dados ECG</h3>
                </div>
                <p class="text-gray-600 mb-4">Análise de dados numéricos de ECG</p>
                <button onclick="showDataAnalysisForm()" class="w-full bg-green-600 text-white py-2 rounded-lg hover:bg-green-700 transition">
                    <i class="fas fa-calculator mr-2"></i>Analisar Dados
                </button>
            </div>

            <!-- File Upload -->
            <div class="bg-white rounded-xl shadow-lg p-6 card-hover border-l-4 border-purple-500">
                <div class="flex items-center mb-4">
                    <div class="bg-purple-100 p-3 rounded-lg">
                        <i class="fas fa-file-upload text-purple-600 text-xl"></i>
                    </div>
                    <h3 class="text-lg font-semibold ml-4">Upload de Arquivo</h3>
                </div>
                <p class="text-gray-600 mb-4">Envie arquivos ECG (CSV, TXT, NPY)</p>
                <button onclick="showUploadForm()" class="w-full bg-purple-600 text-white py-2 rounded-lg hover:bg-purple-700 transition">
                    <i class="fas fa-upload mr-2"></i>Enviar Arquivo
                </button>
            </div>

            <!-- Batch Analysis -->
            <div class="bg-white rounded-xl shadow-lg p-6 card-hover border-l-4 border-orange-500">
                <div class="flex items-center mb-4">
                    <div class="bg-orange-100 p-3 rounded-lg">
                        <i class="fas fa-layer-group text-orange-600 text-xl"></i>
                    </div>
                    <h3 class="text-lg font-semibold ml-4">Análise em Lote</h3>
                </div>
                <p class="text-gray-600 mb-4">Processe múltiplas imagens ECG simultaneamente</p>
                <button onclick="showBatchAnalysisForm()" class="w-full bg-orange-600 text-white py-2 rounded-lg hover:bg-orange-700 transition">
                    <i class="fas fa-tasks mr-2"></i>Análise em Lote
                </button>
            </div>

            <!-- Models -->
            <div class="bg-white rounded-xl shadow-lg p-6 card-hover border-l-4 border-indigo-500">
                <div class="flex items-center mb-4">
                    <div class="bg-indigo-100 p-3 rounded-lg">
                        <i class="fas fa-brain text-indigo-600 text-xl"></i>
                    </div>
                    <h3 class="text-lg font-semibold ml-4">Modelos IA</h3>
                </div>
                <p class="text-gray-600 mb-4">Visualize modelos de IA disponíveis</p>
                <button onclick="showModels()" class="w-full bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700 transition">
                    <i class="fas fa-cogs mr-2"></i>Ver Modelos
                </button>
            </div>

            <!-- Documentation -->
            <div class="bg-white rounded-xl shadow-lg p-6 card-hover border-l-4 border-yellow-500">
                <div class="flex items-center mb-4">
                    <div class="bg-yellow-100 p-3 rounded-lg">
                        <i class="fas fa-book text-yellow-600 text-xl"></i>
                    </div>
                    <h3 class="text-lg font-semibold ml-4">Documentação</h3>
                </div>
                <p class="text-gray-600 mb-4">API docs e guias de uso</p>
                <div class="space-y-2">
                    <a href="/docs" target="_blank" class="block w-full bg-yellow-600 text-white py-2 rounded-lg hover:bg-yellow-700 transition text-center">
                        <i class="fas fa-code mr-2"></i>Swagger UI
                    </a>
                    <a href="/redoc" target="_blank" class="block w-full bg-yellow-500 text-white py-2 rounded-lg hover:bg-yellow-600 transition text-center">
                        <i class="fas fa-book-open mr-2"></i>ReDoc
                    </a>
                </div>
            </div>
        </div>

        <!-- Results Area -->
        <div id="resultsArea" class="hidden bg-white rounded-xl shadow-lg p-6">
            <h3 class="text-xl font-semibold mb-4">Resultados da Análise</h3>
            <div id="resultsContent"></div>
        </div>
    </main>

    <!-- Modal -->
    <div id="modal" class="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center z-50">
        <div class="bg-white rounded-xl p-6 m-4 max-w-4xl w-full max-h-96 overflow-y-auto">
            <div class="flex justify-between items-center mb-4">
                <h3 id="modalTitle" class="text-xl font-semibold"></h3>
                <button onclick="closeModal()" class="text-gray-500 hover:text-gray-700">
                    <i class="fas fa-times text-xl"></i>
                </button>
            </div>
            <div id="modalContent"></div>
        </div>
    </div>

    <script>
        // API Base URL
        const API_BASE = window.location.origin;

        // Utility functions
        function showModal(title, content) {
            document.getElementById('modalTitle').textContent = title;
            document.getElementById('modalContent').innerHTML = content;
            document.getElementById('modal').classList.remove('hidden');
            document.getElementById('modal').classList.add('flex');
        }

        function closeModal() {
            document.getElementById('modal').classList.add('hidden');
            document.getElementById('modal').classList.remove('flex');
        }

        function showResults(content) {
            document.getElementById('resultsContent').innerHTML = content;
            document.getElementById('resultsArea').classList.remove('hidden');
            document.getElementById('resultsArea').scrollIntoView({ behavior: 'smooth' });
        }

        // Health check
        async function checkHealth() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                
                const statusCard = document.getElementById('statusCard');
                
                if (data.status === 'healthy') {
                    statusCard.innerHTML = `
                        <div class="flex items-center justify-between">
                            <div>
                                <h2 class="text-xl font-semibold text-gray-800">Sistema Online</h2>
                                <p class="text-gray-600">Todos os serviços funcionando normalmente</p>
                                <p class="text-sm text-gray-500">Modelos: ${data.services.models_loaded} | Modo: ${data.mode}</p>
                            </div>
                            <div class="w-4 h-4 bg-green-500 rounded-full"></div>
                        </div>
                    `;
                } else {
                    throw new Error('Sistema não saudável');
                }
            } catch (error) {
                const statusCard = document.getElementById('statusCard');
                statusCard.innerHTML = `
                    <div class="flex items-center justify-between">
                        <div>
                            <h2 class="text-xl font-semibold text-gray-800">Sistema Offline</h2>
                            <p class="text-gray-600">Erro na conectividade</p>
                        </div>
                        <div class="w-4 h-4 bg-red-500 rounded-full"></div>
                    </div>
                `;
            }
        }

        // Show image analysis form
        function showImageAnalysisForm() {
            const content = `
                <form onsubmit="analyzeECGImage(event)" class="space-y-4" enctype="multipart/form-data">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">ID do Paciente</label>
                        <input type="text" id="imagePatientId" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Imagem ECG</label>
                        <div class="upload-area p-6 rounded-lg text-center">
                            <input type="file" id="ecgImageFile" accept=".jpg,.jpeg,.png,.bmp,.tiff,.pdf" required class="hidden">
                            <div onclick="document.getElementById('ecgImageFile').click()" class="cursor-pointer">
                                <i class="fas fa-cloud-upload-alt text-4xl text-gray-400 mb-2"></i>
                                <p class="text-gray-600">Clique para selecionar imagem ECG</p>
                                <p class="text-sm text-gray-500">JPG, PNG, PDF (máx. 50MB)</p>
                            </div>
                        </div>
                        <div id="imagePreview" class="mt-2 hidden">
                            <img id="previewImg" class="max-w-full h-32 object-contain rounded">
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Modelo IA</label>
                            <select id="imageModelName" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                                <option value="demo_ecg_classifier">Demo ECG Classifier</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Threshold Qualidade</label>
                            <input type="number" id="qualityThreshold" value="0.3" min="0" max="1" step="0.1" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                        </div>
                    </div>
                    <div class="flex items-center">
                        <input type="checkbox" id="createFhir" checked class="mr-2">
                        <label for="createFhir" class="text-sm text-gray-700">Criar observação FHIR</label>
                    </div>
                    <button type="submit" class="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition">
                        <i class="fas fa-magic mr-2"></i>Analisar Imagem ECG
                    </button>
                </form>
            `;
            showModal('Análise de Imagem ECG', content);
            
            // Add file change listener
            document.getElementById('ecgImageFile').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('previewImg').src = e.target.result;
                        document.getElementById('imagePreview').classList.remove('hidden');
                    };
                    reader.readAsDataURL(file);
                }
            });
        }

        // Analyze ECG Image
        async function analyzeECGImage(event) {
            event.preventDefault();
            
            const formData = new FormData();
            formData.append('patient_id', document.getElementById('imagePatientId').value);
            formData.append('image_file', document.getElementById('ecgImageFile').files[0]);
            formData.append('model_name', document.getElementById('imageModelName').value);
            formData.append('quality_threshold', document.getElementById('qualityThreshold').value);
            formData.append('create_fhir', document.getElementById('createFhir').checked);
            
            try {
                const response = await fetch(`${API_BASE}/api/v1/ecg/image/analyze`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    closeModal();
                    showImageAnalysisResults(result);
                } else {
                    throw new Error(result.detail || 'Erro na análise');
                }
            } catch (error) {
                alert('Erro na análise: ' + error.message);
            }
        }

        function showImageAnalysisResults(result) {
            const qualityColor = result.digitization.quality_score > 0.6 ? 'green' : 
                                result.digitization.quality_score > 0.3 ? 'yellow' : 'red';
            
            const content = `
                <div class="space-y-6">
                    <!-- Header -->
                    <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h4 class="text-lg font-semibold text-blue-800 mb-2">
                            <i class="fas fa-image mr-2"></i>Análise de Imagem ECG Concluída
                        </h4>
                        <div class="grid grid-cols-2 gap-4 text-sm">
                            <div>
                                <strong>Paciente:</strong> ${result.patient_id}<br>
                                <strong>Arquivo:</strong> ${result.image_info.filename}<br>
                                <strong>Tamanho:</strong> ${(result.image_info.size_bytes / 1024).toFixed(1)} KB
                            </div>
                            <div>
                                <strong>Derivações:</strong> ${result.digitization.leads_extracted}<br>
                                <strong>Qualidade:</strong> <span class="text-${qualityColor}-600 font-semibold">${(result.digitization.quality_score * 100).toFixed(1)}%</span><br>
                                <strong>Grade:</strong> ${result.digitization.grid_detected ? 'Detectada' : 'Não detectada'}
                            </div>
                        </div>
                    </div>

                    <!-- Diagnosis -->
                    <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                        <h5 class="font-semibold text-green-800 mb-2">
                            <i class="fas fa-stethoscope mr-2"></i>Diagnóstico Principal
                        </h5>
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <p class="text-lg font-semibold text-green-700">${result.primary_diagnosis.diagnosis}</p>
                                <p class="text-sm text-green-600">Confiança: ${(result.primary_diagnosis.confidence * 100).toFixed(1)}%</p>
                            </div>
                            <div>
                                <p class="text-sm"><strong>Nível:</strong> ${result.primary_diagnosis.confidence_level}</p>
                                <p class="text-sm"><strong>Classe:</strong> ${result.primary_diagnosis.predicted_class}</p>
                            </div>
                        </div>
                    </div>

                    <!-- Quality Alerts -->
                    ${result.quality_alerts && result.quality_alerts.length > 0 ? `
                    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                        <h5 class="font-semibold text-yellow-800 mb-2">
                            <i class="fas fa-exclamation-triangle mr-2"></i>Alertas de Qualidade
                        </h5>
                        <ul class="list-disc list-inside text-sm text-yellow-700">
                            ${result.quality_alerts.map(alert => `<li>${alert}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}

                    <!-- Clinical Recommendations -->
                    <div class="bg-purple-50 border border-purple-200 rounded-lg p-4">
                        <h5 class="font-semibold text-purple-800 mb-2">
                            <i class="fas fa-user-md mr-2"></i>Recomendações Clínicas
                        </h5>
                        <div class="grid grid-cols-2 gap-4 text-sm">
                            <div>
                                <p><strong>Revisão Clínica:</strong> ${result.clinical_recommendations.clinical_review_required ? 'Necessária' : 'Não necessária'}</p>
                                <p><strong>Atenção Urgente:</strong> ${result.clinical_recommendations.urgent_attention ? 'Sim' : 'Não'}</p>
                            </div>
                            <div>
                                <p><strong>Follow-up:</strong> ${result.clinical_recommendations.follow_up_recommended ? 'Recomendado' : 'Não necessário'}</p>
                                <p><strong>Testes Adicionais:</strong> ${result.clinical_recommendations.additional_tests ? result.clinical_recommendations.additional_tests.length : 0}</p>
                            </div>
                        </div>
                    </div>

                    <!-- Technical Details -->
                    <details class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <summary class="font-semibold text-gray-800 cursor-pointer">
                            <i class="fas fa-cog mr-2"></i>Detalhes Técnicos
                        </summary>
                        <div class="mt-4 space-y-2 text-sm">
                            <p><strong>Modelo Usado:</strong> ${result.model_info.model_name}</p>
                            <p><strong>Taxa de Amostragem:</strong> ${result.digitization.sampling_rate_estimated} Hz</p>
                            <p><strong>Calibração:</strong> ${result.digitization.calibration_applied ? 'Aplicada' : 'Não aplicada'}</p>
                            <p><strong>FHIR:</strong> ${result.fhir_observation ? result.fhir_observation.created ? 'Criado' : 'Erro' : 'Não solicitado'}</p>
                        </div>
                    </details>
                </div>
            `;
            
            showResults(content);
        }

        // Show data analysis form (existing function)
        function showDataAnalysisForm() {
            const content = `
                <form onsubmit="analyzeECGData(event)" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">ID do Paciente</label>
                        <input type="text" id="dataPatientId" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Dados ECG (separados por vírgula)</label>
                        <textarea id="ecgData" required rows="4" placeholder="1.2, -0.5, 2.1, -1.8, 0.9..." class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"></textarea>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Taxa de Amostragem (Hz)</label>
                        <input type="number" id="samplingRate" value="500" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent">
                    </div>
                    <button type="submit" class="w-full bg-green-600 text-white py-2 rounded-lg hover:bg-green-700 transition">
                        <i class="fas fa-chart-line mr-2"></i>Analisar Dados ECG
                    </button>
                </form>
            `;
            showModal('Análise de Dados ECG', content);
        }

        // Analyze ECG Data (existing function)
        async function analyzeECGData(event) {
            event.preventDefault();
            
            const formData = new FormData();
            formData.append('patient_id', document.getElementById('dataPatientId').value);
            formData.append('ecg_data', document.getElementById('ecgData').value);
            formData.append('sampling_rate', document.getElementById('samplingRate').value);
            
            try {
                const response = await fetch(`${API_BASE}/api/v1/ecg/analyze`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    closeModal();
                    showDataAnalysisResults(result);
                } else {
                    throw new Error(result.detail || 'Erro na análise');
                }
            } catch (error) {
                alert('Erro na análise: ' + error.message);
            }
        }

        function showDataAnalysisResults(result) {
            const content = `
                <div class="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
                    <h4 class="text-lg font-semibold text-green-800 mb-2">Análise de Dados Concluída</h4>
                    <p class="text-green-700">Paciente: ${result.patient_id}</p>
                    <p class="text-green-700">Confiança: ${(result.results.confidence * 100).toFixed(1)}%</p>
                </div>
                <div class="grid md:grid-cols-2 gap-4">
                    <div>
                        <h5 class="font-semibold mb-2">Resultados da Análise</h5>
                        <pre class="bg-gray-100 p-3 rounded text-sm overflow-x-auto">${JSON.stringify(result.results, null, 2)}</pre>
                    </div>
                    <div>
                        <h5 class="font-semibold mb-2">Recomendações</h5>
                        <div class="space-y-2">
                            <p class="text-sm"><strong>Nível de Confiança:</strong> ${result.recommendations.confidence_level}</p>
                            <p class="text-sm"><strong>Revisão Clínica:</strong> ${result.recommendations.clinical_review ? 'Recomendada' : 'Não necessária'}</p>
                            <p class="text-sm"><strong>Follow-up:</strong> ${result.recommendations.follow_up ? 'Necessário' : 'Não necessário'}</p>
                        </div>
                    </div>
                </div>
            `;
            showResults(content);
        }

        // Show upload form (existing function)
        function showUploadForm() {
            const content = `
                <form onsubmit="uploadFile(event)" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">ID do Paciente</label>
                        <input type="text" id="uploadPatientId" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Arquivo ECG (CSV, TXT, NPY)</label>
                        <input type="file" id="ecgFile" accept=".csv,.txt,.npy" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Taxa de Amostragem (Hz)</label>
                        <input type="number" id="uploadSamplingRate" value="500" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent">
                    </div>
                    <button type="submit" class="w-full bg-purple-600 text-white py-2 rounded-lg hover:bg-purple-700 transition">
                        <i class="fas fa-upload mr-2"></i>Enviar e Analisar
                    </button>
                </form>
            `;
            showModal('Upload de Arquivo ECG', content);
        }

        // Upload file (existing function)
        async function uploadFile(event) {
            event.preventDefault();
            
            const formData = new FormData();
            formData.append('patient_id', document.getElementById('uploadPatientId').value);
            formData.append('file', document.getElementById('ecgFile').files[0]);
            formData.append('sampling_rate', document.getElementById('uploadSamplingRate').value);
            
            try {
                const response = await fetch(`${API_BASE}/api/v1/ecg/upload-file`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    closeModal();
                    showUploadResults(result);
                } else {
                    throw new Error(result.detail || 'Erro no upload');
                }
            } catch (error) {
                alert('Erro no upload: ' + error.message);
            }
        }

        function showUploadResults(result) {
            const content = `
                <div class="bg-purple-50 border border-purple-200 rounded-lg p-4 mb-4">
                    <h4 class="text-lg font-semibold text-purple-800 mb-2">Upload e Análise Concluídos</h4>
                    <p class="text-purple-700">Arquivo: ${result.file_info.filename}</p>
                    <p class="text-purple-700">Amostras: ${result.file_info.samples}</p>
                    <p class="text-purple-700">Confiança: ${(result.results.confidence * 100).toFixed(1)}%</p>
                </div>
                <pre class="bg-gray-100 p-3 rounded text-sm overflow-x-auto">${JSON.stringify(result, null, 2)}</pre>
            `;
            showResults(content);
        }

        // Show batch analysis form
        function showBatchAnalysisForm() {
            const content = `
                <form onsubmit="batchAnalyze(event)" class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">ID do Paciente</label>
                        <input type="text" id="batchPatientId" required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Múltiplas Imagens ECG (máx. 10)</label>
                        <input type="file" id="batchFiles" accept=".jpg,.jpeg,.png,.bmp,.tiff,.pdf" multiple required class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent">
                        <p class="text-sm text-gray-500 mt-1">Selecione até 10 arquivos de imagem</p>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-2">Modelo IA</label>
                        <select id="batchModelName" class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent">
                            <option value="demo_ecg_classifier">Demo ECG Classifier</option>
                        </select>
                    </div>
                    <button type="submit" class="w-full bg-orange-600 text-white py-2 rounded-lg hover:bg-orange-700 transition">
                        <i class="fas fa-tasks mr-2"></i>Analisar em Lote
                    </button>
                </form>
            `;
            showModal('Análise em Lote', content);
        }

        // Batch analyze
        async function batchAnalyze(event) {
            event.preventDefault();
            
            const files = document.getElementById('batchFiles').files;
            if (files.length > 10) {
                alert('Máximo 10 arquivos permitidos');
                return;
            }
            
            const formData = new FormData();
            formData.append('patient_id', document.getElementById('batchPatientId').value);
            formData.append('model_name', document.getElementById('batchModelName').value);
            
            for (let file of files) {
                formData.append('files', file);
            }
            
            try {
                const response = await fetch(`${API_BASE}/api/v1/ecg/image/batch-analyze`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    closeModal();
                    showBatchResults(result);
                } else {
                    throw new Error(result.detail || 'Erro na análise em lote');
                }
            } catch (error) {
                alert('Erro na análise em lote: ' + error.message);
            }
        }

        function showBatchResults(result) {
            const content = `
                <div class="bg-orange-50 border border-orange-200 rounded-lg p-4 mb-4">
                    <h4 class="text-lg font-semibold text-orange-800 mb-2">Análise em Lote Concluída</h4>
                    <div class="grid grid-cols-3 gap-4 text-sm">
                        <div><strong>Total:</strong> ${result.total_files}</div>
                        <div><strong>Sucessos:</strong> ${result.successful_analyses}</div>
                        <div><strong>Falhas:</strong> ${result.failed_analyses}</div>
                    </div>
                    <p class="text-orange-700 mt-2">Qualidade Média: ${(result.average_quality * 100).toFixed(1)}%</p>
                </div>
                <div class="space-y-3">
                    ${result.results.map((file, index) => `
                        <div class="border rounded-lg p-3 ${file.digitization_success ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}">
                            <div class="flex justify-between items-center">
                                <span class="font-medium">${file.filename}</span>
                                <span class="text-sm ${file.digitization_success ? 'text-green-600' : 'text-red-600'}">
                                    ${file.digitization_success ? 'Sucesso' : 'Falha'}
                                </span>
                            </div>
                            ${file.digitization_success ? `
                                <div class="text-sm mt-1">
                                    <span>Diagnóstico: ${file.primary_diagnosis || 'N/A'}</span> |
                                    <span>Confiança: ${file.confidence ? (file.confidence * 100).toFixed(1) + '%' : 'N/A'}</span> |
                                    <span>Qualidade: ${file.quality_score ? (file.quality_score * 100).toFixed(1) + '%' : 'N/A'}</span>
                                </div>
                            ` : `
                                <div class="text-sm text-red-600 mt-1">${file.error || 'Erro desconhecido'}</div>
                            `}
                        </div>
                    `).join('')}
                </div>
            `;
            showResults(content);
        }

        // Show models (existing function)
        async function showModels() {
            try {
                const response = await fetch(`${API_BASE}/api/v1/ecg/models`);
                const data = await response.json();
                
                const content = `
                    <div class="space-y-4">
                        <div class="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                            <h4 class="text-lg font-semibold text-indigo-800 mb-2">Modelos Disponíveis</h4>
                            <p class="text-indigo-700">Total: ${data.total_models} modelos</p>
                        </div>
                        <div class="space-y-2">
                            ${data.available_models.map(model => `
                                <div class="border rounded-lg p-3">
                                    <h5 class="font-semibold">${model}</h5>
                                    <pre class="text-xs bg-gray-100 p-2 rounded mt-2">${JSON.stringify(data.model_details[model], null, 2)}</pre>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
                showModal('Modelos de IA', content);
            } catch (error) {
                alert('Erro ao carregar modelos: ' + error.message);
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            checkHealth();
        });
    </script>
</body>
</html>"""
    
    with open(static_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)
    
    logger.info("Frontend aprimorado criado com suporte a análise de imagens")


# Criar aplicação FastAPI
app = FastAPI(
    title="CardioAI Pro - Sistema Completo com Análise de Imagens",
    description="""
    Sistema Completo de Análise de ECG com Inteligência Artificial e Digitalização de Imagens
    
    ## Funcionalidades Principais
    
    ### 🖼️ **Análise de Imagens ECG**
    * Upload de imagens ECG (JPG, PNG, PDF, etc.)
    * Digitalização automática de traçados
    * Extração de dados numéricos
    * Análise com modelos de IA
    * Diagnóstico automático
    
    ### 📊 **Análise de Dados ECG**
    * Processamento de dados numéricos
    * Modelos de IA avançados
    * Suporte a modelos .h5 pré-treinados
    * Análise em tempo real
    
    ### 🔬 **Funcionalidades Avançadas**
    * Análise em lote de múltiplas imagens
    * Compatibilidade FHIR R4
    * Recomendações clínicas automáticas
    * Sistema de qualidade e confiança
    * Interface web completa
    
    ## Endpoints Principais
    
    ### Análise de Imagens
    * `POST /api/v1/ecg/image/analyze` - Análise completa de imagem ECG
    * `POST /api/v1/ecg/image/digitize-only` - Apenas digitalização
    * `POST /api/v1/ecg/image/batch-analyze` - Análise em lote
    * `GET /api/v1/ecg/image/supported-formats` - Formatos suportados
    
    ### Análise de Dados
    * `POST /api/v1/ecg/analyze` - Análise de dados numéricos
    * `POST /api/v1/ecg/upload-file` - Upload de arquivo de dados
    * `GET /api/v1/ecg/models` - Modelos disponíveis
    
    ### Sistema
    * `GET /` - Interface web principal
    * `GET /health` - Status do sistema
    * `GET /docs` - Documentação da API
    """,
    version="2.0.0-complete",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")


# Incluir routers da API
try:
    from app.api.v1.ecg_image_endpoints import router as image_router
    app.include_router(image_router, prefix="/api/v1")
    logger.info("Router de análise de imagens incluído com sucesso")
except ImportError as e:
    logger.error(f"Erro ao incluir router de imagens: {str(e)}")

# Incluir endpoints existentes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Interface web principal do CardioAI Pro."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>CardioAI Pro</h1>
                <p>Interface web não encontrada. Acesse <a href="/docs">/docs</a> para a documentação da API.</p>
            </body>
        </html>
        """)


@app.get("/health")
async def health_check():
    """Endpoint de health check."""
    try:
        from app.services.model_service_enhanced import get_enhanced_model_service
        
        model_service = get_enhanced_model_service()
        models = model_service.list_models()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0-complete",
            "mode": "complete_with_images",
            "services": {
                "model_service": "running",
                "models_loaded": len(models),
                "available_models": models,
                "image_digitizer": "available",
                "frontend": "integrated",
                "backend": "running"
            },
            "capabilities": {
                "ecg_image_analysis": True,
                "ecg_data_analysis": True,
                "batch_processing": True,
                "fhir_compatibility": True,
                "h5_model_support": True,
                "web_interface": True
            },
            "system": {
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                "working_directory": os.getcwd(),
                "static_files": "served"
            }
        }
    except Exception as e:
        logger.error(f"Erro no health check: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/info")
async def system_info():
    """Informações detalhadas do sistema."""
    try:
        from app.services.model_service_enhanced import get_enhanced_model_service
        
        model_service = get_enhanced_model_service()
        models = model_service.list_models()
        model_info = {}
        
        for model_name in models:
            info = model_service.get_model_info(model_name)
            model_info[model_name] = info
        
        return {
            "system": {
                "name": "CardioAI Pro",
                "version": "2.0.0-complete",
                "mode": "complete_with_images",
                "description": "Sistema completo de análise de ECG com digitalização de imagens e IA",
                "startup_time": datetime.now().isoformat()
            },
            "interface": {
                "web_ui": "integrated",
                "main_page": "/",
                "documentation": "/docs",
                "redoc": "/redoc",
                "health_check": "/health"
            },
            "capabilities": {
                "ecg_image_analysis": True,
                "ecg_data_analysis": True,
                "image_digitization": True,
                "batch_processing": True,
                "fhir_compatibility": True,
                "h5_model_support": True,
                "real_time_processing": True,
                "web_interface": True,
                "api_documentation": True
            },
            "supported_formats": {
                "images": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf"],
                "data_files": [".csv", ".txt", ".npy"],
                "models": [".h5", ".pkl", ".joblib"]
            },
            "models": {
                "total": len(models),
                "available": models,
                "details": model_info
            },
            "api_version": "v1",
            "endpoints": {
                "image_analysis": "/api/v1/ecg/image/analyze",
                "batch_analysis": "/api/v1/ecg/image/batch-analyze",
                "data_analysis": "/api/v1/ecg/analyze",
                "file_upload": "/api/v1/ecg/upload-file",
                "models": "/api/v1/ecg/models"
            }
        }
    except Exception as e:
        logger.error(f"Erro ao obter informações: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Incluir endpoints de dados ECG existentes
@app.post("/api/v1/ecg/analyze")
async def analyze_ecg(
    patient_id: str = Form(...),
    ecg_data: str = Form(...),
    sampling_rate: int = Form(500),
    leads: Optional[str] = Form("I")
):
    """Analisa dados de ECG."""
    try:
        from app.services.model_service_enhanced import get_enhanced_model_service
        from app.schemas.fhir import create_ecg_observation
        import json
        
        # Parse dos dados ECG
        try:
            if ecg_data.startswith('[') and ecg_data.endswith(']'):
                ecg_array = np.array(json.loads(ecg_data))
            else:
                ecg_array = np.array([float(x.strip()) for x in ecg_data.split(',')])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Formato de dados ECG inválido: {str(e)}")
        
        model_service = get_enhanced_model_service()
        models = model_service.list_models()
        
        if not models:
            raise HTTPException(status_code=503, detail="Nenhum modelo disponível")
        
        model_name = models[0]
        result = model_service.predict_ecg(model_name, ecg_array)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        observation = create_ecg_observation(patient_id, ecg_data, sampling_rate, result)
        
        return {
            "patient_id": patient_id,
            "analysis_id": f"analysis_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "model_used": model_name,
            "sampling_rate": sampling_rate,
            "leads": leads.split(',') if leads else ["I"],
            "results": result,
            "fhir_observation": {
                "id": observation.id,
                "status": observation.status.value,
                "resource_type": observation.resourceType
            },
            "recommendations": {
                "confidence_level": "high" if result["confidence"] > 0.8 else "moderate" if result["confidence"] > 0.6 else "low",
                "clinical_review": result["confidence"] < 0.7,
                "follow_up": result["confidence"] < 0.5
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise de ECG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ecg/upload-file")
async def upload_ecg_file(
    patient_id: str = Form(...),
    sampling_rate: int = Form(500),
    file: UploadFile = File(...)
):
    """Upload e análise de arquivo ECG."""
    try:
        from app.services.model_service_enhanced import get_enhanced_model_service
        import io
        import pandas as pd
        
        if not file.filename.lower().endswith(('.csv', '.txt', '.npy')):
            raise HTTPException(status_code=400, detail="Formato de arquivo não suportado. Use CSV, TXT ou NPY.")
        
        content = await file.read()
        
        if file.filename.lower().endswith('.npy'):
            ecg_array = np.load(io.BytesIO(content))
        elif file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            ecg_array = df.iloc[:, 0].values
        else:  # .txt
            text_content = content.decode('utf-8')
            if ',' in text_content:
                ecg_array = np.array([float(x.strip()) for x in text_content.split(',')])
            else:
                ecg_array = np.array([float(x.strip()) for x in text_content.split('\n') if x.strip()])
        
        model_service = get_enhanced_model_service()
        models = model_service.list_models()
        
        if not models:
            raise HTTPException(status_code=503, detail="Nenhum modelo disponível")
        
        model_name = models[0]
        result = model_service.predict_ecg(model_name, ecg_array)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "patient_id": patient_id,
            "file_info": {
                "filename": file.filename,
                "size": len(content),
                "samples": len(ecg_array)
            },
            "analysis_id": f"file_analysis_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "model_used": model_name,
            "sampling_rate": sampling_rate,
            "results": result,
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no upload de arquivo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ecg/models")
async def list_models():
    """Lista modelos disponíveis."""
    try:
        from app.services.model_service_enhanced import get_enhanced_model_service
        
        model_service = get_enhanced_model_service()
        models = model_service.list_models()
        model_details = {}
        
        for model_name in models:
            info = model_service.get_model_info(model_name)
            model_details[model_name] = info
        
        return {
            "total_models": len(models),
            "available_models": models,
            "model_details": model_details,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

