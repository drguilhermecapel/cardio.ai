#!/usr/bin/env python3
"""
ECG ANALYZER AVANÇADO - PARTE 6: INTERFACE DE USUÁRIO E SISTEMA DE EXPORTAÇÃO
Implementa interface web, API REST e sistema completo de exportação
"""

import os
import io
import base64
import json
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import tempfile
import zipfile

# Web framework
from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Visualização e relatórios
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Exportação para diferentes formatos
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.drawing.image import Image as XLImage
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Para DICOM
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    logging.warning("pydicom não instalado - exportação DICOM não disponível")

class ECGWebInterface:
    """Interface web para o sistema ECG Analyzer"""
    
    def __init__(self, analyzer, port=5000):
        self.app = Flask(__name__)
        CORS(self.app)  # Permitir CORS para API
        self.analyzer = analyzer
        self.port = port
        
        # Configurações
        self.app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
        self.app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
        self.app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'jpg', 'jpeg', 'png', 'npy', 'mat', 'csv'}
        
        # Cache de resultados
        self.results_cache = {}
        
        # Registrar rotas
        self._register_routes()
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _allowed_file(self, filename):
        """Verifica se arquivo é permitido"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.app.config['ALLOWED_EXTENSIONS']
    
    def _register_routes(self):
        """Registra todas as rotas da aplicação"""
        
        @self.app.route('/')
        def index():
            """Página principal"""
            return render_template_string(self._get_index_template())
        
        @self.app.route('/api/analyze', methods=['POST'])
        def analyze():
            """Endpoint para análise de ECG"""
            try:
                # Verificar arquivo
                if 'file' not in request.files:
                    return jsonify({'error': 'Nenhum arquivo enviado'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'Arquivo sem nome'}), 400
                
                if not self._allowed_file(file.filename):
                    return jsonify({'error': 'Tipo de arquivo não permitido'}), 400
                
                # Salvar arquivo temporário
                filename = secure_filename(file.filename)
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Analisar
                result = self.analyzer.analyze_file(filepath)
                
                # Cachear resultado
                analysis_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + filename
                self.results_cache[analysis_id] = result
                
                # Limpar arquivo temporário
                os.remove(filepath)
                
                # Preparar resposta
                if result['success']:
                    response = {
                        'success': True,
                        'analysis_id': analysis_id,
                        'summary': self._prepare_summary(result),
                        'urgency': result.get('clinical_report', {}).get('urgency', 'NORMAL'),
                        'risk': result.get('clinical_report', {}).get('risk', 'MINIMAL')
                    }
                else:
                    response = {
                        'success': False,
                        'error': result.get('error', 'Erro desconhecido')
                    }
                
                return jsonify(response)
                
            except Exception as e:
                self.logger.error(f"Erro na análise: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/results/<analysis_id>')
        def get_results(analysis_id):
            """Obtém resultados completos"""
            if analysis_id not in self.results_cache:
                return jsonify({'error': 'Análise não encontrada'}), 404
            
            result = self.results_cache[analysis_id]
            
            # Converter para formato serializável
            serializable_result = self._make_serializable(result)
            
            return jsonify(serializable_result)
        
        @self.app.route('/api/export/<analysis_id>/<format>')
        def export_results(analysis_id, format):
            """Exporta resultados em diferentes formatos"""
            if analysis_id not in self.results_cache:
                return jsonify({'error': 'Análise não encontrada'}), 404
            
            if format not in ['pdf', 'excel', 'word', 'json', 'dicom', 'hl7']:
                return jsonify({'error': 'Formato não suportado'}), 400
            
            result = self.results_cache[analysis_id]
            
            try:
                exporter = ECGExporter()
                
                if format == 'pdf':
                    file_path = exporter.export_to_pdf(result)
                elif format == 'excel':
                    file_path = exporter.export_to_excel(result)
                elif format == 'word':
                    file_path = exporter.export_to_word(result)
                elif format == 'json':
                    file_path = exporter.export_to_json(result)
                elif format == 'dicom':
                    file_path = exporter.export_to_dicom(result)
                elif format == 'hl7':
                    file_path = exporter.export_to_hl7(result)
                
                return send_file(
                    file_path,
                    as_attachment=True,
                    download_name=f'ecg_analysis_{analysis_id}.{format}'
                )
                
            except Exception as e:
                self.logger.error(f"Erro na exportação: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/compare', methods=['POST'])
        def compare_ecgs():
            """Compara dois ECGs"""
            try:
                data = request.json
                if 'analysis_id1' not in data or 'analysis_id2' not in data:
                    return jsonify({'error': 'IDs de análise não fornecidos'}), 400
                
                id1, id2 = data['analysis_id1'], data['analysis_id2']
                
                if id1 not in self.results_cache or id2 not in self.results_cache:
                    return jsonify({'error': 'Análise não encontrada'}), 404
                
                # Realizar comparação
                from ecg_analyzer_part5 import ComparativeAnalyzer
                comparator = ComparativeAnalyzer()
                
                ecg1 = self._extract_ecg_data(self.results_cache[id1])
                ecg2 = self._extract_ecg_data(self.results_cache[id2])
                
                comparison = comparator.compare_ecgs(ecg1, ecg2, (id1, id2))
                
                return jsonify(self._make_serializable(comparison))
                
            except Exception as e:
                self.logger.error(f"Erro na comparação: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/batch', methods=['POST'])
        def batch_analyze():
            """Análise em lote"""
            try:
                files = request.files.getlist('files')
                if not files:
                    return jsonify({'error': 'Nenhum arquivo enviado'}), 400
                
                results = []
                for file in files:
                    if file and self._allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                        file.save(filepath)
                        
                        # Analisar
                        result = self.analyzer.analyze_file(filepath)
                        
                        # Adicionar à lista
                        results.append({
                            'filename': filename,
                            'success': result['success'],
                            'summary': self._prepare_summary(result) if result['success'] else None,
                            'error': result.get('error')
                        })
                        
                        # Limpar
                        os.remove(filepath)
                
                return jsonify({
                    'total': len(files),
                    'processed': len(results),
                    'results': results
                })
                
            except Exception as e:
                self.logger.error(f"Erro na análise em lote: {str(e)}")
                return jsonify({'error': str(e)}), 500
    
    def _get_index_template(self):
        """Template HTML da página principal"""
        return """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG Analyzer - Sistema Avançado de Análise</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }
        .card {
            border: none;
            box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
            transition: transform 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .upload-area {
            border: 3px dashed #dee2e6;
            border-radius: 1rem;
            padding: 3rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            border-color: #667eea;
            background-color: #f8f9fa;
        }
        .upload-area.dragover {
            border-color: #667eea;
            background-color: #e7e9fc;
        }
        .result-card {
            display: none;
        }
        .urgency-badge {
            font-size: 1.2rem;
            padding: 0.5rem 1rem;
        }
        .risk-indicator {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }
        .risk-bar {
            height: 100%;
            transition: width 0.5s ease;
        }
        .risk-minimal { background-color: #28a745; width: 20%; }
        .risk-low { background-color: #ffc107; width: 40%; }
        .risk-moderate { background-color: #fd7e14; width: 60%; }
        .risk-high { background-color: #dc3545; width: 80%; }
        .risk-very-high { background-color: #721c24; width: 100%; }
        .loading {
            display: none;
        }
        .feature-icon {
            font-size: 3rem;
            color: #667eea;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1 class="display-4"><i class="fas fa-heartbeat"></i> ECG Analyzer</h1>
            <p class="lead">Sistema Avançado de Análise Eletrocardiográfica com Inteligência Artificial</p>
        </div>
    </div>

    <div class="container">
        <!-- Upload Section -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        <h3 class="card-title mb-4">Análise de ECG</h3>
                        
                        <div class="upload-area" id="uploadArea">
                            <i class="fas fa-cloud-upload-alt fa-3x mb-3"></i>
                            <h4>Arraste o arquivo ECG aqui</h4>
                            <p class="text-muted">ou clique para selecionar</p>
                            <input type="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png,.npy,.mat,.csv" style="display: none;">
                            <small class="text-muted">Formatos aceitos: PDF, JPG, PNG, NPY, MAT, CSV (máx. 50MB)</small>
                        </div>
                        
                        <div class="loading text-center mt-4">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Analisando...</span>
                            </div>
                            <p class="mt-2">Analisando ECG... Por favor aguarde.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results Section -->
        <div class="row result-card" id="resultCard">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0"><i class="fas fa-chart-line"></i> Resultado da Análise</h4>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h5>Informações Gerais</h5>
                                <p><strong>ID da Análise:</strong> <span id="analysisId"></span></p>
                                <p><strong>Ritmo:</strong> <span id="rhythm"></span></p>
                                <p><strong>Frequência Cardíaca:</strong> <span id="heartRate"></span> bpm</p>
                            </div>
                            <div class="col-md-6">
                                <h5>Avaliação de Risco</h5>
                                <p><strong>Urgência:</strong> <span id="urgency" class="badge urgency-badge"></span></p>
                                <p><strong>Risco Cardiovascular:</strong></p>
                                <div class="risk-indicator">
                                    <div id="riskBar" class="risk-bar"></div>
                                </div>
                                <p class="text-center mt-2"><span id="riskText"></span></p>
                            </div>
                        </div>
                        
                        <hr>
                        
                        <div class="row">
                            <div class="col-12">
                                <h5>Principais Achados</h5>
                                <ul id="findingsList"></ul>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-12">
                                <h5>Recomendações</h5>
                                <ul id="recommendationsList"></ul>
                            </div>
                        </div>
                        
                        <hr>
                        
                        <div class="text-center">
                            <h5>Exportar Resultados</h5>
                            <div class="btn-group" role="group">
                                <button class="btn btn-primary" onclick="exportResults('pdf')">
                                    <i class="fas fa-file-pdf"></i> PDF
                                </button>
                                <button class="btn btn-success" onclick="exportResults('excel')">
                                    <i class="fas fa-file-excel"></i> Excel
                                </button>
                                <button class="btn btn-info" onclick="exportResults('word')">
                                    <i class="fas fa-file-word"></i> Word
                                </button>
                                <button class="btn btn-secondary" onclick="exportResults('json')">
                                    <i class="fas fa-file-code"></i> JSON
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Features Section -->
        <div class="row mt-5">
            <div class="col-12">
                <h2 class="text-center mb-4">Recursos do Sistema</h2>
            </div>
            <div class="col-md-3 text-center mb-4">
                <div class="card h-100">
                    <div class="card-body">
                        <i class="fas fa-brain feature-icon"></i>
                        <h5 class="card-title">IA Avançada</h5>
                        <p class="card-text">Modelos de deep learning treinados com milhares de ECGs</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3 text-center mb-4">
                <div class="card h-100">
                    <div class="card-body">
                        <i class="fas fa-waveform feature-icon"></i>
                        <h5 class="card-title">Delineação Precisa</h5>
                        <p class="card-text">Detecção automática de ondas P, QRS, T e intervalos</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3 text-center mb-4">
                <div class="card h-100">
                    <div class="card-body">
                        <i class="fas fa-stethoscope feature-icon"></i>
                        <h5 class="card-title">Interpretação Clínica</h5>
                        <p class="card-text">Laudos estruturados com recomendações baseadas em guidelines</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3 text-center mb-4">
                <div class="card h-100">
                    <div class="card-body">
                        <i class="fas fa-file-export feature-icon"></i>
                        <h5 class="card-title">Múltiplos Formatos</h5>
                        <p class="card-text">Exportação para PDF, Excel, Word, DICOM e HL7</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentAnalysisId = null;
        
        // Setup upload area
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.querySelector('.loading');
        const resultCard = document.getElementById('resultCard');
        
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
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            // Validate file
            const validTypes = ['application/pdf', 'image/jpeg', 'image/png'];
            const validExtensions = ['npy', 'mat', 'csv'];
            const extension = file.name.split('.').pop().toLowerCase();
            
            if (!validTypes.includes(file.type) && !validExtensions.includes(extension)) {
                alert('Tipo de arquivo não suportado!');
                return;
            }
            
            if (file.size > 50 * 1024 * 1024) {
                alert('Arquivo muito grande! Máximo 50MB.');
                return;
            }
            
            // Upload and analyze
            analyzeECG(file);
        }
        
        async function analyzeECG(file) {
            loading.style.display = 'block';
            resultCard.style.display = 'none';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    currentAnalysisId = data.analysis_id;
                    displayResults(data);
                } else {
                    alert('Erro na análise: ' + data.error);
                }
            } catch (error) {
                alert('Erro ao comunicar com servidor: ' + error);
            } finally {
                loading.style.display = 'none';
            }
        }
        
        function displayResults(data) {
            // Update UI with results
            document.getElementById('analysisId').textContent = data.analysis_id;
            document.getElementById('rhythm').textContent = data.summary.rhythm || 'N/A';
            document.getElementById('heartRate').textContent = data.summary.heart_rate || 'N/A';
            
            // Urgency
            const urgencyBadge = document.getElementById('urgency');
            urgencyBadge.textContent = data.urgency;
            urgencyBadge.className = 'badge urgency-badge';
            
            switch(data.urgency) {
                case 'CRITICAL':
                    urgencyBadge.classList.add('bg-danger');
                    break;
                case 'HIGH':
                    urgencyBadge.classList.add('bg-warning');
                    break;
                case 'MODERATE':
                    urgencyBadge.classList.add('bg-info');
                    break;
                default:
                    urgencyBadge.classList.add('bg-success');
            }
            
            // Risk
            const riskBar = document.getElementById('riskBar');
            const riskText = document.getElementById('riskText');
            riskBar.className = 'risk-bar risk-' + data.risk.toLowerCase().replace('_', '-');
            riskText.textContent = data.risk.replace('_', ' ');
            
            // Findings
            const findingsList = document.getElementById('findingsList');
            findingsList.innerHTML = '';
            if (data.summary.findings && data.summary.findings.length > 0) {
                data.summary.findings.forEach(finding => {
                    const li = document.createElement('li');
                    li.textContent = finding;
                    findingsList.appendChild(li);
                });
            } else {
                findingsList.innerHTML = '<li>Nenhum achado significativo</li>';
            }
            
            // Recommendations
            const recommendationsList = document.getElementById('recommendationsList');
            recommendationsList.innerHTML = '';
            if (data.summary.recommendations && data.summary.recommendations.length > 0) {
                data.summary.recommendations.forEach(rec => {
                    const li = document.createElement('li');
                    li.textContent = rec;
                    recommendationsList.appendChild(li);
                });
            } else {
                recommendationsList.innerHTML = '<li>Manter acompanhamento regular</li>';
            }
            
            resultCard.style.display = 'block';
        }
        
        async function exportResults(format) {
            if (!currentAnalysisId) {
                alert('Nenhuma análise disponível');
                return;
            }
            
            window.location.href = `/api/export/${currentAnalysisId}/${format}`;
        }
    </script>
</body>
</html>
        """
    
    def _prepare_summary(self, result: Dict) -> Dict:
        """Prepara resumo dos resultados"""
        summary = {}
        
        if 'clinical_report' in result and result['clinical_report'].get('success'):
            report = result['clinical_report']['report']
            
            summary['rhythm'] = report.rhythm
            summary['heart_rate'] = report.heart_rate
            summary['findings'] = [f.finding for f in report.findings[:5]]
            summary['recommendations'] = report.recommendations[:5]
        
        return summary
    
    def _extract_ecg_data(self, result: Dict) -> Dict:
        """Extrai dados ECG do resultado"""
        ecg_data = {
            'timestamp': datetime.fromisoformat(result['timestamp']),
            'heart_rate': 0,
            'rhythm': 'Unknown',
            'intervals': {},
            'findings': [],
            'ml_predictions': {}
        }
        
        if 'clinical_report' in result and result['clinical_report'].get('success'):
            report = result['clinical_report']['report']
            ecg_data['heart_rate'] = report.heart_rate
            ecg_data['rhythm'] = report.rhythm
            ecg_data['intervals'] = report.intervals
            ecg_data['findings'] = [f.finding for f in report.findings]
        
        if 'ml_predictions' in result:
            ecg_data['ml_predictions'] = result['ml_predictions'].get('predictions', {})
        
        return ecg_data
    
    def _make_serializable(self, obj: Any) -> Any:
        """Converte objeto para formato serializável JSON"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def run(self):
        """Inicia servidor web"""
        self.logger.info(f"Iniciando servidor ECG Analyzer na porta {self.port}")
        self.logger.info(f"Acesse: http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)


class ECGExporter:
    """Sistema de exportação para múltiplos formatos"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def export_to_pdf(self, result: Dict) -> str:
        """Exporta para PDF com formatação profissional"""
        
        filename = os.path.join(self.temp_dir, f"ecg_report_{datetime.now():%Y%m%d_%H%M%S}.pdf")
        doc = SimpleDocTemplate(filename, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Estilo customizado
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Título
        story.append(Paragraph("Relatório de Análise Eletrocardiográfica", title_style))
        story.append(Spacer(1, 20))
        
        # Informações do exame
        if 'clinical_report' in result and result['clinical_report'].get('success'):
            report = result['clinical_report']['report']
            
            # Dados básicos
            data = [
                ['Data/Hora:', datetime.now().strftime('%d/%m/%Y %H:%M')],
                ['ID do Paciente:', report.patient_id],
                ['Qualidade do Exame:', f"{report.quality_score:.1%}"]
            ]
            
            t = Table(data, colWidths=[3*inch, 4*inch])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(t)
            story.append(Spacer(1, 20))
            
            # Análise do ritmo
            story.append(Paragraph("Análise do Ritmo", styles['Heading2']))
            rhythm_data = [
                ['Ritmo:', report.rhythm],
                ['Frequência Cardíaca:', f"{report.heart_rate} bpm"]
            ]
            
            t = Table(rhythm_data, colWidths=[3*inch, 4*inch])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#e7e9fc')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(t)
            story.append(Spacer(1, 20))
            
            # Intervalos
            if report.intervals:
                story.append(Paragraph("Intervalos Medidos", styles['Heading2']))
                interval_data = [['Intervalo', 'Valor (ms)', 'Referência']]
                
                references = {
                    'PR': '120-200',
                    'QRS': '80-120',
                    'QT': '350-450',
                    'QTc': '350-450'
                }
                
                for interval, value in report.intervals.items():
                    ref = references.get(interval, 'N/A')
                    interval_data.append([interval, f"{value:.0f}", ref])
                
                t = Table(interval_data, colWidths=[2*inch, 2*inch, 2*inch])
                t.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(t)
                story.append(Spacer(1, 20))
            
            # Achados principais
            if report.findings:
                story.append(Paragraph("Achados Principais", styles['Heading2']))
                
                for i, finding in enumerate(report.findings, 1):
                    # Determinar cor baseada na urgência
                    if finding.urgency.value == 'CRÍTICO':
                        color = colors.red
                    elif finding.urgency.value == 'ALTO':
                        color = colors.orange
                    else:
                        color = colors.black
                    
                    finding_style = ParagraphStyle(
                        f'Finding{i}',
                        parent=styles['Normal'],
                        textColor=color,
                        fontSize=11,
                        spaceAfter=10
                    )
                    
                    text = f"<b>{i}. {finding.finding}</b>"
                    if finding.clinical_significance:
                        text += f" - {finding.clinical_significance}"
                    
                    story.append(Paragraph(text, finding_style))
                    
                    if finding.criteria:
                        criteria_text = "Critérios: " + ", ".join(finding.criteria[:3])
                        story.append(Paragraph(criteria_text, styles['Normal']))
                    
                    story.append(Spacer(1, 10))
            
            # Estratificação de risco
            story.append(PageBreak())
            story.append(Paragraph("Estratificação de Risco", styles['Heading2']))
            
            risk_colors = {
                'MUITO ALTO': colors.red,
                'ALTO': colors.orange,
                'MODERADO': colors.yellow,
                'BAIXO': colors.lightgreen,
                'MÍNIMO': colors.green
            }
            
            risk_data = [
                ['Risco Cardiovascular:', report.risk_stratification.value],
                ['Urgência:', report.urgency_level.value]
            ]
            
            t = Table(risk_data, colWidths=[3*inch, 4*inch])
            # Aplicar cor baseada no risco
            risk_color = risk_colors.get(report.risk_stratification.value, colors.grey)
            
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (1, 0), (1, 0), risk_color),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(t)
            story.append(Spacer(1, 20))
            
            # Recomendações
            if report.recommendations:
                story.append(Paragraph("Recomendações", styles['Heading2']))
                
                for i, rec in enumerate(report.recommendations, 1):
                    rec_text = f"{i}. {rec}"
                    story.append(Paragraph(rec_text, styles['Normal']))
                    story.append(Spacer(1, 5))
            
            # Adicionar gráficos se disponíveis
            if 'delineation' in result:
                story.append(PageBreak())
                story.append(Paragraph("Análise Gráfica", styles['Heading2']))
                
                # Criar gráfico temporário
                fig, ax = plt.subplots(figsize=(7, 4))
                
                # Plotar sinal principal (DII)
                if 'consolidated' in result['delineation']:
                    signal = result['delineation']['consolidated'].get('signal', [])
                    if len(signal) > 0:
                        time = np.arange(len(signal)) / 500  # Assumindo 500Hz
                        ax.plot(time[:2500], signal[:2500], 'b-', linewidth=0.5)
                        ax.set_xlabel('Tempo (s)')
                        ax.set_ylabel('Amplitude (mV)')
                        ax.set_title('ECG - Derivação II (primeiros 5s)')
                        ax.grid(True, alpha=0.3)
                
                # Salvar e adicionar ao PDF
                graph_path = os.path.join(self.temp_dir, 'ecg_graph.png')
                plt.savefig(graph_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                if os.path.exists(graph_path):
                    img = Image(graph_path, width=6*inch, height=3.5*inch)
                    story.append(img)
            
            # Rodapé
            story.append(Spacer(1, 30))
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                alignment=TA_CENTER
            )
            
            footer_text = f"""
            Este laudo foi gerado por sistema de IA e deve ser correlacionado com dados clínicos.<br/>
            Confiança da interpretação: {report.interpreter_confidence:.1%}<br/>
            Sistema ECG Analyzer v1.0 - {datetime.now().strftime('%d/%m/%Y %H:%M')}
            """
            story.append(Paragraph(footer_text, footer_style))
        
        # Construir PDF
        doc.build(story)
        
        return filename
    
    def export_to_excel(self, result: Dict) -> str:
        """Exporta para Excel com múltiplas abas"""