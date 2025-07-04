// CardioAI Pro - Funcionalidades Avançadas
class CardioAIInterface {
    constructor() {
        this.apiBase = 'https://15000-iyv5qeky2ds1x06rhqju4-10e6e8b4.manusvm.computer';
        this.selectedFile = null;
        this.analysisHistory = [];
        this.init();
    }

    init() {
        this.loadSystemStatus();
        this.setupEventListeners();
        this.loadAnalysisHistory();
        this.startStatusMonitoring();
    }

    // Configurar event listeners
    setupEventListeners() {
        // Drag and drop
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea) {
            uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadArea.addEventListener('drop', this.handleDrop.bind(this));
            uploadArea.addEventListener('click', () => {
                document.getElementById('fileInput').click();
            });
        }

        // File input
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileInputChange.bind(this));
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));
    }

    // Manipular drag over
    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    // Manipular drag leave
    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }

    // Manipular drop
    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFileSelect(files[0]);
        }
    }

    // Manipular mudança no input de arquivo
    handleFileInputChange(e) {
        if (e.target.files.length > 0) {
            this.handleFileSelect(e.target.files[0]);
        }
    }

    // Manipular atalhos de teclado
    handleKeyboardShortcuts(e) {
        // Ctrl+U para upload
        if (e.ctrlKey && e.key === 'u') {
            e.preventDefault();
            document.getElementById('fileInput').click();
        }
        
        // Enter para analisar (se arquivo selecionado)
        if (e.key === 'Enter' && this.selectedFile) {
            this.analyzeFile();
        }
        
        // Escape para limpar
        if (e.key === 'Escape') {
            this.clearSelection();
        }
    }

    // Carregar status do sistema
    async loadSystemStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();
            
            this.updateStatusDisplay(data);
            
            // Carregar informações detalhadas
            const infoResponse = await fetch(`${this.apiBase}/`);
            const infoData = await infoResponse.json();
            
            this.updateSystemInfo(infoData);
            
        } catch (error) {
            console.error('Erro ao carregar status:', error);
            this.showError('Erro ao conectar com o servidor');
            this.updateStatusDisplay({ status: 'error' });
        }
    }

    // Atualizar display de status
    updateStatusDisplay(data) {
        const statusGrid = document.getElementById('statusGrid');
        if (!statusGrid) return;

        const getStatusColor = (status) => {
            switch (status) {
                case 'healthy': return '#4CAF50';
                case 'error': return '#f44336';
                default: return '#ff9800';
            }
        };

        const getBiasColor = (detected) => {
            return detected ? '#ff9800' : '#4CAF50';
        };

        statusGrid.innerHTML = `
            <div class="status-item" style="background: linear-gradient(135deg, ${getStatusColor(data.status)} 0%, ${getStatusColor(data.status)}dd 100%);">
                <i class="fas fa-heartbeat"></i>
                <h3>Sistema</h3>
                <p>${data.status === 'healthy' ? 'Funcionando' : data.status === 'error' ? 'Erro' : 'Carregando'}</p>
            </div>
            <div class="status-item" style="background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);">
                <i class="fas fa-brain"></i>
                <h3>Modelo</h3>
                <p>${data.model_type || 'N/A'}</p>
            </div>
            <div class="status-item" style="background: linear-gradient(135deg, ${getBiasColor(data.bias_detected)} 0%, ${getBiasColor(data.bias_detected)}dd 100%);">
                <i class="fas fa-shield-alt"></i>
                <h3>Bias</h3>
                <p>${data.bias_detected ? 'Detectado' : 'OK'}</p>
            </div>
            <div class="status-item" style="background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%);">
                <i class="fas fa-cog"></i>
                <h3>Modo</h3>
                <p>${data.using_demo_model ? 'Demo' : 'Produção'}</p>
            </div>
        `;
    }

    // Atualizar informações do sistema
    updateSystemInfo(data) {
        // Adicionar informações detalhadas se necessário
        if (data.model_info && data.model_info.note) {
            this.showSystemNote(data.model_info.note);
        }
    }

    // Mostrar nota do sistema
    showSystemNote(note) {
        const statusCard = document.querySelector('.status-card');
        if (statusCard && !statusCard.querySelector('.system-note')) {
            const noteElement = document.createElement('div');
            noteElement.className = 'system-note';
            noteElement.style.cssText = `
                background: linear-gradient(135deg, #FFC107 0%, #FF8F00 100%);
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                font-size: 0.9rem;
            `;
            noteElement.innerHTML = `<i class="fas fa-info-circle"></i> <strong>Nota:</strong> ${note}`;
            statusCard.appendChild(noteElement);
        }
    }

    // Manipular seleção de arquivo
    handleFileSelect(file) {
        // Validar tipo de arquivo
        const allowedTypes = [
            'text/csv',
            'text/plain',
            'application/octet-stream', // Para .npy
            'image/jpeg',
            'image/jpg',
            'image/png',
            'image/bmp'
        ];

        const allowedExtensions = ['.csv', '.txt', '.npy', '.jpg', '.jpeg', '.png', '.bmp'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

        if (!allowedExtensions.includes(fileExtension)) {
            this.showError(`Tipo de arquivo não suportado: ${fileExtension}`);
            return;
        }

        // Validar tamanho (máximo 10MB)
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            this.showError(`Arquivo muito grande. Máximo permitido: 10MB`);
            return;
        }

        this.selectedFile = file;
        this.updateFileInfo(file);
        this.hideError();
    }

    // Atualizar informações do arquivo
    updateFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const fileType = document.getElementById('fileType');

        if (fileName) fileName.textContent = `Nome: ${file.name}`;
        if (fileSize) fileSize.textContent = `Tamanho: ${this.formatFileSize(file.size)}`;
        if (fileType) fileType.textContent = `Tipo: ${this.getFileTypeDescription(file)}`;

        if (fileInfo) {
            fileInfo.classList.add('show');
        }

        // Adicionar preview se for imagem
        this.addImagePreview(file);
    }

    // Adicionar preview de imagem
    addImagePreview(file) {
        const fileInfo = document.getElementById('fileInfo');
        if (!fileInfo) return;

        // Remover preview anterior
        const existingPreview = fileInfo.querySelector('.image-preview');
        if (existingPreview) {
            existingPreview.remove();
        }

        // Adicionar preview se for imagem
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const preview = document.createElement('div');
                preview.className = 'image-preview';
                preview.style.cssText = `
                    margin-top: 15px;
                    text-align: center;
                `;
                preview.innerHTML = `
                    <h4><i class="fas fa-eye"></i> Preview</h4>
                    <img src="${e.target.result}" style="max-width: 200px; max-height: 150px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                `;
                fileInfo.appendChild(preview);
            };
            reader.readAsDataURL(file);
        }
    }

    // Obter descrição do tipo de arquivo
    getFileTypeDescription(file) {
        const extension = '.' + file.name.split('.').pop().toLowerCase();
        const descriptions = {
            '.csv': 'Dados CSV (Comma Separated Values)',
            '.txt': 'Arquivo de texto com dados ECG',
            '.npy': 'Array NumPy (dados binários)',
            '.jpg': 'Imagem JPEG de ECG',
            '.jpeg': 'Imagem JPEG de ECG',
            '.png': 'Imagem PNG de ECG',
            '.bmp': 'Imagem Bitmap de ECG'
        };
        return descriptions[extension] || file.type || 'Tipo desconhecido';
    }

    // Formatar tamanho do arquivo
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Analisar arquivo
    async analyzeFile() {
        if (!this.selectedFile) {
            this.showError('Por favor, selecione um arquivo primeiro');
            return;
        }

        const analyzeBtn = document.getElementById('analyzeBtn');
        const loading = document.getElementById('loading');
        const resultsSection = document.getElementById('resultsSection');

        // Mostrar loading
        if (analyzeBtn) analyzeBtn.disabled = true;
        if (loading) loading.classList.add('show');
        if (resultsSection) resultsSection.classList.remove('show');
        this.hideError();

        // Simular progresso
        this.simulateProgress();

        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);

            const response = await fetch(`${this.apiBase}/api/v1/ecg/analyze`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.showResults(data);
                this.addToHistory(data);
            } else {
                throw new Error(data.detail || 'Erro na análise');
            }
        } catch (error) {
            console.error('Erro na análise:', error);
            this.showError(`Erro na análise: ${error.message}`);
        } finally {
            if (analyzeBtn) analyzeBtn.disabled = false;
            if (loading) loading.classList.remove('show');
            this.hideProgress();
        }
    }

    // Simular progresso
    simulateProgress() {
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        
        if (progressBar && progressFill) {
            progressBar.style.display = 'block';
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                progressFill.style.width = progress + '%';
                
                if (progress >= 90) {
                    clearInterval(interval);
                }
            }, 200);
        }
    }

    // Esconder progresso
    hideProgress() {
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        
        if (progressBar && progressFill) {
            progressFill.style.width = '100%';
            setTimeout(() => {
                progressBar.style.display = 'none';
                progressFill.style.width = '0%';
            }, 500);
        }
    }

    // Mostrar resultados
    showResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsContent = document.getElementById('resultsContent');
        
        if (!resultsContent) return;

        const analysis = data.analysis;
        const confidence = this.getConfidenceLevel(analysis.probability);
        const confidenceColor = this.getConfidenceColor(confidence);

        resultsContent.innerHTML = `
            <div class="result-card">
                <div class="result-header">
                    <div class="result-icon">
                        <i class="fas fa-stethoscope"></i>
                    </div>
                    <div>
                        <div class="result-title">${analysis.diagnosis || 'Diagnóstico Indisponível'}</div>
                        <div>Arquivo: ${data.filename}</div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">Analisado em: ${new Date().toLocaleString('pt-BR')}</div>
                    </div>
                </div>
                
                <div class="result-details">
                    <div class="detail-item">
                        <div class="detail-label">Probabilidade</div>
                        <div class="detail-value">${((analysis.probability || 0) * 100).toFixed(1)}%</div>
                    </div>
                    <div class="detail-item" style="background: ${confidenceColor};">
                        <div class="detail-label">Confiança</div>
                        <div class="detail-value">${confidence}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Classe ID</div>
                        <div class="detail-value">${analysis.class_id || 'N/A'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Tipo de Arquivo</div>
                        <div class="detail-value">${data.file_type}</div>
                    </div>
                </div>
                
                ${analysis.note ? `
                    <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.2); border-radius: 10px;">
                        <strong><i class="fas fa-info-circle"></i> Nota:</strong> ${analysis.note}
                    </div>
                ` : ''}
                
                <div style="margin-top: 20px; text-align: center;">
                    <button class="btn" onclick="cardioAI.downloadReport()" style="margin-right: 10px;">
                        <i class="fas fa-download"></i> Baixar Relatório
                    </button>
                    <button class="btn" onclick="cardioAI.shareResults()" style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);">
                        <i class="fas fa-share"></i> Compartilhar
                    </button>
                </div>
            </div>
        `;
        
        if (resultsSection) {
            resultsSection.classList.add('show');
        }
    }

    // Obter nível de confiança
    getConfidenceLevel(probability) {
        if (probability >= 0.8) return 'Alto';
        if (probability >= 0.6) return 'Médio';
        return 'Baixo';
    }

    // Obter cor da confiança
    getConfidenceColor(confidence) {
        switch (confidence) {
            case 'Alto': return 'rgba(76, 175, 80, 0.3)';
            case 'Médio': return 'rgba(255, 193, 7, 0.3)';
            case 'Baixo': return 'rgba(244, 67, 54, 0.3)';
            default: return 'rgba(255, 255, 255, 0.2)';
        }
    }

    // Adicionar ao histórico
    addToHistory(data) {
        this.analysisHistory.unshift({
            ...data,
            timestamp: new Date().toISOString()
        });
        
        // Manter apenas os últimos 10
        if (this.analysisHistory.length > 10) {
            this.analysisHistory = this.analysisHistory.slice(0, 10);
        }
        
        this.saveAnalysisHistory();
    }

    // Carregar histórico de análises
    loadAnalysisHistory() {
        try {
            const saved = localStorage.getItem('cardioai_history');
            if (saved) {
                this.analysisHistory = JSON.parse(saved);
            }
        } catch (error) {
            console.error('Erro ao carregar histórico:', error);
        }
    }

    // Salvar histórico de análises
    saveAnalysisHistory() {
        try {
            localStorage.setItem('cardioai_history', JSON.stringify(this.analysisHistory));
        } catch (error) {
            console.error('Erro ao salvar histórico:', error);
        }
    }

    // Baixar relatório
    downloadReport() {
        if (this.analysisHistory.length === 0) {
            this.showError('Nenhuma análise disponível para download');
            return;
        }

        const lastAnalysis = this.analysisHistory[0];
        const report = this.generateReport(lastAnalysis);
        
        const blob = new Blob([report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `cardioai_relatorio_${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    }

    // Gerar relatório
    generateReport(data) {
        return `
RELATÓRIO DE ANÁLISE ECG - CardioAI Pro
========================================

Data/Hora: ${new Date(data.timestamp).toLocaleString('pt-BR')}
Arquivo: ${data.filename}
Tipo: ${data.file_type}

RESULTADO DA ANÁLISE:
--------------------
Diagnóstico: ${data.analysis.diagnosis || 'N/A'}
Probabilidade: ${((data.analysis.probability || 0) * 100).toFixed(1)}%
Classe ID: ${data.analysis.class_id || 'N/A'}
Confiança: ${this.getConfidenceLevel(data.analysis.probability)}

${data.analysis.note ? `
OBSERVAÇÕES:
-----------
${data.analysis.note}
` : ''}

INFORMAÇÕES TÉCNICAS:
--------------------
Sistema: CardioAI Pro v3.1.0
Modelo: Detecção automática de bias ativada
Processamento: ${new Date().toLocaleString('pt-BR')}

AVISO IMPORTANTE:
----------------
Este relatório é gerado por um sistema de inteligência artificial
e deve ser interpretado por profissionais médicos qualificados.
Não substitui consulta médica ou exame clínico.
        `.trim();
    }

    // Compartilhar resultados
    shareResults() {
        if (this.analysisHistory.length === 0) {
            this.showError('Nenhuma análise disponível para compartilhar');
            return;
        }

        const lastAnalysis = this.analysisHistory[0];
        const shareText = `Análise ECG - CardioAI Pro
Diagnóstico: ${lastAnalysis.analysis.diagnosis || 'N/A'}
Probabilidade: ${((lastAnalysis.analysis.probability || 0) * 100).toFixed(1)}%
Data: ${new Date(lastAnalysis.timestamp).toLocaleString('pt-BR')}`;

        if (navigator.share) {
            navigator.share({
                title: 'Resultado CardioAI Pro',
                text: shareText
            });
        } else {
            // Fallback: copiar para clipboard
            navigator.clipboard.writeText(shareText).then(() => {
                this.showSuccess('Resultado copiado para a área de transferência!');
            });
        }
    }

    // Limpar seleção
    clearSelection() {
        this.selectedFile = null;
        const fileInfo = document.getElementById('fileInfo');
        const resultsSection = document.getElementById('resultsSection');
        
        if (fileInfo) fileInfo.classList.remove('show');
        if (resultsSection) resultsSection.classList.remove('show');
        
        const fileInput = document.getElementById('fileInput');
        if (fileInput) fileInput.value = '';
        
        this.hideError();
    }

    // Monitoramento de status
    startStatusMonitoring() {
        // Atualizar status a cada 30 segundos
        setInterval(() => {
            this.loadSystemStatus();
        }, 30000);
    }

    // Mostrar erro
    showError(message) {
        const errorMessage = document.getElementById('errorMessage');
        const errorText = document.getElementById('errorText');
        
        if (errorText) errorText.textContent = message;
        if (errorMessage) {
            errorMessage.classList.add('show');
            setTimeout(() => this.hideError(), 5000);
        }
    }

    // Mostrar sucesso
    showSuccess(message) {
        // Criar elemento de sucesso temporário
        const successDiv = document.createElement('div');
        successDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            animation: slideInRight 0.3s ease;
        `;
        successDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
        
        document.body.appendChild(successDiv);
        
        setTimeout(() => {
            successDiv.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => document.body.removeChild(successDiv), 300);
        }, 3000);
    }

    // Esconder erro
    hideError() {
        const errorMessage = document.getElementById('errorMessage');
        if (errorMessage) {
            errorMessage.classList.remove('show');
        }
    }
}

// Inicializar aplicação
let cardioAI;
document.addEventListener('DOMContentLoaded', function() {
    cardioAI = new CardioAIInterface();
});

// Função global para análise (compatibilidade)
function analyzeFile() {
    if (cardioAI) {
        cardioAI.analyzeFile();
    }
}

