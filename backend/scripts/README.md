# Scripts de Setup do CardioAI Pro

Este diretório contém os scripts para implementar os próximos passos do CardioAI Pro conforme solicitado.

## 🎯 Próximos Passos Implementados

1. **Download dos Datasets**: MIT-BIH Arrhythmia Database, PTB-XL Database, MIMIC-IV-ECG Database, Icentia11k Database, PhysioNet Challenge Datasets
2. **Setup de Hardware**: Configurar GPU NVIDIA para treinamento (com fallback para MPS/CPU)
3. **Implementação de modelo híbrido de IA**: Arquitetura CNN-BiLSTM-Transformer para análise precisa dos ECGs
4. **Treinamento da IA**: Baseado nos datasets para melhor acurácia diagnóstica

## 📁 Scripts Disponíveis

### `run_complete_setup.py` - Script Principal
Executa todos os passos sequencialmente:
```bash
cd backend/scripts
python run_complete_setup.py
```

### `setup_datasets.py` - Download e Configuração dos Datasets
```bash
python setup_datasets.py
```

**Funcionalidades:**
- Download automático de MIT-BIH e PTB-XL
- Download de PhysioNet Challenge 2020/2021
- Instruções para MIMIC-IV-ECG e Icentia11k (requerem credenciais)
- Análise estatística dos datasets
- Criação de dataset unificado em HDF5

### `setup_gpu_training.py` - Configuração de Hardware
```bash
python setup_gpu_training.py
```

**Funcionalidades:**
- Detecção automática de GPU (CUDA/MPS/CPU)
- Teste de operações GPU
- Configuração otimizada para cada tipo de hardware
- Inicialização do modelo híbrido
- Setup do serviço de ML avançado

### `train_model.py` - Treinamento do Modelo Híbrido
```bash
python train_model.py
```

**Funcionalidades:**
- Treinamento com curriculum learning
- Arquitetura híbrida CNN-BiLSTM-Transformer
- Features multimodais (1D + 2D spectrograms + wavelets)
- Augmentação de dados para robustez
- Early stopping e checkpoints
- Métricas de avaliação completas

## 🔧 Requisitos

### Dependências Python
```bash
# Instalar dependências
pip install torch torchvision torchaudio
pip install wfdb h5py pandas numpy scipy
pip install tqdm opencv-python PyWavelets
pip install scikit-learn matplotlib seaborn
```

### Hardware Recomendado
- **GPU**: NVIDIA com CUDA 11.0+ (8GB+ VRAM recomendado)
- **CPU**: 8+ cores para processamento paralelo
- **RAM**: 16GB+ para datasets grandes
- **Storage**: 100GB+ para todos os datasets

## 📊 Datasets Suportados

| Dataset | Tamanho | Registros | Derivações | Status |
|---------|---------|-----------|------------|--------|
| MIT-BIH | ~23MB | 48 | 2 | ✅ Download automático |
| PTB-XL | ~3GB | 21,799 | 12 | ✅ Download automático |
| PhysioNet 2020 | ~6GB | 43,101 | 12 | ✅ Download automático |
| PhysioNet 2021 | ~12GB | 88,253 | 12 | ✅ Download automático |
| MIMIC-IV-ECG | ~50GB | 800,000+ | 12 | 📋 Requer credenciais |
| Icentia11k | ~2TB | 11,000 | 1 | 📋 Requer credenciais |

## 🧠 Arquitetura do Modelo

### Modelo Híbrido CNN-BiLSTM-Transformer
- **CNN**: DenseNet com attention para extração de features
- **BiLSTM**: Análise temporal bidirecional
- **Transformer**: Correlações espaciais-temporais com 8 attention heads
- **Ensemble**: Votação ponderada entre os três componentes

### Features Multimodais
1. **1D Signals**: Sinais ECG brutos processados
2. **2D Spectrograms**: Representação tempo-frequência
3. **Wavelets**: Decomposição wavelet para análise multi-resolução

### Técnicas de Treinamento
- **Curriculum Learning**: Treinamento progressivo por dificuldade
- **Data Augmentation**: Ruído, escala, deslocamento temporal
- **Focal Loss**: Para lidar com classes desbalanceadas
- **Early Stopping**: Prevenção de overfitting

## 📈 Métricas de Performance

### Objetivos de Acurácia
- **Alvo**: 99.6% conforme especificações científicas
- **Métricas**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Validação**: Split 80/20 com validação cruzada opcional

### Monitoramento
- Logs detalhados em tempo real
- Checkpoints automáticos do melhor modelo
- Métricas de qualidade de sinal
- Tempo de processamento por amostra

## 🚀 Execução Rápida

```bash
# Navegar para o diretório
cd backend/scripts

# Executar setup completo (recomendado)
python run_complete_setup.py

# OU executar passos individuais:
python setup_datasets.py
python setup_gpu_training.py
python train_model.py
```

## 📝 Logs e Outputs

Todos os scripts geram logs detalhados:
- `dataset_setup.log` - Log do setup de datasets
- `gpu_training_setup.log` - Log da configuração de GPU
- `model_training.log` - Log do treinamento
- `complete_setup.log` - Log da execução completa

## ⚠️ Notas Importantes

1. **MIMIC-IV-ECG e Icentia11k** requerem:
   - Conta no PhysioNet
   - Treinamento CITI completo
   - Aprovação de acesso aos datasets

2. **GPU Training**:
   - CUDA é detectado automaticamente
   - MPS (Apple Silicon) suportado
   - Fallback para CPU se necessário

3. **Espaço em Disco**:
   - Datasets completos: ~100GB
   - Modelos e checkpoints: ~5GB
   - Logs e cache: ~1GB

## 🔍 Troubleshooting

### Erro de GPU
```bash
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Verificar MPS (Apple)
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Erro de Dependências
```bash
# Reinstalar dependências
pip install --upgrade torch torchvision torchaudio
pip install --upgrade wfdb h5py pandas numpy
```

### Erro de Memória
- Reduzir batch_size no arquivo de configuração
- Usar menos workers para data loading
- Limitar max_records nos datasets

## 📞 Suporte

Para problemas ou dúvidas:
1. Verificar logs detalhados
2. Consultar documentação do CardioAI Pro
3. Reportar issues no repositório GitHub
