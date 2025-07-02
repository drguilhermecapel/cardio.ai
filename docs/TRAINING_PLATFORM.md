# Plataforma de Treinamento de IA - CardioAI Pro

## Novidades da Versão 1.0.0

### 🚀 Nova Plataforma de Treinamento de IA

O CardioAI Pro agora inclui uma plataforma completa de treinamento de modelos de deep learning para análise de ECG, oferecendo:

#### 🧠 Modelos de IA Avançados
- **HeartBEiT**: Vision Transformer adaptado para ECG
- **CNN-LSTM**: Arquitetura híbrida para análise temporal
- **SE-ResNet1D**: ResNet 1D com Squeeze-and-Excitation
- **ECG Transformer**: Transformer padrão adaptado para sinais ECG

#### 📊 Datasets Públicos Suportados
- **MIT-BIH Arrhythmia Database**: 48 registros com anotações de arritmias
- **PTB-XL ECG Database**: 21.837 ECGs de 12 derivações
- **CPSC2018**: 6.877 ECGs de hospitais chineses
- **MIMIC-ECG**: ~800k ECGs de pacientes de UTI
- **Icentia11k**: 11k pacientes com monitoramento contínuo

#### 🔧 Funcionalidades Principais
- **Pipeline Completo**: Pré-processamento, treinamento, avaliação e exportação
- **API REST**: Integração com o sistema principal
- **Configuração Flexível**: Configurações modulares para diferentes cenários
- **Monitoramento**: TensorBoard e Weights & Biases
- **Exportação**: Suporte para PyTorch, ONNX e TorchScript

### 📁 Nova Estrutura do Projeto

```
backend/training/
├── config/                 # Configurações
├── datasets/               # Implementação de datasets
├── models/                 # Arquiteturas de modelos
├── preprocessing/          # Pré-processamento
├── trainers/              # Treinadores
├── evaluation/            # Avaliação
├── utils/                 # Utilitários
├── scripts/               # Scripts utilitários
├── api.py                 # API REST
└── main.py                # Script principal
```

### 🚀 Como Usar

#### Instalação Rápida
```bash
# Executar script de configuração
./scripts/setup_training.sh

# Ou instalação manual
cd backend/training
pip install -r requirements.txt
```

#### Treinamento de Modelo
```bash
# Baixar dataset
python scripts/download_datasets.py --dataset ptbxl

# Treinar modelo
python main.py --model heartbeit --dataset ptbxl --epochs 50

# Exportar modelo
python scripts/export_model.py --model_name heartbeit --checkpoint_path checkpoints/heartbeit_best.pth --num_classes 5
```

#### API REST
```bash
# Iniciar API de treinamento
python api.py

# Endpoints disponíveis em http://localhost:8001
# - GET /datasets - Lista datasets
# - GET /models - Lista modelos
# - POST /training/start - Inicia treinamento
# - GET /training/{job_id}/status - Status do treinamento
```

### 🔗 Integração com o Sistema Principal

A plataforma se integra perfeitamente com o CardioAI Pro existente:

1. **API Unificada**: Endpoints de treinamento integrados à API principal
2. **Autenticação**: Usa o sistema de autenticação existente
3. **Modelos Compartilhados**: Modelos treinados podem ser usados para inferência
4. **Configurações**: Reutiliza configurações do sistema principal

### 📈 Benefícios

- **Personalização**: Treine modelos específicos para seus dados
- **Pesquisa**: Experimente com diferentes arquiteturas
- **Produção**: Exporte modelos para uso em produção
- **Escalabilidade**: Suporte para treinamento distribuído
- **Monitoramento**: Acompanhe o progresso em tempo real

### 🛠️ Requisitos Técnicos

- Python 3.8+
- PyTorch 1.9+
- CUDA (opcional, para GPU)
- 8GB+ RAM recomendado
- 50GB+ espaço em disco para datasets

### 📚 Documentação

Documentação completa disponível em:
- `backend/training/README.md` - Guia completo
- `backend/training/config/` - Configurações detalhadas
- Exemplos de uso nos scripts

### 🔄 Compatibilidade

- ✅ Totalmente compatível com o sistema existente
- ✅ Não afeta funcionalidades atuais
- ✅ Pode ser habilitado/desabilitado conforme necessário
- ✅ Integração opcional com sistemas externos

### 🎯 Próximos Passos

1. Execute o script de configuração
2. Baixe um dataset de teste
3. Treine seu primeiro modelo
4. Explore as diferentes arquiteturas
5. Integre com seu workflow existente

---

**Nota**: Esta é uma adição ao sistema existente. Todas as funcionalidades atuais do CardioAI Pro permanecem inalteradas e totalmente funcionais.

