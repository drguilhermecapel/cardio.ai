# Plataforma de Treinamento de IA - CardioAI Pro

## Visão Geral

Esta plataforma fornece um sistema completo de treinamento de modelos de deep learning para análise de ECG, integrado ao sistema CardioAI Pro existente.

## Características Principais

- **Múltiplas Arquiteturas**: HeartBEiT, CNN-LSTM, SE-ResNet1D, ECG Transformer
- **Datasets Públicos**: MIT-BIH, PTB-XL, CPSC2018, MIMIC-ECG, Icentia11k
- **Pipeline Completo**: Pré-processamento, treinamento, avaliação e exportação
- **API REST**: Integração com o sistema principal
- **Configuração Flexível**: Configurações modulares para diferentes cenários

## Estrutura do Projeto

```
backend/training/
├── config/                 # Configurações
│   ├── training_config.py  # Configuração principal
│   ├── model_configs.py    # Configurações de modelos
│   └── dataset_configs.py  # Configurações de datasets
├── datasets/               # Implementação de datasets
│   ├── base_dataset.py     # Classe base
│   ├── mitbih_dataset.py   # MIT-BIH Dataset
│   ├── ptbxl_dataset.py    # PTB-XL Dataset
│   └── dataset_factory.py  # Fábrica de datasets
├── models/                 # Arquiteturas de modelos
│   ├── base_model.py       # Classe base
│   ├── heartbeit.py        # HeartBEiT Transformer
│   ├── cnn_lstm.py         # CNN-LSTM híbrido
│   └── model_factory.py    # Fábrica de modelos
├── preprocessing/          # Pré-processamento
│   ├── filters.py          # Filtros de sinal
│   ├── normalization.py    # Normalização
│   └── augmentation.py     # Data augmentation
├── trainers/              # Treinadores
│   ├── base_trainer.py     # Classe base
│   └── classification_trainer.py
├── evaluation/            # Avaliação
│   ├── metrics.py          # Métricas
│   └── visualizations.py  # Visualizações
├── utils/                 # Utilitários
├── scripts/               # Scripts utilitários
│   ├── download_datasets.py
│   └── export_model.py
├── api.py                 # API REST
├── main.py                # Script principal
└── requirements.txt       # Dependências
```

## Instalação

1. Instale as dependências:
```bash
cd backend/training
pip install -r requirements.txt
```

2. Configure as variáveis de ambiente (opcional):
```bash
export TRAINING_BATCH_SIZE=32
export TRAINING_EPOCHS=100
export TRAINING_LEARNING_RATE=1e-4
```

## Uso Básico

### 1. Download de Datasets

```bash
python scripts/download_datasets.py --dataset ptbxl
```

### 2. Treinamento de Modelo

```bash
python main.py --model heartbeit --dataset ptbxl --epochs 50 --batch_size 32
```

### 3. Exportação de Modelo

```bash
python scripts/export_model.py --model_name heartbeit --checkpoint_path checkpoints/heartbeit_best.pth --num_classes 5 --format onnx
```

### 4. API REST

```bash
python api.py
```

A API estará disponível em `http://localhost:8001`

## Endpoints da API

- `GET /datasets` - Lista datasets disponíveis
- `GET /models` - Lista modelos disponíveis
- `POST /training/start` - Inicia treinamento
- `GET /training/{job_id}/status` - Status do treinamento
- `POST /models/export` - Exporta modelo treinado

## Configuração

### Configuração de Treinamento

Edite `config/training_config.py` para ajustar:
- Hiperparâmetros de treinamento
- Caminhos de dados e checkpoints
- Configurações de hardware

### Configuração de Modelos

Edite `config/model_configs.py` para:
- Adicionar novos modelos
- Modificar arquiteturas existentes
- Ajustar hiperparâmetros específicos

### Configuração de Datasets

Edite `config/dataset_configs.py` para:
- Adicionar novos datasets
- Modificar metadados
- Configurar downloads

## Modelos Suportados

### HeartBEiT
- Vision Transformer adaptado para ECG
- Baseado em BEiT (Baidu's Enhanced Image Transformer)
- Ideal para capturar dependências de longo prazo

### CNN-LSTM
- Arquitetura híbrida
- CNN para extração de características
- LSTM para modelagem temporal

### SE-ResNet1D
- ResNet 1D com Squeeze-and-Excitation
- Eficiente para sinais temporais
- Boa performance com menos parâmetros

### ECG Transformer
- Transformer padrão adaptado para ECG
- Attention mechanism para análise temporal
- Flexível e interpretável

## Datasets Suportados

### MIT-BIH Arrhythmia Database
- 48 registros de 30 minutos
- Anotações de arritmias
- Padrão para avaliação de algoritmos

### PTB-XL ECG Database
- 21.837 ECGs de 12 derivações
- 71 diagnósticos diferentes
- 5 superclasses principais

### CPSC2018
- 6.877 ECGs de hospitais chineses
- 9 classes de diagnóstico
- Dados de competição

## Integração com o Sistema Principal

A plataforma se integra ao CardioAI Pro através de:

1. **API REST**: Endpoints para treinamento e gerenciamento
2. **Modelos Exportados**: Compatíveis com o sistema de inferência
3. **Configurações Compartilhadas**: Reutilização de configurações existentes

## Monitoramento

- **TensorBoard**: Visualização de métricas de treinamento
- **Weights & Biases**: Experimentos e comparações (opcional)
- **Logs**: Sistema de logging detalhado

## Exemplo de Uso Completo

```python
from backend.training.datasets.dataset_factory import DatasetFactory
from backend.training.models.model_factory import ModelFactory
from backend.training.trainers.classification_trainer import ClassificationTrainer

# 1. Carregar dataset
dataset = DatasetFactory.create_dataset("ptbxl", "/path/to/data")

# 2. Criar modelo
model = ModelFactory.create_model("heartbeit", num_classes=5, input_channels=12)

# 3. Configurar treinamento
trainer = ClassificationTrainer(model, train_loader, val_loader, optimizer, criterion)

# 4. Treinar
trainer.train()

# 5. Avaliar
metrics = trainer.evaluate(test_loader)
```

## Contribuição

Para adicionar novos modelos ou datasets:

1. Implemente a classe base correspondente
2. Adicione configurações apropriadas
3. Registre na fábrica correspondente
4. Adicione testes unitários

## Licença

Este projeto está licenciado sob os mesmos termos do CardioAI Pro principal.

