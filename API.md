# Documentação da API CardioAI Pro

## Visão Geral

A API CardioAI Pro fornece endpoints para análise de ECG usando modelos de inteligência artificial. A API é construída com FastAPI e segue os princípios RESTful.

## Base URL

```
http://localhost:8000
```

## Endpoints

### Informações Gerais

#### `GET /`

Retorna informações básicas sobre o sistema.

**Resposta**:
```json
{
  "name": "CardioAI Pro",
  "version": "2.0.0",
  "description": "Sistema Avançado de Análise de ECG com IA",
  "status": "running",
  "features": [
    "Análise automática de ECG",
    "Modelos ensemble de deep learning",
    "Explicabilidade com Grad-CAM e SHAP",
    "Compatibilidade FHIR R4",
    "Sistema de incerteza bayesiana",
    "Auditoria completa",
    "APIs RESTful",
    "Interface web responsiva"
  ],
  "endpoints": {
    "docs": "/docs",
    "redoc": "/redoc",
    "health": "/health",
    "api": "/api/v1",
    "ecg_analysis": "/api/v1/ecg/analyze",
    "models": "/api/v1/models",
    "fhir": "/api/v1/fhir"
  }
}
```

#### `GET /health`

Verifica o status de saúde do sistema.

**Resposta**:
```json
{
  "status": "healthy",
  "service": "CardioAI Pro",
  "version": "2.0.0",
  "timestamp": "2025-07-04T14:30:00.000Z",
  "services": {
    "model_service": "running",
    "ecg_service": "running",
    "api_service": "running"
  },
  "models_loaded": 1,
  "system_info": {
    "python_version": "3.11+",
    "tensorflow": "available",
    "pytorch": "available",
    "fastapi": "available"
  }
}
```

#### `GET /info`

Retorna informações detalhadas sobre o sistema.

**Resposta**:
```json
{
  "system": {
    "name": "CardioAI Pro",
    "version": "2.0.0",
    "description": "Sistema Avançado de Análise de ECG",
    "architecture": "Hierárquica Multi-tarefa"
  },
  "capabilities": {
    "ecg_formats": ["SCP-ECG", "DICOM", "HL7 aECG", "CSV", "TXT", "NPY", "EDF", "WFDB"],
    "sampling_rates": ["250-1000 Hz"],
    "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1-V6"],
    "ai_models": ["CNN-1D", "LSTM", "GRU", "Transformers", "Ensemble"],
    "explainability": ["Grad-CAM", "SHAP", "Feature Importance"],
    "standards": ["FHIR R4", "HL7", "DICOM"]
  },
  "models": {
    "loaded": ["demo_ecg_classifier"],
    "details": {
      "demo_ecg_classifier": {
        "type": "sklearn_demo",
        "input_shape": [5000],
        "output_classes": 10,
        "created": "2025-07-04T14:30:00.000Z",
        "description": "Modelo de demonstração para análise de ECG"
      }
    },
    "total": 1
  },
  "performance": {
    "target_auc": "> 0.97",
    "inference_time": "< 1s",
    "batch_processing": "supported"
  }
}
```

### API v1

#### `GET /api/v1/health`

Verifica o status de saúde da API v1.

**Resposta**:
```json
{
  "status": "healthy",
  "api_version": "v1",
  "endpoints": [
    "/ecg/upload",
    "/ecg/analyze/{process_id}",
    "/models",
    "/models/{model_name}"
  ]
}
```

#### `POST /api/v1/ecg/upload`

Faz upload e processa um arquivo de ECG.

**Parâmetros**:
- `file`: Arquivo de ECG (CSV, TXT, NPY, DAT, EDF, JSON)

**Resposta**:
```json
{
  "process_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_name": "ecg_sample.csv",
  "file_format": "csv",
  "processed_data": {
    "raw_data": [...],
    "filtered_data": [...],
    "normalized_data": [...],
    "features": {
      "mean": 0.0,
      "std": 1.0,
      "min": -2.5,
      "max": 2.5,
      "range": 5.0,
      "heart_rate": 75.0,
      "rr_intervals": [...],
      "rr_std": 0.05,
      "peak_count": 10,
      "peak_positions": [...]
    }
  },
  "metadata": {
    "sampling_rate": 500,
    "leads": ["II"],
    "units": "mV",
    "file_format": "csv_single_column",
    "patient_data": {}
  },
  "processing_timestamp": "2025-07-04T14:30:00.000Z"
}
```

#### `POST /api/v1/ecg/analyze/{process_id}`

Analisa um ECG previamente processado usando modelo de IA.

**Parâmetros**:
- `process_id`: ID do processamento prévio (path)
- `model_name`: Nome do modelo a usar (query, opcional)

**Resposta**:
```json
{
  "process_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_used": "demo_ecg_classifier",
  "prediction": {
    "predicted_class": 0,
    "diagnosis": "Normal",
    "confidence": 0.92,
    "confidence_level": "muito_alta",
    "probabilities": {
      "Normal": 0.92,
      "Fibrilação Atrial": 0.03,
      "Bradicardia": 0.01,
      "Taquicardia": 0.01,
      "Arritmia Ventricular": 0.01,
      "Bloqueio AV": 0.005,
      "Isquemia": 0.005,
      "Infarto do Miocárdio": 0.005,
      "Hipertrofia Ventricular": 0.005,
      "Anormalidade Inespecífica": 0.01
    },
    "recommendations": {
      "clinical_review_required": false,
      "follow_up": ["Acompanhamento de rotina"],
      "clinical_notes": ["ECG dentro dos padrões normais"],
      "urgent_attention": false
    },
    "model_metadata": {
      "type": "sklearn_demo",
      "input_shape": [5000],
      "output_classes": 10,
      "created": "2025-07-04T14:30:00.000Z",
      "description": "Modelo de demonstração para análise de ECG"
    },
    "analysis_timestamp": "2025-07-04T14:30:00.000Z",
    "visualization": {
      "ecg_plot": "data:image/png;base64,...",
      "probability_plot": "data:image/png;base64,..."
    }
  },
  "analysis_timestamp": "2025-07-04T14:30:00.000Z"
}
```

#### `GET /api/v1/models`

Lista todos os modelos disponíveis.

**Resposta**:
```json
{
  "models": ["demo_ecg_classifier"],
  "count": 1,
  "timestamp": "2025-07-04T14:30:00.000Z"
}
```

#### `GET /api/v1/models/{model_name}`

Obtém informações detalhadas sobre um modelo específico.

**Parâmetros**:
- `model_name`: Nome do modelo (path)

**Resposta**:
```json
{
  "type": "sklearn_demo",
  "input_shape": [5000],
  "output_classes": 10,
  "created": "2025-07-04T14:30:00.000Z",
  "description": "Modelo de demonstração para análise de ECG"
}
```

## Códigos de Status

- `200 OK`: Requisição bem-sucedida
- `400 Bad Request`: Parâmetros inválidos
- `404 Not Found`: Recurso não encontrado
- `500 Internal Server Error`: Erro interno do servidor

## Autenticação

A API atualmente não requer autenticação para fins de desenvolvimento. Em ambiente de produção, será implementada autenticação OAuth2 ou JWT.

## Limitações de Taxa

Não há limitações de taxa implementadas atualmente. Em ambiente de produção, serão implementadas limitações adequadas.

## Exemplos de Uso

### Upload e Análise de ECG

```python
import requests
import json

# Upload de arquivo ECG
with open('ecg_sample.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/ecg/upload',
        files={'file': ('ecg_sample.csv', f, 'text/csv')}
    )

process_data = response.json()
process_id = process_data['process_id']

# Análise do ECG
response = requests.post(
    f'http://localhost:8000/api/v1/ecg/analyze/{process_id}'
)

analysis_result = response.json()
print(f"Diagnóstico: {analysis_result['prediction']['diagnosis']}")
print(f"Confiança: {analysis_result['prediction']['confidence']}")
```

### Listar Modelos Disponíveis

```python
import requests

response = requests.get('http://localhost:8000/api/v1/models')
models = response.json()['models']

print(f"Modelos disponíveis: {models}")
```