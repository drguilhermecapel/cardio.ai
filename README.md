# CardioAI Pro - Interpretador de ECG com Inteligência Artificial

Sistema avançado para análise e interpretação automática de eletrocardiogramas usando inteligência artificial.

## Características Principais

- **Análise Automática de ECG**: Processamento e interpretação de sinais de ECG
- **Detecção de Arritmias**: Identificação automática de irregularidades no ritmo cardíaco
- **Interpretação com IA**: Análise inteligente com relatórios detalhados
- **API REST Completa**: Interface de programação para integração
- **Processamento de Sinais**: Filtros avançados e pré-processamento
- **Detecção de Anormalidades**: Identificação de padrões anômalos

## Tecnologias Utilizadas

- **Backend**: FastAPI, Python 3.11+
- **Machine Learning**: PyTorch, NumPy, SciPy
- **Processamento de Sinais**: SciPy, filtros digitais
- **API**: FastAPI, Pydantic, Uvicorn
- **Banco de Dados**: SQLAlchemy, PostgreSQL

## Instalação

### Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes Python)

### Passos de Instalação

1. Clone o repositório:
```bash
git clone https://github.com/drguilhermecapel/cardio.ai.git
cd cardio.ai
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
cd backend
python -m app.main
```

A aplicação estará disponível em `http://localhost:8000`

## Uso da API

### Endpoints Principais

#### 1. Análise de ECG Simulado
```http
POST /ecg/analyze
Content-Type: application/json

{
    "patient_id": "12345",
    "patient_name": "João Silva",
    "patient_age": 45,
    "sampling_rate": 500
}
```

#### 2. Upload e Análise de Arquivo
```http
POST /ecg/analyze-file
Content-Type: multipart/form-data

file: [arquivo ECG em formato JSON ou CSV]
patient_id: "12345"
patient_name: "João Silva"
patient_age: 45
sampling_rate: 500
```

#### 3. Análise de Exemplo
```http
GET /ecg/sample-analysis
```

#### 4. Status do Interpretador
```http
GET /ecg/status
```

### Exemplo de Resposta

```json
{
    "analysis_id": "ECG_20250102_143022_1",
    "timestamp": "2025-01-02T14:30:22.123456",
    "patient_info": {
        "patient_id": "12345",
        "patient_name": "João Silva",
        "patient_age": 45
    },
    "signal_quality": "Boa",
    "rhythm_analysis": {
        "rhythm": "Ritmo sinusal normal",
        "regularity": "Regular",
        "heart_rate": 75.2,
        "rr_variability": 0.045
    },
    "r_peaks_count": 12,
    "abnormalities": [],
    "confidence_score": 0.85,
    "interpretation": "Frequência cardíaca: 75.2 bpm\nRitmo: Ritmo sinusal normal\nNenhuma anormalidade significativa detectada\nCONCLUSÃO: ECG dentro dos parâmetros normais"
}
```

## Funcionalidades do Interpretador

### 1. Pré-processamento de Sinais
- Normalização do sinal
- Filtro passa-banda (0.5-40 Hz)
- Remoção de baseline drift
- Redução de ruído

### 2. Detecção de Características
- Detecção automática de picos R
- Cálculo de intervalos RR
- Análise de variabilidade do ritmo
- Medição de amplitudes

### 3. Análise Clínica
- Cálculo da frequência cardíaca
- Classificação do ritmo cardíaco
- Detecção de arritmias
- Identificação de anormalidades

### 4. Interpretação Automática
- Geração de relatórios textuais
- Classificação de severidade
- Recomendações clínicas
- Score de confiança

## Estrutura do Projeto

```
cardio.ai/
├── backend/
│   └── app/
│       ├── api/
│       │   ├── __init__.py
│       │   └── ecg_api.py
│       ├── services/
│       │   ├── __init__.py
│       │   └── ecg_interpreter.py
│       ├── schemas/
│       ├── models/
│       ├── __init__.py
│       └── main.py
├── requirements.txt
└── README.md
```

## Desenvolvimento

### Executar em Modo de Desenvolvimento

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testes

```bash
pytest
```

### Documentação da API

Acesse `http://localhost:8000/docs` para ver a documentação interativa da API (Swagger UI).

## Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contato

Dr. Guilherme Capel - drguilhermecapel@gmail.com

Link do Projeto: [https://github.com/drguilhermecapel/cardio.ai](https://github.com/drguilhermecapel/cardio.ai)

## Aviso Médico

⚠️ **IMPORTANTE**: Este sistema é destinado apenas para fins educacionais e de pesquisa. Não deve ser usado para diagnóstico médico real sem supervisão de profissionais qualificados. Sempre consulte um cardiologista para interpretação clínica de ECGs.

