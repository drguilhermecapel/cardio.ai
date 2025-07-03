# CardioAI Pro - Sistema Avançado de Análise de ECG

## 🚀 Visão Geral

O CardioAI Pro é um sistema completo de análise de eletrocardiograma (ECG) baseado em inteligência artificial, implementando uma arquitetura hierárquica multi-tarefa de última geração para interpretação automática de ECGs.

## 🏗️ Arquitetura do Sistema

### 1. Camada de Aquisição e Pré-processamento
- **Suporte a múltiplos formatos**: SCP-ECG, DICOM, HL7 aECG, CSV, TXT, NPY
- **Pipeline de pré-processamento**: Filtragem digital, remoção de artefatos, normalização
- **Taxa de amostragem**: 250-1000 Hz (idealmente 500 Hz)
- **Segmentação temporal**: Detecção automática de complexos QRS

### 2. Modelos de IA Hierárquicos
- **Nível 1**: CNNs 1D para detecção de características básicas (ondas P, QRS, T)
- **Nível 2**: RNNs (LSTM/GRU) para análise de ritmo e arritmias
- **Nível 3**: Transformers para diagnóstico integral e correlação entre derivações
- **Ensemble**: Combinação de múltiplos modelos para máxima precisão

### 3. Sistema de Explicabilidade
- **Grad-CAM**: Mapas de ativação para visualizar regiões importantes
- **SHAP**: Análise de contribuição de características
- **Feature Importance**: Análise de sensibilidade por perturbação
- **Relatórios automáticos**: Justificativas clínicas das decisões

### 4. Validação e Confiabilidade
- **Incerteza Bayesiana**: Quantificação de confiança nas predições
- **Detecção OOD**: Identificação de casos fora da distribuição de treinamento
- **Scores de confiança**: Métricas de qualidade das predições
- **Alertas automáticos**: Notificações para casos de baixa confiança

### 5. Integração e Interoperabilidade
- **FHIR R4**: Compatibilidade total com padrões de interoperabilidade
- **APIs RESTful**: Endpoints para integração com sistemas HIS/PACS
- **Auditoria completa**: Rastreamento de todas as operações
- **Segurança**: Criptografia AES-256 e TLS 1.3

## 🛠️ Tecnologias Utilizadas

### Backend
- **FastAPI**: Framework web moderno e rápido
- **Python 3.11+**: Linguagem principal
- **TensorFlow/Keras**: Modelos de deep learning
- **PyTorch**: Modelos alternativos e pesquisa
- **scikit-learn**: Modelos tradicionais de ML
- **NumPy/SciPy**: Computação científica
- **Pandas**: Manipulação de dados

### Processamento de Sinais
- **WFDB**: Leitura de formatos médicos
- **PyWavelets**: Análise wavelet
- **SciPy.signal**: Filtragem digital
- **NeuroKit2**: Processamento de sinais fisiológicos

### Explicabilidade
- **SHAP**: Explicações de modelos
- **LIME**: Interpretabilidade local
- **Grad-CAM**: Mapas de ativação

### Visualização
- **Matplotlib**: Gráficos científicos
- **Seaborn**: Visualizações estatísticas
- **Plotly**: Gráficos interativos

### Frontend
- **React**: Interface de usuário moderna
- **TypeScript**: Tipagem estática
- **Tailwind CSS**: Estilização
- **Recharts**: Gráficos para React

## 📦 Instalação

### Pré-requisitos
- Python 3.11 ou superior
- Node.js 18 ou superior
- Git

### Backend
```bash
# Clonar repositório
git clone https://github.com/drguilhermecapel/cardio.ai.git
cd cardio.ai

# Criar ambiente virtual
python -m venv cardioai_env
source cardioai_env/bin/activate  # Linux/Mac
# ou
cardioai_env\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Iniciar servidor
cd backend
python -m app.main
```

### Frontend
```bash
# Instalar dependências
cd frontend
npm install

# Iniciar desenvolvimento
npm run dev

# Build para produção
npm run build
```

## 🚀 Uso Rápido

### 1. Análise de ECG via API

```python
import requests
import numpy as np

# Dados de ECG (exemplo)
ecg_data = np.random.randn(1000).tolist()

# Requisição para análise
response = requests.post("http://localhost:8000/api/v1/ecg/analyze", json={
    "patient_id": "PATIENT_001",
    "ecg_data": ecg_data,
    "sampling_rate": 500,
    "leads": ["I"]
})

result = response.json()
print(f"Confiança: {result['confidence']:.3f}")
print(f"Predições: {result['predictions']}")
```

### 2. Upload de Arquivo ECG

```bash
curl -X POST "http://localhost:8000/api/v1/ecg/upload-file" \
     -F "patient_id=PATIENT_002" \
     -F "file=@ecg_data.csv" \
     -F "sampling_rate=500"
```

### 3. Explicabilidade

```python
# Gerar explicação para análise
response = requests.post("http://localhost:8000/api/v1/ecg/explain/analysis_id", json={
    "ecg_data": ecg_data,
    "model_name": "demo_ecg_classifier"
})

explanation = response.json()
# Visualizações em base64 disponíveis em explanation['explanation']['analyses']
```

## 📊 Recursos FHIR R4

O sistema é totalmente compatível com FHIR R4:

### Observações ECG
```json
{
  "resourceType": "Observation",
  "status": "final",
  "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "procedure"}]}],
  "code": {"coding": [{"system": "http://loinc.org", "code": "11524-6", "display": "EKG study"}]},
  "subject": {"reference": "Patient/PATIENT_001"},
  "valueQuantity": {"value": 0.95, "unit": "confidence_score"}
}
```

### Relatórios Diagnósticos
```json
{
  "resourceType": "DiagnosticReport",
  "status": "final",
  "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "CG"}]}],
  "code": {"coding": [{"system": "http://loinc.org", "code": "11524-6", "display": "EKG study"}]},
  "conclusion": "ECG analisado com alta confiança. Resultados dentro dos parâmetros esperados."
}
```

## 🧪 Testes

### Executar Testes Completos
```bash
python test_system.py
```

### Executar Testes Lite (sem dependências pesadas)
```bash
python test_system_lite.py
```

### Testes Unitários
```bash
pytest tests/
```

## 📈 Performance

- **Target AUC**: > 0.97
- **Tempo de inferência**: < 1s por ECG
- **Processamento em lote**: Suportado
- **Escalabilidade**: Horizontal via containers

## 🔒 Segurança e Compliance

- **Criptografia**: AES-256 para dados em repouso
- **Transmissão**: TLS 1.3
- **Compliance**: HIPAA, LGPD, ISO 13485
- **Auditoria**: Log completo de operações
- **Autenticação**: JWT com refresh tokens

## 📚 Documentação da API

Acesse a documentação interativa:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Equipe

- **Dr. Guilherme Capel** - Cardiologista e Desenvolvedor Principal
- **Equipe de IA** - Desenvolvimento de modelos de machine learning
- **Equipe de Software** - Desenvolvimento de sistema e infraestrutura

## 📞 Suporte

- **Email**: drguilhermecapel@gmail.com
- **Issues**: [GitHub Issues](https://github.com/drguilhermecapel/cardio.ai/issues)
- **Documentação**: [Wiki do Projeto](https://github.com/drguilhermecapel/cardio.ai/wiki)

## 🔄 Changelog

### v2.0.0 (2025-01-03)
- ✨ Arquitetura hierárquica multi-tarefa implementada
- ✨ Sistema de explicabilidade com Grad-CAM e SHAP
- ✨ Compatibilidade FHIR R4 completa
- ✨ Sistema de incerteza bayesiana
- ✨ APIs RESTful modernas
- ✨ Interface web responsiva
- 🔧 Pipeline de pré-processamento avançado
- 🔧 Sistema de auditoria e logging
- 🔧 Testes automatizados

### v1.0.0 (2024-06-30)
- 🎉 Versão inicial do sistema
- 📊 Modelos básicos de classificação ECG
- 🖥️ Interface web simples
- 📁 Suporte a formatos básicos de ECG

---

**CardioAI Pro** - Revolucionando a análise de ECG com inteligência artificial de última geração.

