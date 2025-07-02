# 📊 Guia de Cobertura de Testes - CardioAI Pro

## 🎯 Objetivos de Cobertura

### Requisitos Gerais
- **Cobertura Global**: ≥ 80% (ANVISA/FDA)
- **Componentes Críticos Médicos**: 100%
- **Componentes Médicos Gerais**: ≥ 95%

### Status Atual
- ✅ **Backend**: 82.78% (Meta atingida)
- ❌ **ML Model Service**: 79% (Precisa > 80%)
- ⚠️ **Frontend**: Não medido (Requer configuração)

## 🚀 Como Executar Testes com Cobertura

### Backend

```bash
# Navegar para o diretório backend
cd backend

# Executar todos os testes com cobertura
poetry run pytest --cov=app --cov-report=html --cov-report=term --cov-fail-under=80

# Executar apenas testes do ML Model Service
poetry run pytest tests/test_ml_model_service_coverage.py --cov=app.services.ml_model_service --cov-fail-under=80

# Executar testes de componentes críticos (100% requerido)
poetry run pytest tests/test_medical_critical_components.py \
  --cov=app.services.ecg \
  --cov=app.services.diagnosis \
  --cov-fail-under=100

# Visualizar relatório HTML
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Frontend

```bash
# Navegar para o diretório frontend
cd frontend

# Instalar dependências
npm install

# Executar todos os testes com cobertura
npm run test -- --coverage --watchAll=false

# Executar apenas componentes médicos críticos
npm run test -- --coverage --watchAll=false \
  --testPathPattern="medical|ecg|diagnosis" \
  --coverageThreshold='{"global":{"branches":100,"functions":100,"lines":100,"statements":100}}'

# Visualizar relatório HTML
open coverage/lcov-report/index.html  # macOS
xdg-open coverage/lcov-report/index.html  # Linux
start coverage/lcov-report/index.html  # Windows
```

## 📋 Componentes Críticos (100% Cobertura Obrigatória)

### Backend
| Arquivo | Descrição | Status |
|---------|-----------|--------|
| `app/services/ecg/analysis.py` | Análise de sinais ECG | ⏳ |
| `app/services/ecg/signal_quality.py` | Qualidade do sinal | ⏳ |
| `app/services/diagnosis/engine.py` | Motor de diagnóstico | ⏳ |
| `app/utils/medical/validation.py` | Validação médica | ⏳ |
| `app/core/medical_safety.py` | Segurança médica | ⏳ |

### Frontend
| Arquivo | Descrição | Status |
|---------|-----------|--------|
| `src/components/medical/ECGVisualization.tsx` | Visualização ECG | ⏳ |
| `src/components/medical/DiagnosisDisplay.tsx` | Display diagnóstico | ⏳ |
| `src/services/ecg/analysis.ts` | Análise ECG | ⏳ |
| `src/services/diagnosis/engine.ts` | Motor diagnóstico | ⏳ |
| `src/utils/medical/validation.ts` | Validação médica | ⏳ |

## 🔧 Configuração de CI/CD

### GitHub Actions
O pipeline automatizado executa em:
- Push para `main` ou `develop`
- Pull Requests

Arquivo: `.github/workflows/test-coverage.yml`

### Verificação Local
```bash
# Verificar compliance antes do commit
python scripts/check_compliance_thresholds.py \
  --backend-coverage backend/coverage.xml \
  --frontend-coverage frontend/coverage/lcov.info \
  --output compliance-report.json
```

## 📈 Melhorando a Cobertura

### 1. Identificar Áreas Descobertas
```bash
# Backend - gerar relatório detalhado
cd backend
poetry run pytest --cov=app --cov-report=html
# Abrir htmlcov/index.html e procurar por linhas vermelhas

# Frontend - gerar relatório detalhado
cd frontend
npm run test -- --coverage --watchAll=false
# Abrir coverage/lcov-report/index.html
```

### 2. Priorizar por Criticidade
1. **Componentes Críticos** (100% obrigatório)
2. **Componentes Médicos** (≥ 95%)
3. **Serviços Core** (≥ 90%)
4. **Utilitários** (≥ 80%)

### 3. Técnicas para Aumentar Cobertura

#### Testes de Casos Extremos
```python
# Exemplo: Testar valores limites
def test_heart_rate_boundaries():
    assert analyze_heart_rate(0) == "bradycardia_severe"
    assert analyze_heart_rate(40) == "bradycardia"
    assert analyze_heart_rate(60) == "normal"
    assert analyze_heart_rate(100) == "normal"
    assert analyze_heart_rate(101) == "tachycardia"
    assert analyze_heart_rate(250) == "tachycardia_severe"
```

#### Testes de Exceções
```python
# Exemplo: Testar todos os caminhos de erro
def test_ecg_analysis_exceptions():
    with pytest.raises(InvalidSignalException):
        analyze_ecg(None)
    
    with pytest.raises(SignalTooShortException):
        analyze_ecg([1, 2, 3])
    
    with pytest.raises(SamplingRateException):
        analyze_ecg(valid_signal, sampling_rate=0)
```

#### Mocking Efetivo
```typescript
// Exemplo: Mock de dependências externas
jest.mock('@/services/api', () => ({
  analyzeECG: jest.fn().mockResolvedValue({
    diagnosis: 'normal_sinus_rhythm',
    confidence: 0.95
  })
}));
```

## 🏥 Compliance Regulatório

### ANVISA RDC 40/2015
- Software Classe II requer cobertura mínima de 80%
- Componentes críticos devem ter rastreabilidade completa
- Documentação de validação obrigatória

### FDA 21 CFR 820.30
- Design controls exigem verificação adequada
- Testes devem cobrir todos os requisitos funcionais
- Evidência de teste para cada funcionalidade crítica

### Relatórios de Compliance
```bash
# Gerar relatório ANVISA
python scripts/generate_anvisa_report.py

# Gerar relatório FDA
python scripts/generate_fda_report.py

# Relatório consolidado
make compliance-report
```

## 🚨 Troubleshooting

### Problema: Testes Falhando no CI mas Passando Localmente
**Solução**: Verificar variáveis de ambiente e configurações do banco de dados de teste

```bash
# Reproduzir ambiente do CI localmente
export DATABASE_URL=postgresql://cardioai:testpass@localhost:5432/cardioai_test
export ENVIRONMENT=test
poetry run pytest
```

### Problema: Cobertura Caindo Após Merge
**Solução**: Configurar pre-commit hooks

```bash
# Instalar pre-commit
pip install pre-commit

# Configurar hooks
pre-commit install

# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: test-coverage
        name: Check test coverage
        entry: make test-coverage-check
        language: system
        pass_filenames: false
```

### Problema: Componentes Não Aparecem no Relatório
**Solução**: Verificar configuração de cobertura

```javascript
// jest.config.js - Verificar collectCoverageFrom
collectCoverageFrom: [
  'src/**/*.{ts,tsx}',
  '!src/**/*.d.ts',
  '!src/**/*.stories.tsx'
]
```

## 📚 Recursos Adicionais

- [Jest Coverage Documentation](https://jestjs.io/docs/configuration#collectcoverage-boolean)
- [Pytest Coverage Plugin](https://pytest-cov.readthedocs.io/)
- [ANVISA - Software como Dispositivo Médico](https://www.gov.br/anvisa/pt-br/assuntos/regulamentacao/legislacao/resolucoes-da-diretoria-colegiada)
- [FDA - Medical Device Software Validation](https://www.fda.gov/medical-devices/software-medical-device-samd)

## ✅ Checklist de Revisão

Antes de fazer merge na branch principal:

- [ ] Cobertura global ≥ 80%
- [ ] Componentes críticos = 100%
- [ ] Componentes médicos ≥ 95%
- [ ] Todos os testes passando
- [ ] CI/CD verde
- [ ] Relatório de compliance gerado
- [ ] Code review aprovado
- [ ] Documentação atualizada

---

**Última atualização**: Junho 2025  
**Responsável**: Equipe de Qualidade CardioAI Pro
