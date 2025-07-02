#!/bin/bash
# Script automático para aplicar todas as correções do CardioAI Pro

echo "🚀 APLICANDO TODAS AS CORREÇÕES DO CARDIOAI PRO"
echo "=============================================="
echo ""

# Verificar se estamos no diretório correto
if [ ! -d "backend" ]; then
    if [ -d "../backend" ]; then
        cd ..
    else
        echo "❌ Erro: Diretório 'backend' não encontrado!"
        echo "   Execute este script na raiz do projeto ou dentro do diretório backend."
        exit 1
    fi
fi

cd backend

# Função para executar script Python e verificar sucesso
run_python_script() {
    local script_name=$1
    local description=$2
    
    echo ""
    echo "📌 $description"
    echo "-------------------------------------------"
    
    if [ -f "$script_name" ]; then
        python "$script_name"
        if [ $? -eq 0 ]; then
            echo "✅ Sucesso: $description"
            return 0
        else
            echo "❌ Falha: $description"
            return 1
        fi
    else
        echo "⚠️  Script não encontrado: $script_name"
        echo "   Criando o script..."
        return 2
    fi
}

# 1. Criar os scripts de correção se não existirem
echo "📝 Verificando scripts de correção..."

# Criar script principal se não existir
if [ ! -f "cardioai-fix-script.py" ]; then
    echo "   Criando cardioai-fix-script.py..."
    # Aqui você deve copiar o conteúdo do script principal
    echo "   ⚠️  Por favor, copie o conteúdo do script principal para cardioai-fix-script.py"
fi

# 2. Instalar dependências necessárias
echo ""
echo "📦 Instalando dependências..."
pip install pytest pytest-asyncio pytest-cov pytest-mock aiosqlite httpx sqlalchemy

# 3. Executar correções na ordem correta
echo ""
echo "🔧 APLICANDO CORREÇÕES"
echo "====================="

# Script principal
run_python_script "cardioai-fix-script.py" "Correções principais"
MAIN_RESULT=$?

# Correção de ValidationException
run_python_script "fix-validation-exception.py" "Correção ValidationException"
VALIDATION_RESULT=$?

# Correções adicionais
run_python_script "fix-additional-issues.py" "Correções adicionais"
ADDITIONAL_RESULT=$?

# 4. Limpar cache do pytest
echo ""
echo "🧹 Limpando cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null
echo "✅ Cache limpo"

# 5. Executar testes
echo ""
echo "🧪 EXECUTANDO TESTES"
echo "==================="

# Testes críticos primeiro
echo ""
echo "🎯 Executando testes críticos..."
pytest tests/test_ecg_service_critical_coverage.py -v --tb=short
CRITICAL_RESULT=$?

if [ $CRITICAL_RESULT -eq 0 ]; then
    echo "✅ Testes críticos passaram!"
else
    echo "❌ Testes críticos falharam"
fi

# Todos os testes com cobertura
echo ""
echo "📊 Executando todos os testes com cobertura..."
pytest --cov=app --cov-report=term-missing --cov-report=html --cov-report=json -q

# Extrair porcentagem de cobertura
if [ -f "coverage.json" ]; then
    COVERAGE=$(python -c "import json; data=json.load(open('coverage.json')); print(f\"{data['totals']['percent_covered']:.1f}\")" 2>/dev/null)
    
    if [ ! -z "$COVERAGE" ]; then
        echo ""
        echo "📊 COBERTURA TOTAL: ${COVERAGE}%"
        
        # Verificar se atingiu a meta
        if (( $(echo "$COVERAGE >= 80" | bc -l) )); then
            echo "✅ META DE 80% ALCANÇADA!"
        else
            echo "⚠️  Cobertura abaixo de 80%"
        fi
    fi
fi

# 6. Relatório final
echo ""
echo "📋 RELATÓRIO FINAL"
echo "=================="
echo ""

if [ $MAIN_RESULT -eq 0 ] && [ $CRITICAL_RESULT -eq 0 ]; then
    echo "✅ Correções aplicadas com sucesso!"
    echo "✅ Testes críticos passando!"
    
    if [ ! -z "$COVERAGE" ] && (( $(echo "$COVERAGE >= 80" | bc -l) )); then
        echo "✅ Cobertura de código adequada!"
        echo ""
        echo "🎉 SUCESSO COMPLETO! O CardioAI Pro está pronto!"
    else
        echo "⚠️  Cobertura ainda precisa melhorar"
        echo ""
        echo "📌 Próximos passos:"
        echo "   1. Abra htmlcov/index.html no navegador"
        echo "   2. Identifique arquivos com baixa cobertura"
        echo "   3. Adicione testes para as linhas não cobertas"
    fi
else
    echo "❌ Algumas correções ou testes falharam"
    echo ""
    echo "📌 Ações recomendadas:"
    echo "   1. Verifique os logs de erro acima"
    echo "   2. Execute testes específicos com -vv para mais detalhes"
    echo "   3. Use pytest --pdb para debug interativo"
fi

echo ""
echo "📂 Arquivos gerados:"
echo "   - htmlcov/index.html (relatório de cobertura visual)"
echo "   - coverage.json (dados de cobertura)"
echo "   - .coverage (banco de dados de cobertura)"

echo ""
echo "=============================================="
echo "Script concluído!"
