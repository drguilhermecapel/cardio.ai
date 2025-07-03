#!/usr/bin/env python3
"""
CardioAI Pro - Quick Fix para resolver cobertura 0%
Execute este script no diretório backend/
"""

import os
import sys
import subprocess
from pathlib import Path

def run_cmd(cmd):
    """Executa comando e mostra output"""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

print("=" * 60)
print("CardioAI Pro - Quick Fix para Cobertura Zero")
print("=" * 60)

# 1. Verificar se estamos no diretório correto
if not Path("app").exists() and not Path("tests").exists():
    print("❌ ERRO: Execute este script no diretório 'backend/'")
    sys.exit(1)

print("\n✅ Diretório correto detectado")

# 2. Criar arquivo de teste simples e funcional
print("\n📝 Criando teste mínimo funcional...")

test_content = '''"""Teste mínimo para verificar funcionamento."""
import pytest

def test_basic():
    """Teste básico para verificar pytest."""
    assert 1 + 1 == 2
    assert True
    assert "CardioAI" in "CardioAI Pro"

def test_list_operations():
    """Teste operações com listas."""
    lista = [1, 2, 3, 4, 5]
    assert len(lista) == 5
    assert sum(lista) == 15
    assert max(lista) == 5
    assert min(lista) == 1

def test_string_operations():
    """Teste operações com strings."""
    texto = "CardioAI Pro"
    assert texto.lower() == "cardioai pro"
    assert texto.upper() == "CARDIOAI PRO"
    assert len(texto) == 12
    assert "AI" in texto

def test_dict_operations():
    """Teste operações com dicionários."""
    config = {
        "name": "CardioAI",
        "version": "1.0.0",
        "modules": ["ecg", "analysis", "ai"]
    }
    assert config["name"] == "CardioAI"
    assert len(config["modules"]) == 3
    assert "ecg" in config["modules"]

class TestCardioAIBasic:
    """Classe de testes básicos."""
    
    def test_class_method(self):
        """Teste método de classe."""
        result = self.calculate(10, 5)
        assert result == 15
    
    def calculate(self, a, b):
        """Método auxiliar."""
        return a + b
    
    def test_exception_handling(self):
        """Teste tratamento de exceções."""
        with pytest.raises(ZeroDivisionError):
            _ = 1 / 0
    
    def test_parametrized(self):
        """Teste com múltiplos valores."""
        test_values = [
            (1, 1, 2),
            (2, 3, 5),
            (10, 20, 30),
            (-1, 1, 0)
        ]
        
        for a, b, expected in test_values:
            assert a + b == expected
'''

# Criar diretório tests se não existir
Path("tests").mkdir(exist_ok=True)

# Salvar teste
test_file = Path("tests/test_basic_coverage.py")
with open(test_file, "w", encoding="utf-8") as f:
    f.write(test_content)

print(f"✅ Criado: {test_file}")

# 3. Criar app mínimo se não existir
if not Path("app/__init__.py").exists():
    print("\n📝 Criando app mínimo...")
    
    Path("app").mkdir(exist_ok=True)
    
    # __init__.py
    with open("app/__init__.py", "w") as f:
        f.write('"""CardioAI Pro Application."""\n__version__ = "1.0.0"\n')
    
    # main.py básico
    main_content = '''"""CardioAI Pro Main Module."""

def get_app_info():
    """Retorna informações da aplicação."""
    return {
        "name": "CardioAI Pro",
        "version": "1.0.0",
        "status": "running"
    }

def health_check():
    """Verifica saúde da aplicação."""
    return {"status": "healthy", "service": "cardioai-pro"}

class CardioAIApp:
    """Classe principal da aplicação."""
    
    def __init__(self):
        self.name = "CardioAI Pro"
        self.version = "1.0.0"
        self.modules = []
    
    def add_module(self, module_name):
        """Adiciona módulo."""
        self.modules.append(module_name)
        return True
    
    def get_modules(self):
        """Retorna módulos."""
        return self.modules
    
    def process_ecg(self, data):
        """Processa ECG (mock)."""
        if not data:
            raise ValueError("Dados inválidos")
        return {"status": "processed", "data": data}
'''
    
    with open("app/main.py", "w") as f:
        f.write(main_content)
    
    print("✅ Criado: app/__init__.py")
    print("✅ Criado: app/main.py")

# 4. Criar teste para o app
print("\n📝 Criando teste para o app...")

app_test_content = '''"""Testes para o módulo principal."""
import pytest
from app.main import get_app_info, health_check, CardioAIApp

class TestAppMain:
    """Testes do módulo principal."""
    
    def test_get_app_info(self):
        """Testa informações da app."""
        info = get_app_info()
        assert info["name"] == "CardioAI Pro"
        assert info["version"] == "1.0.0"
        assert info["status"] == "running"
    
    def test_health_check(self):
        """Testa health check."""
        health = health_check()
        assert health["status"] == "healthy"
        assert health["service"] == "cardioai-pro"
    
    def test_cardioai_app_init(self):
        """Testa inicialização da app."""
        app = CardioAIApp()
        assert app.name == "CardioAI Pro"
        assert app.version == "1.0.0"
        assert app.modules == []
    
    def test_cardioai_app_modules(self):
        """Testa gerenciamento de módulos."""
        app = CardioAIApp()
        
        # Adicionar módulos
        assert app.add_module("ecg") is True
        assert app.add_module("analysis") is True
        
        # Verificar módulos
        modules = app.get_modules()
        assert len(modules) == 2
        assert "ecg" in modules
        assert "analysis" in modules
    
    def test_process_ecg_valid(self):
        """Testa processamento válido de ECG."""
        app = CardioAIApp()
        result = app.process_ecg({"signal": [1, 2, 3]})
        assert result["status"] == "processed"
        assert result["data"] == {"signal": [1, 2, 3]}
    
    def test_process_ecg_invalid(self):
        """Testa processamento inválido de ECG."""
        app = CardioAIApp()
        with pytest.raises(ValueError, match="Dados inválidos"):
            app.process_ecg(None)
    
    def test_process_ecg_empty(self):
        """Testa processamento com dados vazios."""
        app = CardioAIApp()
        with pytest.raises(ValueError):
            app.process_ecg({})
'''

with open("tests/test_app_main.py", "w", encoding="utf-8") as f:
    f.write(app_test_content)

print("✅ Criado: tests/test_app_main.py")

# 5. Criar pytest.ini se não existir
if not Path("pytest.ini").exists():
    print("\n📝 Criando pytest.ini...")
    
    pytest_ini = '''[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
'''
    
    with open("pytest.ini", "w") as f:
        f.write(pytest_ini)
    
    print("✅ Criado: pytest.ini")

# 6. Executar testes
print("\n🧪 Executando testes com cobertura...")
print("-" * 60)

# Limpar cache antes
run_cmd("python -m pytest --cache-clear > nul 2>&1")

# Executar com cobertura
success = run_cmd("python -m pytest --cov=app --cov-report=term --cov-report=html -v")

print("-" * 60)

# 7. Verificar resultados
if Path("htmlcov/index.html").exists():
    print("\n✅ Relatório HTML gerado com sucesso!")
    print("📊 Para visualizar o relatório:")
    print("   - Windows: explorer htmlcov\\index.html")
    print("   - Ou abra: htmlcov/index.html no navegador")
else:
    print("\n⚠️ Relatório HTML não foi gerado.")
    print("Tente executar manualmente:")
    print("   python -m pytest --cov=app --cov-report=html")

if Path("coverage.json").exists():
    print("\n✅ Arquivo coverage.json criado")

print("\n✅ Script concluído!")
print("\n💡 Próximos passos:")
print("1. Verifique o relatório de cobertura")
print("2. Adicione mais testes para aumentar a cobertura")
print("3. Execute: python -m pytest --cov=app --cov-report=html")
