"""Teste mínimo para verificar funcionamento."""
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
