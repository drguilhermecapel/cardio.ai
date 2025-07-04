"""
Testes unitários para o serviço unificado de modelos
"""

import unittest
import numpy as np
import os
import tempfile
from pathlib import Path
import sys

# Adicionar diretório pai ao path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar serviço a ser testado
from app.services.unified_model_service import UnifiedModelService, get_model_service


class TestUnifiedModelService(unittest.TestCase):
    """Testes para o serviço unificado de modelos."""
    
    def setUp(self):
        """Configuração para cada teste."""
        # Criar diretório temporário para modelos
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_service = UnifiedModelService(models_dir=self.temp_dir.name)
    
    def tearDown(self):
        """Limpeza após cada teste."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Testar inicialização do serviço."""
        # Verificar se o serviço foi inicializado corretamente
        self.assertIsNotNone(self.model_service)
        self.assertEqual(self.model_service.models_dir, Path(self.temp_dir.name))
        
        # Verificar se pelo menos um modelo demo foi criado
        self.assertGreaterEqual(len(self.model_service.models), 1)
        self.assertIn('demo_ecg_classifier', self.model_service.models)
    
    def test_list_models(self):
        """Testar listagem de modelos."""
        models = self.model_service.list_models()
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 1)
        self.assertIn('demo_ecg_classifier', models)
    
    def test_get_model_info(self):
        """Testar obtenção de informações do modelo."""
        info = self.model_service.get_model_info('demo_ecg_classifier')
        self.assertIsInstance(info, dict)
        self.assertIn('type', info)
        self.assertEqual(info['type'], 'sklearn_demo')
    
    def test_predict_ecg(self):
        """Testar predição de ECG."""
        # Criar dados de teste
        test_data = np.random.randn(5000)
        
        # Realizar predição
        result = self.model_service.predict_ecg('demo_ecg_classifier', test_data)
        
        # Verificar resultado
        self.assertIsInstance(result, dict)
        self.assertIn('predicted_class', result)
        self.assertIn('diagnosis', result)
        self.assertIn('confidence', result)
        self.assertIn('probabilities', result)
    
    def test_nonexistent_model(self):
        """Testar comportamento com modelo inexistente."""
        # Tentar obter informações de modelo inexistente
        info = self.model_service.get_model_info('nonexistent_model')
        self.assertIn('error', info)
        
        # Tentar predição com modelo inexistente
        test_data = np.random.randn(5000)
        with self.assertRaises(ValueError):
            self.model_service.predict_ecg('nonexistent_model', test_data)
    
    def test_singleton_instance(self):
        """Testar se get_model_service retorna a mesma instância."""
        service1 = get_model_service()
        service2 = get_model_service()
        self.assertIs(service1, service2)


if __name__ == '__main__':
    unittest.main()