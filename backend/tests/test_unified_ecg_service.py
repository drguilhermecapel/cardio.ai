"""
Testes unitários para o serviço unificado de ECG
"""

import unittest
import numpy as np
import os
import tempfile
import json
from pathlib import Path
import sys

# Adicionar diretório pai ao path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar serviço a ser testado
from app.services.unified_ecg_service import UnifiedECGService, get_ecg_service


class TestUnifiedECGService(unittest.TestCase):
    """Testes para o serviço unificado de ECG."""
    
    def setUp(self):
        """Configuração para cada teste."""
        self.ecg_service = UnifiedECGService()
        
        # Criar diretório temporário para dados
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ecg_service.data_dir = Path(self.temp_dir.name)
        
        # Criar arquivo de ECG de teste
        self.test_ecg_data = np.sin(np.linspace(0, 20*np.pi, 5000))
        self.test_file_path = os.path.join(self.temp_dir.name, "test_ecg.csv")
        np.savetxt(self.test_file_path, self.test_ecg_data)
        
        # Criar arquivo JSON de teste
        self.test_json_path = os.path.join(self.temp_dir.name, "test_ecg.json")
        with open(self.test_json_path, 'w') as f:
            json.dump({
                "ecg_data": self.test_ecg_data.tolist(),
                "metadata": {
                    "sampling_rate": 500,
                    "leads": ["II"],
                    "units": "mV",
                    "patient_data": {
                        "id": "test123",
                        "age": 45,
                        "gender": "M"
                    }
                }
            }, f)
    
    def tearDown(self):
        """Limpeza após cada teste."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Testar inicialização do serviço."""
        self.assertIsNotNone(self.ecg_service)
        self.assertEqual(self.ecg_service.sampling_rate, 500)
        self.assertEqual(self.ecg_service.target_length, 5000)
    
    def test_process_csv_file(self):
        """Testar processamento de arquivo CSV."""
        result = self.ecg_service.process_ecg_file(self.test_file_path, "csv")
        
        # Verificar resultado
        self.assertIsInstance(result, dict)
        self.assertIn("process_id", result)
        self.assertIn("processed_data", result)
        self.assertIn("metadata", result)
        
        # Verificar dados processados
        processed_data = result["processed_data"]
        self.assertIn("raw_data", processed_data)
        self.assertIn("filtered_data", processed_data)
        self.assertIn("normalized_data", processed_data)
    
    def test_process_json_file(self):
        """Testar processamento de arquivo JSON."""
        result = self.ecg_service.process_ecg_file(self.test_json_path, "json")
        
        # Verificar resultado
        self.assertIsInstance(result, dict)
        self.assertIn("process_id", result)
        self.assertIn("processed_data", result)
        self.assertIn("metadata", result)
        
        # Verificar metadados
        metadata = result["metadata"]
        self.assertEqual(metadata["sampling_rate"], 500)
        self.assertEqual(metadata["leads"], ["II"])
        self.assertEqual(metadata["patient_data"]["age"], 45)
    
    def test_process_ecg_data(self):
        """Testar processamento direto de dados de ECG."""
        result = self.ecg_service.process_ecg_data(self.test_ecg_data)
        
        # Verificar resultado
        self.assertIsInstance(result, dict)
        self.assertIn("process_id", result)
        self.assertIn("processed_data", result)
        self.assertIn("metadata", result)
    
    def test_analyze_ecg(self):
        """Testar análise de ECG."""
        # Primeiro processar dados
        process_result = self.ecg_service.process_ecg_data(self.test_ecg_data)
        process_id = process_result["process_id"]
        
        # Analisar dados processados
        analysis_result = self.ecg_service.analyze_ecg(process_id)
        
        # Verificar resultado
        self.assertIsInstance(analysis_result, dict)
        self.assertIn("process_id", analysis_result)
        self.assertIn("model_used", analysis_result)
        self.assertIn("prediction", analysis_result)
        
        # Verificar predição
        prediction = analysis_result["prediction"]
        self.assertIn("predicted_class", prediction)
        self.assertIn("diagnosis", prediction)
        self.assertIn("confidence", prediction)
    
    def test_invalid_file_format(self):
        """Testar comportamento com formato de arquivo inválido."""
        with self.assertRaises(ValueError):
            self.ecg_service.process_ecg_file(self.test_file_path, "invalid_format")
    
    def test_nonexistent_process_id(self):
        """Testar comportamento com ID de processo inexistente."""
        result = self.ecg_service.analyze_ecg("nonexistent_id")
        self.assertIn("error", result)
    
    def test_singleton_instance(self):
        """Testar se get_ecg_service retorna a mesma instância."""
        service1 = get_ecg_service()
        service2 = get_ecg_service()
        self.assertIs(service1, service2)


if __name__ == '__main__':
    unittest.main()