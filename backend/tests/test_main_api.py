"""
Testes de integração para a API principal
"""

import unittest
from fastapi.testclient import TestClient
import os
import sys
import json
import numpy as np
import tempfile

# Adicionar diretório pai ao path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar aplicação a ser testada
from app.main_unified import app


class TestMainAPI(unittest.TestCase):
    """Testes para a API principal."""
    
    def setUp(self):
        """Configuração para cada teste."""
        self.client = TestClient(app)
        
        # Criar arquivo de ECG de teste
        self.test_ecg_data = np.sin(np.linspace(0, 20*np.pi, 5000))
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        np.savetxt(self.temp_file.name, self.test_ecg_data)
        self.temp_file.close()
    
    def tearDown(self):
        """Limpeza após cada teste."""
        os.unlink(self.temp_file.name)
    
    def test_root_endpoint(self):
        """Testar endpoint raiz."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "CardioAI Pro")
        self.assertEqual(data["version"], "2.0.0")
        self.assertIn("endpoints", data)
    
    def test_health_endpoint(self):
        """Testar endpoint de health check."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "CardioAI Pro")
        self.assertIn("services", data)
    
    def test_info_endpoint(self):
        """Testar endpoint de informações."""
        response = self.client.get("/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("system", data)
        self.assertIn("capabilities", data)
        self.assertIn("models", data)
    
    def test_api_health_endpoint(self):
        """Testar endpoint de health check da API."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["api_version"], "v1")
        self.assertIn("endpoints", data)
    
    def test_list_models_endpoint(self):
        """Testar endpoint de listagem de modelos."""
        response = self.client.get("/api/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertIn("count", data)
        self.assertGreaterEqual(data["count"], 1)
    
    def test_model_info_endpoint(self):
        """Testar endpoint de informações de modelo."""
        # Primeiro listar modelos para obter um nome válido
        response = self.client.get("/api/v1/models")
        models = response.json()["models"]
        
        if models:
            # Testar com modelo existente
            response = self.client.get(f"/api/v1/models/{models[0]}")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("type", data)
        
        # Testar com modelo inexistente
        response = self.client.get("/api/v1/models/nonexistent_model")
        self.assertEqual(response.status_code, 404)
    
    def test_ecg_upload_analyze_flow(self):
        """Testar fluxo completo de upload e análise de ECG."""
        # Upload de arquivo
        with open(self.temp_file.name, "rb") as f:
            response = self.client.post(
                "/api/v1/ecg/upload",
                files={"file": ("test_ecg.csv", f, "text/csv")}
            )
        
        self.assertEqual(response.status_code, 200)
        upload_data = response.json()
        self.assertIn("process_id", upload_data)
        
        # Análise do ECG
        process_id = upload_data["process_id"]
        response = self.client.post(f"/api/v1/ecg/analyze/{process_id}")
        
        self.assertEqual(response.status_code, 200)
        analysis_data = response.json()
        self.assertIn("prediction", analysis_data)
        self.assertIn("model_used", analysis_data)


if __name__ == '__main__':
    unittest.main()