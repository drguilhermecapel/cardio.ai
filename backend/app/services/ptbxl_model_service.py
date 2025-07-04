# backend/app/services/ptbxl_model_service.py

import os
import json
import numpy as np
import tensorflow as tf  # <-- IMPORTANTE: Importar TensorFlow
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class PTBXLModelService:
    """
    Serviço de modelo modificado para carregar e servir um TensorFlow SavedModel.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PTBXLModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        # --- MODIFICAÇÃO 1: Alterar o caminho para o diretório do SavedModel ---
        # O caminho agora aponta para a pasta, não para um arquivo.
        self.model_path = os.path.join(os.path.dirname(__file__), '../../models/ptbxl_saved_model')
        self.class_map_path = os.path.join(os.path.dirname(__file__), '../../models/ptbxl_classes.json')
        
        self.model = None
        self.predictor = None
        self._output_key = None
        self.class_names = []
        self._load_resources()
        self._initialized = True

    def _load_resources(self):
        """Carrega o modelo e os recursos associados."""
        print("Carregando recursos para PTBXLModelService...")
        
        # Verificar se o diretório do SavedModel existe
        if not os.path.isdir(self.model_path):
            logger.warning(f"Diretório do SavedModel não encontrado: {self.model_path}")
            logger.info("Tentando usar modelo .h5 como fallback...")
            self._load_h5_fallback()
            return

        # --- MODIFICAÇÃO 2: Alterar a lógica de carregamento do modelo ---
        try:
            # Carrega o modelo usando a função específica do TensorFlow para SavedModel
            self.model = tf.saved_model.load(self.model_path)
            # O modelo carregado geralmente é uma função ou um objeto com 'signatures'.
            # Vamos pegar a assinatura padrão de serviço, que é o que o TF Serving usaria.
            self.predictor = self.model.signatures['serving_default']
            print("TensorFlow SavedModel carregado com sucesso.")
            
            # Vamos inspecionar as saídas esperadas da predição
            self._output_key = list(self.predictor.structured_outputs.keys())[0]
            print(f"Chave de saída da predição identificada: '{self._output_key}'")

        except Exception as e:
            print(f"ERRO CRÍTICO ao carregar o TensorFlow SavedModel: {e}")
            logger.info("Tentando usar modelo .h5 como fallback...")
            self._load_h5_fallback()
            return

        # Carregar mapeamento de classes
        try:
            with open(self.class_map_path) as f:
                self.class_names = json.load(f)
            print("Classes de diagnóstico carregadas.")
        except Exception as e:
            logger.warning(f"Erro ao carregar classes: {e}")
            self._create_default_classes()

    def _load_h5_fallback(self):
        """Carrega modelo .h5 como fallback se SavedModel não estiver disponível."""
        try:
            # Caminhos possíveis para o modelo .h5
            h5_paths = [
                os.path.join(os.path.dirname(__file__), '../../models/ptbxl_model.h5'),
                os.path.join(os.path.dirname(__file__), '../../models/ecg_model_final.h5'),
                'models/ptbxl_model.h5',
                'models/ecg_model_final.h5'
            ]
            
            for h5_path in h5_paths:
                if os.path.exists(h5_path):
                    logger.info(f"Carregando modelo .h5 fallback: {h5_path}")
                    self.model = tf.keras.models.load_model(h5_path)
                    self.predictor = None  # Não há predictor para modelo Keras direto
                    self._output_key = None
                    logger.info("Modelo .h5 carregado como fallback")
                    return
            
            logger.error("Nenhum modelo encontrado (nem SavedModel nem .h5)")
            self._create_dummy_model()
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo .h5 fallback: {e}")
            self._create_dummy_model()

    def _create_dummy_model(self):
        """Cria um modelo dummy para testes."""
        logger.warning("Criando modelo dummy para testes...")
        self.model = None
        self.predictor = None
        self._output_key = None

    def _create_default_classes(self):
        """Cria mapeamento de classes padrão."""
        self.class_names = [
            "Normal ECG",
            "Atrial Fibrillation", 
            "1st Degree AV Block",
            "Left Bundle Branch Block",
            "Right Bundle Branch Block",
            "Premature Atrial Contraction",
            "Premature Ventricular Contraction", 
            "ST-T Change",
            "Sinus Bradycardia",
            "Sinus Tachycardia",
            "Left Atrial Enlargement",
            "Left Ventricular Hypertrophy",
            "Right Ventricular Hypertrophy",
            "Myocardial Infarction"
        ]

    def predict(self, data: np.ndarray) -> dict:
        """
        Realiza uma predição usando o SavedModel carregado.

        Args:
            data (np.ndarray): O dado de entrada para o modelo.

        Returns:
            Um dicionário com as probabilidades de cada classe.
        """
        if self.model is None:
            return {"error": "Modelo não foi carregado corretamente."}

        try:
            # --- MODIFICAÇÃO 3: Adaptar a chamada de predição ---
            if self.predictor is not None:
                # Usar SavedModel com predictor
                # Garante que a entrada seja um tensor do tipo correto (float32 é comum)
                input_tensor = tf.constant(data, dtype=tf.float32)
                
                # Chama a predição através da assinatura do modelo
                predictions_dict = self.predictor(input_tensor)
                
                # A saída de uma assinatura é um dicionário. Precisamos extrair o tensor de saída.
                probabilities = predictions_dict[self._output_key].numpy()[0] # [0] para pegar o primeiro item do batch
            else:
                # Usar modelo Keras direto (fallback)
                probabilities = self.model.predict(data, verbose=0)[0]

            # Mapeia as probabilidades para os nomes das classes
            diagnostic = [
                {"class": self.class_names[i] if i < len(self.class_names) else f"Class_{i}", 
                 "probability": float(probabilities[i])}
                for i in range(len(probabilities))
            ]
            
            # Ordena o resultado pela maior probabilidade
            diagnostic.sort(key=lambda x: x['probability'], reverse=True)
            
            return {"diagnostics": diagnostic}

        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {"error": f"Erro na predição: {str(e)}"}

    # Funções de auxílio como get_model, get_class_names, etc., podem ser mantidas ou adaptadas
    def get_model(self):
        # Retornar o modelo carregado pode ser útil para o serviço de explicabilidade
        return self.model
    
    def get_class_names(self) -> list:
        return self.class_names

    def predict_ecg(self, ecg_data: np.ndarray, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Realiza predição de ECG compatível com a interface existente.
        
        Args:
            ecg_data: Array ECG no formato (12, 1000) ou (batch, 12, 1000)
            metadata: Metadados adicionais
            
        Returns:
            Dicionário com resultados da predição
        """
        try:
            if self.model is None:
                return {"error": "Modelo não disponível"}
            
            # Garantir formato correto
            if ecg_data.ndim == 2:
                ecg_data = ecg_data[np.newaxis, :]  # Adicionar dimensão batch
            
            logger.info(f"Realizando predição - Input shape: {ecg_data.shape}")
            
            # Usar método predict padrão
            result = self.predict(ecg_data)
            
            if "error" in result:
                return result
            
            # Processar resultados para formato compatível
            diagnoses = result.get("diagnostics", [])
            
            return {
                "success": True,
                "model_used": "tensorflow_savedmodel" if self.predictor else "tensorflow_keras",
                "primary_diagnosis": diagnoses[0] if diagnoses else None,
                "top_diagnoses": diagnoses[:5],
                "total_classes": len(self.class_names),
                "metadata": metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Erro na predição ECG: {e}")
            return {"error": f"Erro na predição: {str(e)}"}

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações do modelo."""
        return {
            "model_type": "tensorflow_savedmodel" if self.predictor else "tensorflow_keras",
            "model_available": self.model is not None,
            "savedmodel_path": self.model_path,
            "classes_loaded": len(self.class_names),
            "predictor_available": self.predictor is not None,
            "output_key": self._output_key
        }

# Instância global do serviço
_ptbxl_service_instance = None

def get_ptbxl_service() -> PTBXLModelService:
    """Retorna instância singleton do serviço PTB-XL."""
    global _ptbxl_service_instance
    
    if _ptbxl_service_instance is None:
        logger.info("Criando nova instância do serviço PTB-XL...")
        _ptbxl_service_instance = PTBXLModelService()
    
    return _ptbxl_service_instance

def reset_ptbxl_service():
    """Reinicializa o serviço PTB-XL."""
    global _ptbxl_service_instance
    _ptbxl_service_instance = None
    logger.info("Serviço PTB-XL reinicializado")

