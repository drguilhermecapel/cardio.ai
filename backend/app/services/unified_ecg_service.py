# backend/app/services/unified_ecg_service.py
import numpy as np
import logging
from typing import List

from .interfaces import ECGDigitizerInterface, PreprocessingPipelineInterface, MLModelServiceInterface
from app.schemas.ecg import ECGDiagnosis
from app.utils.ecg_processor import ECGProcessor

logger = logging.getLogger(__name__)

class UnifiedECGService:
    """
    Serviço unificado para orquestrar o diagnóstico de ECG a partir
    de diferentes fontes (sinal digital ou imagem).
    """
    def __init__(self,
                 digitizer: ECGDigitizerInterface,
                 pipeline: PreprocessingPipelineInterface,
                 model_service: MLModelServiceInterface,
                 ecg_processor: ECGProcessor):
        self.digitizer = digitizer
        self.pipeline = pipeline
        self.model_service = model_service
        self.ecg_processor = ecg_processor
        logger.info("UnifiedECGService inicializado com sucesso.")

    async def diagnose_from_image(self, image_path: str) -> ECGDiagnosis:
        """
        Pipeline completo para diagnóstico a partir de uma imagem de ECG.
        1. Digitaliza a imagem para extrair o sinal.
        2. Pré-processa o sinal extraído.
        3. Realiza a predição com o modelo.
        4. Formata e retorna o diagnóstico.
        """
        logger.info(f"Iniciando diagnóstico a partir da imagem: {image_path}")
        try:
            # 1. Digitalização da Imagem
            signal_data, sampling_rate = await self.digitizer.digitize(image_path)
            if signal_data is None or sampling_rate is None:
                raise ValueError("A digitalização da imagem falhou e não retornou dados de sinal.")
            logger.info(f"Imagem digitalizada. Shape do sinal: {signal_data.shape}, Frequência: {sampling_rate} Hz")

            # O restante do pipeline é comum
            return await self._process_and_diagnose(signal_data, sampling_rate)

        except Exception as e:
            logger.error(f"Erro durante o diagnóstico a partir da imagem: {e}", exc_info=True)
            raise

    async def diagnose_from_signal(self, signal_data: List[List[float]], sampling_rate: int = 500) -> ECGDiagnosis:
        """
        Pipeline completo para diagnóstico a partir de um sinal digital.
        1. Converte a lista de dados para um array numpy.
        2. Pré-processa o sinal.
        3. Realiza a predição com o modelo.
        4. Formata e retorna o diagnóstico.
        """
        logger.info("Iniciando diagnóstico a partir de sinal digital.")
        try:
            signal_np = np.array(signal_data, dtype=np.float32)
            
            # O restante do pipeline é comum
            return await self._process_and_diagnose(signal_np, sampling_rate)

        except Exception as e:
            logger.error(f"Erro durante o diagnóstico a partir do sinal: {e}", exc_info=True)
            raise

    async def _process_and_diagnose(self, signal_data: np.ndarray, sampling_rate: int) -> ECGDiagnosis:
        """
        Lógica de processamento e diagnóstico compartilhada.
        """
        # 2. Pré-processamento do Sinal
        logger.info(f"Iniciando pré-processamento do sinal com shape: {signal_data.shape}")
        processed_signal = await self.pipeline.process(signal_data, sampling_rate)
        logger.info(f"Sinal pré-processado. Shape final: {processed_signal.shape}")
        
        # 3. Predição do Modelo
        logger.info("Enviando sinal para o serviço do modelo para predição.")
        raw_prediction = await self.model_service.predict(processed_signal)
        logger.info(f"Predição bruta recebida do modelo: {raw_prediction}")

        # 4. Cálculo de Métricas e Formatação do Diagnóstico
        logger.info("Calculando métricas do ECG e formatando o diagnóstico.")
        heart_rate = self.ecg_processor.calculate_heart_rate(signal_data, sampling_rate)
        qrs_duration = self.ecg_processor.calculate_qrs_duration(signal_data, sampling_rate)
        
        diagnosis = self.model_service.format_prediction(raw_prediction)
        
        # Adiciona as métricas calculadas ao diagnóstico
        diagnosis.heart_rate = heart_rate
        diagnosis.qrs_duration = f"{qrs_duration:.2f} ms"
        
        logger.info(f"Diagnóstico final gerado: {diagnosis.model_dump_json(indent=2)}")
        
        return diagnosis

