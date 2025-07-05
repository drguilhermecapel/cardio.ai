# backend/app/services/ecg_digitizer_service.py
import numpy as np
import cv2
from ecg_digitize.ecg_digitizer import digitize_ecg_image
from fastapi import HTTPException
import logging
from typing import Dict, Any, List

# Configuração do Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECGDigitizerService:
    """
    Serviço para encapsular a lógica de digitalização de imagens de ECG.
    Realiza o processamento da imagem e extrai os dados do sinal.
    """

    def digitize_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Digitaliza uma imagem de ECG para extrair os dados do sinal.

        Args:
            image_bytes: A imagem de ECG no formato de bytes.

        Returns:
            Um dicionário contendo os dados do sinal, taxa de amostragem e nomes das derivações.
        
        Raises:
            HTTPException: Se a imagem não puder ser decodificada ou se os dados não puderem ser extraídos.
        """
        try:
            # Converte os bytes da imagem para um array OpenCV, que é o formato esperado pela biblioteca.
            image_np = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            if image is None:
                logger.error("Falha ao decodificar a imagem. O formato pode ser inválido ou o arquivo corrompido.")
                raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem. Verifique o formato do arquivo (PNG, JPG).")

            logger.info("Iniciando a digitalização da imagem de ECG.")
            
            # Chama a função principal da biblioteca ECG-Digitiser.
            # Ela retorna os sinais, a taxa de amostragem e os nomes das derivações detectadas.
            digitized_ecg_data, sampling_rate, lead_names_detected = digitize_ecg_image(image)

            if digitized_ecg_data is None or digitized_ecg_data.size == 0:
                logger.warning("A digitalização não retornou dados. A imagem pode não ser um ECG válido ou ter baixa qualidade.")
                raise HTTPException(status_code=400, detail="Não foi possível extrair dados de ECG da imagem. Verifique a qualidade e o enquadramento da imagem.")

            logger.info(f"Digitalização concluída. Taxa de amostragem: {sampling_rate} Hz. Shape dos dados: {digitized_ecg_data.shape}")

            # A biblioteca retorna os canais nas colunas. Transpomos para que cada linha seja um canal.
            signals: List[List[float]] = digitized_ecg_data.T.tolist()

            # Padrão de 12 derivações
            standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

            # Garante que temos 12 derivações, preenchendo com zeros se necessário
            if len(signals) < 12:
                logger.warning(f"Detectadas apenas {len(signals)} derivações. Preenchendo as restantes com zeros.")
                num_points = len(signals[0]) if signals else 0
                padding = [[0.0] * num_points for _ in range(12 - len(signals))]
                signals.extend(padding)
            elif len(signals) > 12:
                logger.warning(f"Detectadas {len(signals)} derivações. Truncando para 12.")
                signals = signals[:12]

            return {
                "signal_data": signals,
                "sampling_rate": int(sampling_rate),
                "lead_names": standard_leads
            }

        except HTTPException as http_exc:
            # Re-lança exceções HTTP para que o FastAPI as manipule corretamente.
            raise http_exc
        except Exception as e:
            logger.exception(f"Ocorreu um erro inesperado durante a digitalização: {e}")
            raise HTTPException(status_code=500, detail=f"Um erro interno ocorreu no processo de digitalização: {str(e)}")

# Cria uma instância única do serviço (padrão Singleton) para ser usada em toda a aplicação.
ecg_digitizer_service = ECGDigitizerService()

