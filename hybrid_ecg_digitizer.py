
import numpy as np
import cv2
from typing import Union, Dict, List, Tuple, Optional
import os
from scipy import signal, interpolate
from PIL import Image
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridECGDigitizer:
    """
    Digitizer híbrido que tenta múltiplos backends para máxima compatibilidade
    """

    def __init__(self, target_length: int = 1000, verbose: bool = True):
        self.target_length = target_length
        self.verbose = verbose
        self.digitizers = self._initialize_digitizers()
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def _initialize_digitizers(self) -> List[Tuple[str, object]]:
        """Inicializa todos os digitizers disponíveis"""
        digitizers = []

        # ECG-Extract
        try:
            from ecg_extract import ECGExtractor
            digitizers.append(('ecg_extract', ECGExtractor()))
            logger.info("✅ ECG-Extract carregado")
        except ImportError:
            logger.warning("❌ ECG-Extract não disponível")

        # ECG-Scanner
        try:
            from ecg_scanner import ECGScanner
            scanner = ECGScanner()
            digitizers.append(('ecg_scanner', scanner))
            logger.info("✅ ECG-Scanner carregado")
        except ImportError:
            logger.warning("❌ ECG-Scanner não disponível")

        # PyECG
        try:
            from pyecg import ECGDigitizer as PyECGDigitizer
            digitizers.append(('pyecg', PyECGDigitizer()))
            logger.info("✅ PyECG carregado")
        except ImportError:
            logger.warning("❌ PyECG não disponível")

        # Fallback: OpenCV personalizado
        digitizers.append(('opencv_custom', self))
        logger.info("✅ OpenCV fallback disponível")

        return digitizers

    def digitize(self, file_path: str) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Digitaliza ECG tentando múltiplos métodos

        Returns:
            Dict contendo:
                - 'data': array (12, target_length)
                - 'method': método usado
                - 'quality': métricas de qualidade
                - 'metadata': informações adicionais
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        errors = []

        # Tentar cada digitizer
        for name, digitizer in self.digitizers:
            try:
                if self.verbose:
                    print(f"🔄 Tentando {name}...")

                if name == 'ecg_extract':
                    data = self._use_ecg_extract(digitizer, file_path)
                elif name == 'ecg_scanner':
                    data = self._use_ecg_scanner(digitizer, file_path)
                elif name == 'pyecg':
                    data = self._use_pyecg(digitizer, file_path)
                elif name == 'opencv_custom':
                    data = self._use_opencv_custom(file_path)
                else:
                    continue

                # Validar dados
                quality = self._assess_quality(data)
                if quality['score'] > 0.5:
                    if self.verbose:
                        print(f"✅ Sucesso com {name} (qualidade: {quality['score']:.2f})")

                    return {
                        'data': data,
                        'method': name,
                        'quality': quality,
                        'metadata': {
                            'file': os.path.basename(file_path),
                            'shape': data.shape
                        }
                    }
                else:
                    errors.append(f"{name}: Qualidade baixa ({quality['score']:.2f})")

            except Exception as e:
                errors.append(f"{name}: {str(e)}")

        # Se todos falharam
        raise RuntimeError(f"Todos os métodos falharam:\n" + "\n".join(errors))

    def _use_ecg_extract(self, extractor, file_path: str) -> np.ndarray:
        """Usa ECG-Extract"""
        data = extractor.extract(file_path)

        # Padronizar tamanho
        if hasattr(extractor, 'standardize'):
            return extractor.standardize(data, length=self.target_length)
        else:
            return self._resample_to_target(data)

    def _use_ecg_scanner(self, scanner, file_path: str) -> np.ndarray:
        """Usa ECG-Scanner"""
        scanner.load_image(file_path)
        data = scanner.extract_leads()

        # Garantir 12 derivações
        if data.shape[0] != 12:
            raise ValueError(f"ECG-Scanner retornou {data.shape[0]} derivações")

        return self._resample_to_target(data)

    def _use_pyecg(self, digitizer, file_path: str) -> np.ndarray:
        """Usa PyECG"""
        result = digitizer.digitize(file_path)
        data = result.get_12_leads()
        return self._resample_to_target(data)

    def _use_opencv_custom(self, file_path: str) -> np.ndarray:
        """Implementação customizada com OpenCV"""
        # Carregar imagem
        if file_path.lower().endswith('.pdf'):
            image = self._pdf_to_image(file_path)
        else:
            image = cv2.imread(file_path)

        if image is None:
            raise ValueError("Não foi possível carregar a imagem")

        # Pré-processar
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar layout e extrair
        ecg_data = self._extract_12_leads_opencv(gray)

        return ecg_data

    def _pdf_to_image(self, pdf_path: str) -> np.ndarray:
        """Converte PDF para imagem"""
        from pdf2image import convert_from_path

        pages = convert_from_path(pdf_path, dpi=300)
        if not pages:
            raise ValueError("PDF vazio")

        # Converter para formato OpenCV
        pil_image = pages[0]
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image

    def _extract_12_leads_opencv(self, gray_image: np.ndarray) -> np.ndarray:
        """Extração customizada com OpenCV"""
        height, width = gray_image.shape

        # Melhorar contraste
        gray_image = cv2.equalizeHist(gray_image)

        # Detectar grid (assumindo 3x4)
        leads_data = np.zeros((12, self.target_length))

        # Dimensões de cada célula
        rows, cols = 3, 4
        cell_h = height // rows
        cell_w = width // cols

        for idx in range(12):
            row = idx // cols
            col = idx % cols

            # Extrair região
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w

            region = gray_image[y1:y2, x1:x2]

            # Extrair sinal
            signal_data = self._extract_signal_from_region_advanced(region)

            # Padronizar tamanho
            if len(signal_data) > 0:
                leads_data[idx] = self._resample_signal(signal_data, self.target_length)

        return leads_data

    def _extract_signal_from_region_advanced(self, region: np.ndarray) -> np.ndarray:
        """Extração avançada de sinal de uma região"""
        # Aplicar threshold adaptativo
        thresh = cv2.adaptiveThreshold(
            region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Remover ruído
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Extrair contorno principal
        signal_points = []
        h, w = cleaned.shape

        for x in range(w):
            column = cleaned[:, x]
            points = np.where(column > 0)[0]

            if len(points) > 0:
                # Pegar ponto médio ponderado
                weights = column[points] / 255.0
                y = np.average(points, weights=weights)
                signal_points.append(y)
            elif signal_points:
                # Interpolar
                signal_points.append(signal_points[-1])

        if not signal_points:
            return np.zeros(self.target_length)

        # Converter para array e normalizar
        signal = np.array(signal_points)

        # Inverter e centralizar
        signal = h/2 - signal

        # Remover outliers
        signal = self._remove_outliers(signal)

        # Normalizar
        if np.std(signal) > 0:
            signal = (signal - np.mean(signal)) / np.std(signal)

        return signal

    def _remove_outliers(self, signal: np.ndarray, threshold: float = 3) -> np.ndarray:
        """Remove outliers usando z-score"""
        z_scores = np.abs((signal - np.mean(signal)) / (np.std(signal) + 1e-10))
        signal_clean = signal.copy()

        # Substituir outliers por interpolação
        outlier_indices = np.where(z_scores > threshold)[0]
        for idx in outlier_indices:
            if 0 < idx < len(signal) - 1:
                signal_clean[idx] = (signal_clean[idx-1] + signal_clean[idx+1]) / 2

        return signal_clean

    def _resample_signal(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Reamostra sinal preservando morfologia"""
        if len(signal) == 0:
            return np.zeros(target_length)

        # Usar interpolação spline para preservar forma de onda
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_length)

        # Spline cúbica
        try:
            f = interpolate.UnivariateSpline(x_old, signal, s=0, k=3)
            return f(x_new)
        except:
            # Fallback para linear
            f = interpolate.interp1d(x_old, signal, kind='linear',
                                   bounds_error=False, fill_value=0)
            return f(x_new)

    def _resample_to_target(self, data: np.ndarray) -> np.ndarray:
        """Reamostra todas as derivações para tamanho alvo"""
        n_leads = data.shape[0]
        resampled = np.zeros((n_leads, self.target_length))

        for i in range(n_leads):
            resampled[i] = self._resample_signal(data[i], self.target_length)

        return resampled

    def _assess_quality(self, data: np.ndarray) -> Dict[str, float]:
        """Avalia qualidade da digitalização"""
        scores = []
        issues = []

        for i, lead in enumerate(data):
            # Verificar variância
            std = np.std(lead)
            if std < 0.01:
                scores.append(0)
                issues.append(f"Lead {self.lead_names[i]}: sem sinal")
            else:
                # Calcular SNR aproximado
                diff = np.diff(lead)
                noise_var = np.var(diff)
                signal_var = np.var(lead)

                if noise_var > 0:
                    snr = signal_var / noise_var
                    score = min(snr / 10, 1.0)
                else:
                    score = 1.0

                scores.append(score)

                if score < 0.5:
                    issues.append(f"Lead {self.lead_names[i]}: baixa qualidade")

        overall_score = np.mean(scores)

        return {
            'score': overall_score,
            'lead_scores': scores,
            'issues': issues,
            'usable': overall_score > 0.5
        }
