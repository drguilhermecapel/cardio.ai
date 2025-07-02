
import numpy as np
import cv2
from typing import Union, Dict, List, Tuple, Optional
import os
from scipy import signal, interpolate
from scipy.signal import find_peaks, butter, filtfilt
from PIL import Image
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from skimage import morphology, filters
import warnings
warnings.filterwarnings('ignore')

class CustomECGDigitizer:
    """
    Digitizer ECG customizado otimizado para Google Colab
    """

    def __init__(self, target_length: int = 1000, debug: bool = False):
        self.target_length = target_length
        self.debug = debug
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # Configurações de detecção
        self.config = {
            'dpi': 300,
            'min_signal_std': 0.01,
            'quality_threshold': 0.3,
            'grid_removal': True,
            'denoise': True,
            'adaptive_threshold': True
        }

    def digitize(self, file_path: str) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Digitaliza ECG de arquivo PDF/JPG/JPEG
        """
        print(f"📄 Processando: {os.path.basename(file_path)}")

        # Carregar imagem
        image = self._load_image(file_path)

        if self.debug:
            self._show_image(image, "Imagem Original")

        # Pré-processar
        processed = self._preprocess_image(image)

        # Detectar layout
        layout = self._detect_layout(processed)
        print(f"📐 Layout detectado: {layout['type']}")

        # Extrair derivações
        ecg_data = self._extract_leads(processed, layout)

        # Avaliar qualidade
        quality = self._assess_quality(ecg_data)

        # Pós-processar sinais
        if quality['score'] > self.config['quality_threshold']:
            ecg_data = self._post_process_signals(ecg_data)
            quality = self._assess_quality(ecg_data)  # Reavaliar

        print(f"✅ Digitalização concluída (qualidade: {quality['score']:.2f})")

        return {
            'data': ecg_data,
            'method': 'custom_opencv',
            'quality': quality,
            'layout': layout,
            'metadata': {
                'file': os.path.basename(file_path),
                'shape': ecg_data.shape
            }
        }

    def _load_image(self, file_path: str) -> np.ndarray:
        """Carrega imagem de arquivo"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            # Converter PDF para imagem
            pages = convert_from_path(file_path, dpi=self.config['dpi'])
            if not pages:
                raise ValueError("PDF vazio ou corrompido")

            # Converter PIL para OpenCV
            pil_image = pages[0]
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            # Carregar imagem diretamente
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Não foi possível carregar: {file_path}")

        return image

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Pré-processa imagem para melhor extração"""
        # Converter para escala de cinza
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Redimensionar se muito grande
        max_dim = 3000
        if gray.shape[0] > max_dim or gray.shape[1] > max_dim:
            scale = max_dim / max(gray.shape)
            new_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

        # Melhorar contraste
        gray = cv2.equalizeHist(gray)

        # Remover ruído
        if self.config['denoise']:
            gray = cv2.fastNlMeansDenoising(gray, h=10)

        # Remover grade se configurado
        if self.config['grid_removal']:
            gray = self._remove_grid(gray)

        return gray

    def _remove_grid(self, image: np.ndarray) -> np.ndarray:
        """Remove linhas de grade do papel milimetrado"""
        # Detectar linhas usando transformada de Hough
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                               minLineLength=100, maxLineGap=10)

        if lines is not None:
            # Criar máscara das linhas
            mask = np.ones_like(image) * 255
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Verificar se é linha horizontal ou vertical
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 5 or angle > 175 or (85 < angle < 95):
                    cv2.line(mask, (x1, y1), (x2, y2), 0, 2)

            # Aplicar inpainting
            result = cv2.inpaint(image, 255 - mask, 3, cv2.INPAINT_TELEA)
            return result

        return image

    def _detect_layout(self, image: np.ndarray) -> Dict:
        """Detecta o layout do ECG (3x4, 6x2, etc.)"""
        h, w = image.shape
        aspect_ratio = w / h

        # Heurísticas para detectar layout
        if aspect_ratio > 1.3:  # Paisagem
            if aspect_ratio > 1.8:
                layout_type = "3x4_plus"  # 3x4 com tira de ritmo
            else:
                layout_type = "3x4"
        else:  # Retrato
            layout_type = "6x2"

        # Detectar número de colunas e linhas
        if layout_type == "3x4":
            rows, cols = 3, 4
        elif layout_type == "3x4_plus":
            rows, cols = 4, 4  # Incluindo tira de ritmo
        else:
            rows, cols = 6, 2

        return {
            'type': layout_type,
            'rows': rows,
            'cols': cols,
            'has_rhythm_strip': 'plus' in layout_type
        }

    def _extract_leads(self, image: np.ndarray, layout: Dict) -> np.ndarray:
        """Extrai sinais das 12 derivações"""
        h, w = image.shape
        rows = layout['rows']
        cols = layout['cols']

        # Calcular dimensões de cada célula
        if layout['has_rhythm_strip']:
            # Última linha é tira de ritmo (mais alta)
            cell_h = h // (rows + 1)  # Aproximação
            rhythm_h = cell_h * 2
        else:
            cell_h = h // rows
            rhythm_h = 0

        cell_w = w // cols

        # Extrair cada derivação
        ecg_signals = np.zeros((12, self.target_length))

        for idx in range(12):
            row = idx // cols
            col = idx % cols

            # Calcular região da derivação
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w

            # Adicionar margem
            margin = 10
            y1 = max(0, y1 + margin)
            y2 = min(h, y2 - margin)
            x1 = max(0, x1 + margin)
            x2 = min(w, x2 - margin)

            # Extrair região
            region = image[y1:y2, x1:x2]

            if self.debug:
                self._show_image(region, f"Lead {self.lead_names[idx]}")

            # Extrair sinal
            signal_data = self._extract_signal_from_region(region, idx)

            # Padronizar tamanho
            ecg_signals[idx] = self._resample_signal(signal_data, self.target_length)

        return ecg_signals

    def _extract_signal_from_region(self, region: np.ndarray, lead_idx: int) -> np.ndarray:
        """Extrai sinal de uma região específica"""
        h, w = region.shape

        # Binarização adaptativa
        if self.config['adaptive_threshold']:
            binary = cv2.adaptiveThreshold(
                region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 5
            )
        else:
            _, binary = cv2.threshold(region, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Operações morfológicas para limpar
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos por tamanho
        if contours:
            # Pegar o maior contorno (provavelmente o sinal)
            main_contour = max(contours, key=cv2.contourArea)

            # Criar máscara do contorno principal
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [main_contour], -1, 255, -1)

            # Aplicar máscara
            binary = cv2.bitwise_and(binary, mask)

        # Extrair pontos do sinal
        signal_points = []

        for x in range(w):
            column = binary[:, x]
            white_pixels = np.where(column > 0)[0]

            if len(white_pixels) > 0:
                # Estratégias diferentes para diferentes derivações
                if lead_idx in [3, 4, 5]:  # aVR, aVL, aVF podem ter amplitudes menores
                    y = np.mean(white_pixels)
                else:
                    # Pegar o ponto mais central
                    y = np.median(white_pixels)
                signal_points.append(y)
            else:
                # Interpolar pontos faltantes
                if signal_points:
                    signal_points.append(signal_points[-1])
                else:
                    signal_points.append(h // 2)

        # Converter para array numpy
        signal = np.array(signal_points)

        # Inverter e normalizar
        signal = h/2 - signal  # Inverter eixo Y

        # Suavizar sinal
        if len(signal) > 5:
            signal = signal_filters.savgol_filter(signal, 5, 2)

        # Remover tendência
        signal = signal - signal_filters.savgol_filter(signal,
                                                      min(len(signal)-1, 51), 1)

        return signal

    def _resample_signal(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Reamostra sinal para tamanho alvo preservando morfologia"""
        if len(signal) == 0:
            return np.zeros(target_length)

        if len(signal) == target_length:
            return signal

        # Criar interpolador
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_length)

        # Usar diferentes métodos dependendo da relação de tamanhos
        if len(signal) < target_length // 2:
            # Upsampling significativo - usar cubic
            method = 'cubic'
        else:
            # Downsampling ou upsampling pequeno - usar linear
            method = 'linear'

        try:
            f = interpolate.interp1d(x_old, signal, kind=method,
                                   bounds_error=False, fill_value='extrapolate')
            resampled = f(x_new)
        except:
            # Fallback para método mais robusto
            resampled = np.interp(x_new, x_old, signal)

        return resampled

    def _post_process_signals(self, signals: np.ndarray) -> np.ndarray:
        """Pós-processamento dos sinais extraídos"""
        processed = np.zeros_like(signals)

        for i in range(signals.shape[0]):
            signal = signals[i]

            # Remover drift de linha de base
            signal = self._remove_baseline_wander(signal)

            # Filtrar ruído de alta frequência
            signal = self._filter_signal(signal)

            # Normalizar amplitude
            if np.std(signal) > 0:
                signal = (signal - np.mean(signal)) / np.std(signal)

            processed[i] = signal

        return processed

    def _remove_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Remove drift de linha de base"""
        # Filtro passa-alta butterworth
        fs = 500  # Frequência de amostragem assumida
        fc = 0.5  # Frequência de corte

        nyquist = fs / 2
        if fc < nyquist:
            b, a = butter(3, fc / nyquist, btype='high')
            filtered = filtfilt(b, a, signal)
            return filtered

        return signal

    def _filter_signal(self, signal: np.ndarray) -> np.ndarray:
        """Filtra ruído de alta frequência"""
        # Filtro passa-baixa
        fs = 500
        fc = 40  # 40 Hz para ECG diagnóstico

        nyquist = fs / 2
        if fc < nyquist:
            b, a = butter(3, fc / nyquist, btype='low')
            filtered = filtfilt(b, a, signal)
            return filtered

        return signal

    def _assess_quality(self, signals: np.ndarray) -> Dict:
        """Avalia qualidade dos sinais extraídos"""
        scores = []
        issues = []

        for i, signal in enumerate(signals):
            lead_name = self.lead_names[i]

            # Verificar se há sinal
            std = np.std(signal)
            if std < self.config['min_signal_std']:
                scores.append(0)
                issues.append(f"{lead_name}: Sem sinal detectado")
                continue

            # Calcular métricas de qualidade
            # 1. Variação do sinal
            signal_range = np.ptp(signal)

            # 2. Suavidade (menos ruído)
            diff2 = np.diff(signal, n=2)
            smoothness = 1 / (1 + np.std(diff2))

            # 3. Detecção de picos R
            peaks, properties = find_peaks(signal, height=std)
            has_peaks = len(peaks) > 2

            # Score combinado
            score = 0.4 * min(signal_range / 2, 1) + \
                   0.3 * smoothness + \
                   0.3 * (1 if has_peaks else 0)

            scores.append(score)

            if score < 0.5:
                issues.append(f"{lead_name}: Qualidade baixa")

        # Score geral
        overall_score = np.mean(scores)

        # Verificar consistência entre derivações
        if overall_score > 0.5:
            # Verificar se as derivações relacionadas são consistentes
            # Ex: II = I + III (lei de Einthoven)
            if signals.shape[0] >= 3:
                diff = signals[1] - (signals[0] + signals[2])
                consistency_error = np.mean(np.abs(diff))
                if consistency_error > 1:
                    overall_score *= 0.8
                    issues.append("Inconsistência entre derivações")

        return {
            'score': float(overall_score),
            'lead_scores': [float(s) for s in scores],
            'issues': issues,
            'usable': overall_score > self.config['quality_threshold']
        }

    def _show_image(self, image: np.ndarray, title: str = "Image"):
        """Mostra imagem para debug"""
        plt.figure(figsize=(10, 6))
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

    def visualize_extraction(self, file_path: str):
        """Visualiza o processo de extração para debug"""
        result = self.digitize(file_path)

        # Plotar sinais extraídos
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        axes = axes.flatten()

        for i in range(12):
            ax = axes[i]
            ax.plot(result['data'][i])
            ax.set_title(f"{self.lead_names[i]} (Score: {result['quality']['lead_scores'][i]:.2f})")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-3, 3)

        plt.tight_layout()
        plt.suptitle(f"ECG Digitalizado - Qualidade Geral: {result['quality']['score']:.2f}",
                    y=1.02)
        plt.show()

        return result

# Filtros do scipy.signal
from scipy import signal as signal_filters
