# ECG Analyzer com Digitizer Integrado
# Criado em: 2025-06-30

import numpy as np
import cv2
import tensorflow as tf
from pdf2image import convert_from_path
from scipy import signal, interpolate
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Union, Dict, List
import os
from datetime import datetime
import pandas as pd

class ECGAnalyzerStandalone:
    """Analisador ECG standalone com digitizer integrado"""

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path

        # Configurações
        self.target_length = 1000
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # Carregar nomes das classes do CSV
        self._load_class_names()

        # Carregar modelo se fornecido
        if model_path:
            self.load_model(model_path)

    def _load_class_names(self):
        """Carrega nomes das classes do arquivo SCP statements"""
        # Note: This path is relative to the location of the saved file
        scp_file = os.path.join(os.path.dirname(__file__), '../scp_statements.csv')
        if os.path.exists(scp_file):
            df = pd.read_csv(scp_file)
            self.class_names = df['description'].tolist()[:71]
        else:
            # Fallback if the CSV is not found relative to the script
            # This fallback will be problematic if the model requires 71 classes.
            # Consider adding a more robust method to find the class names CSV or
            # passing the list/path explicitly during initialization.
            print("⚠️ SCP statements CSV not found. Using generic class names.")
            self.class_names = [f'Diagnosis_{i}' for i in range(71)] # Escaped curly brace

    def load_model(self, model_path: str):
        """Carrega o modelo"""
        try:
            print(f"🧠 Carregando modelo: {model_path}") # Escaped curly braces
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            print("✅ Modelo carregado com sucesso!")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}") # Escaped curly braces
            return False

    def extract_ecg_from_image(self, file_path: str, debug: bool = False) -> Dict:
        """Extrai ECG de imagem/PDF"""
        print(f"\n📄 Processando: {os.path.basename(file_path)}") # Escaped curly braces

        # Carregar imagem
        image = self._load_file(file_path)
        if image is None:
            return {'success': False, 'error': 'Falha ao carregar arquivo'} # Escaped curly braces

        # Processar
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Extrair sinais (layout 3x4)
        ecg_signals = self._extract_signals(gray)

        # Avaliar qualidade
        quality = self._check_quality(ecg_signals)

        if debug:
            self._show_extraction(ecg_signals, quality)

        return { # Escaped curly braces
            'success': True,
            'signals': ecg_signals,
            'quality': quality
        }

    def analyze(self, file_path: str, show_signals: bool = False) -> Dict:
        """Análise completa do ECG"""
        # Extrair ECG
        extraction = self.extract_ecg_from_image(file_path, debug=show_signals)

        if not extraction['success']:
            return extraction

        if extraction['quality']['score'] < 0.3:
            return { # Escaped curly braces
                'success': False,
                'error': 'Qualidade muito baixa para análise',
                'quality': extraction['quality']
            }

        # Fazer predição se modelo carregado
        if self.model is None:
            return { # Escaped curly braces
                'success': True,
                'extraction': extraction,
                'message': 'ECG extraído mas modelo não carregado'
            }

        # Predição
        ecg_data = extraction['signals']
        predictions = self._predict(ecg_data)

        # Resultado
        result = self._create_result(predictions, extraction['quality'])

        # Mostrar resultado
        self._display_result(result)

        return result

    def _load_file(self, file_path: str) -> np.ndarray:
        """Carrega arquivo PDF ou imagem"""
        try:
            ext = os.path.splitext(file_path)[1].lower()

            if ext == '.pdf':
                pages = convert_from_path(file_path, dpi=300)
                if pages:
                    return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
            else:
                return cv2.imread(file_path)
        except Exception as e:
            print(f"Erro ao carregar: {e}") # Escaped curly braces
            return None

    def _extract_signals(self, gray_image: np.ndarray) -> np.ndarray:
        """Extrai 12 derivações"""
        h, w = gray_image.shape
        signals = np.zeros((12, self.target_length))

        # Layout 3x4
        rows, cols = 3, 4
        cell_h = h // rows
        cell_w = w // cols

        for i in range(12):
            row = i // cols
            col = i % cols

            # Região com margem
            y1 = row * cell_h + 20
            y2 = (row + 1) * cell_h - 20
            x1 = col * cell_w + 20
            x2 = (col + 1) * cell_w - 20

            region = gray_image[y1:y2, x1:x2]
            signal = self._extract_single_signal(region)

            # Reamostrar
            if len(signal) > 0:
                signals[i] = self._resample(signal)

        # Normalizar
        for i in range(12):
            if np.std(signals[i]) > 0:
                signals[i] = (signals[i] - np.mean(signals[i])) / np.std(signals[i])

        return signals

    def _extract_single_signal(self, region: np.ndarray) -> np.ndarray:
        """Extrai sinal de uma região"""
        # Binarizar
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Extrair linha central
        h, w = binary.shape
        signal = []

        for x in range(w):
            col = binary[:, x]
            points = np.where(col > 0)[0]

            if len(points) > 0:
                y = np.median(points)
                signal.append(h/2 - y)
            elif signal:
                signal.append(signal[-1])

        return np.array(signal)

    def _resample(self, signal: np.ndarray) -> np.ndarray:
        """Reamostra para tamanho fixo"""
        if len(signal) == 0:
            return np.zeros(self.target_length)

        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, self.target_length)

        return np.interp(x_new, x_old, signal)

    def _check_quality(self, signals: np.ndarray) -> Dict:
        """Verifica qualidade dos sinais"""
        scores = []

        for i, signal in enumerate(signals):
            std = np.std(signal)
            if std < 0.01:
                scores.append(0)
            else:
                # SNR simples
                noise = np.std(np.diff(signal))
                snr = std / (noise + 1e-10)
                scores.append(min(snr / 10, 1.0))

        return { # Escaped curly braces
            'score': np.mean(scores),
            'lead_scores': scores,
            'good_leads': sum(s > 0.5 for s in scores)
        }

    def _predict(self, ecg_data: np.ndarray) -> np.ndarray:
        """Faz predição com o modelo"""
        # Preparar entrada
        if ecg_data.ndim == 2:
            ecg_data = np.expand_dims(ecg_data, axis=0)

        if len(self.model.input_shape) == 4:
            ecg_data = np.expand_dims(ecg_data, axis=-1)

        return self.model.predict(ecg_data, verbose=0)

    def _create_result(self, predictions: np.ndarray, quality: Dict) -> Dict:
        """Cria resultado da análise"""
        pred = predictions[0]
        top_idx = np.argmax(pred)

        return { # Escaped curly braces
            'success': True,
            'diagnosis': self.class_names[top_idx],
            'confidence': float(pred[top_idx]),
            'all_predictions': { # Escaped curly braces
                self.class_names[i]: float(pred[i])
                for i in range(len(self.class_names))
            },
            'quality': quality
        }

    def _show_extraction(self, signals: np.ndarray, quality: Dict):
        """Mostra sinais extraídos"""
        fig, axes = plt.subplots(4, 3, figsize=(12, 8))
        axes = axes.flatten()

        for i in range(12):
            axes[i].plot(signals[i])
            axes[i].set_title(f"{self.lead_names[i]} ({quality['lead_scores'][i]:.2f})") # Escaped curly braces
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _display_result(self, result: Dict):
        """Exibe resultado"""
        print(f"\n✅ Diagnóstico: {result['diagnosis']}") # Escaped curly braces
        print(f"📊 Confiança: {result['confidence']:.1%}") # Escaped curly braces
        print(f"📈 Qualidade: {result['quality']['score']:.2f}") # Escaped curly braces

# ... (resto dos métodos da classe)
