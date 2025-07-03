"""
Serviço de Digitalização de ECG
Extrai dados de eletrocardiograma a partir de imagens (JPG, PNG, PDF, etc.)
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import io
import base64
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from skimage import morphology, measure, filters
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ECGDigitizer:
    """Classe para digitalização de ECG a partir de imagens."""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf']
        self.grid_detection_params = {
            'min_line_length': 50,
            'max_line_gap': 10,
            'threshold': 100
        }
        self.signal_extraction_params = {
            'min_signal_width': 2,
            'max_signal_width': 10,
            'noise_threshold': 0.1
        }
    
    def process_ecg_image(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Processa imagem de ECG e extrai dados digitalizados.
        
        Args:
            image_data: Dados binários da imagem
            filename: Nome do arquivo
            
        Returns:
            Dict com dados extraídos e metadados
        """
        try:
            logger.info(f"Iniciando processamento de {filename}")
            
            # Verificar formato suportado
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Formato {file_ext} não suportado. Use: {self.supported_formats}")
            
            # Carregar e preprocessar imagem
            if file_ext == '.pdf':
                image = self._extract_image_from_pdf(image_data)
            else:
                image = self._load_image(image_data)
            
            # Pipeline de processamento
            processed_image = self._preprocess_image(image)
            grid_info = self._detect_grid(processed_image)
            leads_data = self._extract_ecg_signals(processed_image, grid_info)
            calibrated_data = self._calibrate_signals(leads_data, grid_info)
            
            # Metadados da extração
            metadata = {
                'filename': filename,
                'image_size': image.shape[:2],
                'grid_detected': grid_info['grid_detected'],
                'leads_found': len(calibrated_data),
                'sampling_rate_estimated': self._estimate_sampling_rate(grid_info),
                'processing_timestamp': datetime.now().isoformat(),
                'calibration_info': grid_info.get('calibration', {}),
                'quality_score': self._calculate_quality_score(calibrated_data)
            }
            
            result = {
                'success': True,
                'ecg_data': calibrated_data,
                'metadata': metadata,
                'preview_image': self._generate_preview(processed_image, leads_data)
            }
            
            logger.info(f"Processamento concluído: {len(calibrated_data)} derivações extraídas")
            return result
            
        except Exception as e:
            logger.error(f"Erro no processamento de {filename}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'filename': filename,
                'timestamp': datetime.now().isoformat()
            }
    
    def _load_image(self, image_data: bytes) -> np.ndarray:
        """Carrega imagem a partir de dados binários."""
        try:
            # Converter bytes para PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Converter para RGB se necessário
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Converter para numpy array
            image = np.array(pil_image)
            
            logger.info(f"Imagem carregada: {image.shape}")
            return image
            
        except Exception as e:
            raise ValueError(f"Erro ao carregar imagem: {str(e)}")
    
    def _extract_image_from_pdf(self, pdf_data: bytes) -> np.ndarray:
        """Extrai primeira página de PDF como imagem."""
        try:
            # Para PDFs, vamos usar uma abordagem simplificada
            # Em produção, usar pdf2image ou PyMuPDF
            logger.warning("Processamento de PDF simplificado - implementar pdf2image para produção")
            
            # Por enquanto, retornar erro informativo
            raise ValueError("Processamento de PDF requer biblioteca pdf2image. Use imagens JPG/PNG.")
            
        except Exception as e:
            raise ValueError(f"Erro ao processar PDF: {str(e)}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocessa imagem para melhorar detecção de ECG."""
        try:
            # Converter para escala de cinza
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Melhorar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Reduzir ruído
            denoised = cv2.medianBlur(enhanced, 3)
            
            # Binarização adaptativa para destacar linhas
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            logger.info("Preprocessamento de imagem concluído")
            return binary
            
        except Exception as e:
            raise ValueError(f"Erro no preprocessamento: {str(e)}")
    
    def _detect_grid(self, image: np.ndarray) -> Dict[str, Any]:
        """Detecta grade do papel de ECG."""
        try:
            # Detectar linhas horizontais e verticais usando Hough Transform
            lines = cv2.HoughLinesP(
                image, 
                rho=1, 
                theta=np.pi/180, 
                threshold=self.grid_detection_params['threshold'],
                minLineLength=self.grid_detection_params['min_line_length'],
                maxLineGap=self.grid_detection_params['max_line_gap']
            )
            
            if lines is None:
                logger.warning("Nenhuma linha detectada - processando sem grade")
                return {
                    'grid_detected': False,
                    'horizontal_lines': [],
                    'vertical_lines': [],
                    'calibration': {'mm_per_pixel_x': 1.0, 'mm_per_pixel_y': 1.0}
                }
            
            # Separar linhas horizontais e verticais
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(angle) < 10 or abs(angle) > 170:  # Linha horizontal
                    horizontal_lines.append(line[0])
                elif abs(abs(angle) - 90) < 10:  # Linha vertical
                    vertical_lines.append(line[0])
            
            # Calcular calibração baseada na grade
            calibration = self._calculate_calibration(horizontal_lines, vertical_lines)
            
            grid_info = {
                'grid_detected': len(horizontal_lines) > 5 and len(vertical_lines) > 5,
                'horizontal_lines': horizontal_lines,
                'vertical_lines': vertical_lines,
                'calibration': calibration
            }
            
            logger.info(f"Grade detectada: {len(horizontal_lines)} linhas horizontais, {len(vertical_lines)} verticais")
            return grid_info
            
        except Exception as e:
            logger.error(f"Erro na detecção de grade: {str(e)}")
            return {
                'grid_detected': False,
                'horizontal_lines': [],
                'vertical_lines': [],
                'calibration': {'mm_per_pixel_x': 1.0, 'mm_per_pixel_y': 1.0}
            }
    
    def _calculate_calibration(self, h_lines: List, v_lines: List) -> Dict[str, float]:
        """Calcula calibração baseada na grade detectada."""
        try:
            calibration = {'mm_per_pixel_x': 1.0, 'mm_per_pixel_y': 1.0}
            
            if len(h_lines) > 1:
                # Calcular espaçamento médio entre linhas horizontais
                h_spacings = []
                h_positions = sorted([line[1] for line in h_lines])
                for i in range(1, len(h_positions)):
                    h_spacings.append(h_positions[i] - h_positions[i-1])
                
                if h_spacings:
                    avg_h_spacing = np.mean(h_spacings)
                    # Assumir grade padrão de ECG: 1mm = pequeno quadrado
                    calibration['mm_per_pixel_y'] = 1.0 / avg_h_spacing
            
            if len(v_lines) > 1:
                # Calcular espaçamento médio entre linhas verticais
                v_spacings = []
                v_positions = sorted([line[0] for line in v_lines])
                for i in range(1, len(v_positions)):
                    v_spacings.append(v_positions[i] - v_positions[i-1])
                
                if v_spacings:
                    avg_v_spacing = np.mean(v_spacings)
                    calibration['mm_per_pixel_x'] = 1.0 / avg_v_spacing
            
            logger.info(f"Calibração calculada: {calibration}")
            return calibration
            
        except Exception as e:
            logger.error(f"Erro no cálculo de calibração: {str(e)}")
            return {'mm_per_pixel_x': 1.0, 'mm_per_pixel_y': 1.0}
    
    def _extract_ecg_signals(self, image: np.ndarray, grid_info: Dict) -> Dict[str, np.ndarray]:
        """Extrai sinais de ECG da imagem processada."""
        try:
            # Detectar contornos que podem ser traçados de ECG
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            leads_data = {}
            lead_count = 0
            
            # Filtrar contornos por tamanho e forma
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Muito pequeno
                    continue
                
                # Extrair pontos do contorno
                points = contour.reshape(-1, 2)
                
                # Ordenar pontos por coordenada x
                points = points[points[:, 0].argsort()]
                
                # Verificar se parece com um traçado de ECG
                if self._is_ecg_trace(points):
                    # Interpolar para obter sinal uniforme
                    signal_data = self._interpolate_signal(points)
                    
                    if len(signal_data) > 100:  # Sinal mínimo válido
                        lead_name = f"Lead_{lead_count + 1}"
                        leads_data[lead_name] = signal_data
                        lead_count += 1
                        
                        if lead_count >= 12:  # Máximo 12 derivações
                            break
            
            # Se não encontrou traçados, tentar método alternativo
            if not leads_data:
                leads_data = self._extract_signals_alternative(image)
            
            logger.info(f"Sinais extraídos: {len(leads_data)} derivações")
            return leads_data
            
        except Exception as e:
            logger.error(f"Erro na extração de sinais: {str(e)}")
            return {}
    
    def _is_ecg_trace(self, points: np.ndarray) -> bool:
        """Verifica se pontos formam um traçado de ECG válido."""
        try:
            if len(points) < 50:
                return False
            
            # Verificar se há variação suficiente em Y
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            if y_range < 20:  # Muito plano
                return False
            
            # Verificar se há progressão em X
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            if x_range < 100:  # Muito curto
                return False
            
            # Verificar densidade de pontos
            density = len(points) / x_range
            if density < 0.1 or density > 10:  # Densidade inadequada
                return False
            
            return True
            
        except Exception:
            return False
    
    def _interpolate_signal(self, points: np.ndarray) -> np.ndarray:
        """Interpola pontos para criar sinal uniforme."""
        try:
            # Remover pontos duplicados em x
            unique_indices = np.unique(points[:, 0], return_index=True)[1]
            unique_points = points[unique_indices]
            
            if len(unique_points) < 10:
                return np.array([])
            
            # Ordenar por x
            sorted_indices = np.argsort(unique_points[:, 0])
            sorted_points = unique_points[sorted_indices]
            
            # Interpolar para obter sinal uniforme
            x_min, x_max = sorted_points[0, 0], sorted_points[-1, 0]
            x_uniform = np.linspace(x_min, x_max, int(x_max - x_min))
            
            # Interpolação linear
            f = interp1d(sorted_points[:, 0], sorted_points[:, 1], 
                        kind='linear', fill_value='extrapolate')
            y_uniform = f(x_uniform)
            
            # Normalizar sinal
            y_normalized = (y_uniform - np.mean(y_uniform)) / (np.std(y_uniform) + 1e-8)
            
            return y_normalized
            
        except Exception as e:
            logger.error(f"Erro na interpolação: {str(e)}")
            return np.array([])
    
    def _extract_signals_alternative(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Método alternativo para extração de sinais."""
        try:
            leads_data = {}
            
            # Dividir imagem em regiões horizontais (assumindo layout padrão de ECG)
            height, width = image.shape
            num_leads = 4  # Tentar extrair 4 derivações principais
            
            for i in range(num_leads):
                # Definir região de interesse
                y_start = int(i * height / num_leads)
                y_end = int((i + 1) * height / num_leads)
                roi = image[y_start:y_end, :]
                
                # Encontrar linha principal na ROI
                signal_line = self._find_main_signal_line(roi)
                
                if len(signal_line) > 100:
                    lead_name = f"Lead_{i + 1}"
                    leads_data[lead_name] = signal_line
            
            return leads_data
            
        except Exception as e:
            logger.error(f"Erro no método alternativo: {str(e)}")
            return {}
    
    def _find_main_signal_line(self, roi: np.ndarray) -> np.ndarray:
        """Encontra linha principal de sinal na região de interesse."""
        try:
            # Projeção vertical para encontrar linha mais densa
            vertical_projection = np.sum(roi, axis=1)
            
            # Encontrar pico principal
            peak_y = np.argmax(vertical_projection)
            
            # Extrair linha ao redor do pico
            margin = 5
            y_start = max(0, peak_y - margin)
            y_end = min(roi.shape[0], peak_y + margin)
            
            signal_region = roi[y_start:y_end, :]
            
            # Projeção horizontal
            horizontal_projection = np.sum(signal_region, axis=0)
            
            # Suavizar e normalizar
            smoothed = signal.savgol_filter(horizontal_projection, 11, 3)
            normalized = (smoothed - np.mean(smoothed)) / (np.std(smoothed) + 1e-8)
            
            return normalized
            
        except Exception:
            return np.array([])
    
    def _calibrate_signals(self, leads_data: Dict[str, np.ndarray], 
                          grid_info: Dict) -> Dict[str, Dict[str, Any]]:
        """Calibra sinais extraídos usando informações da grade."""
        try:
            calibrated_data = {}
            calibration = grid_info.get('calibration', {})
            
            for lead_name, signal_data in leads_data.items():
                if len(signal_data) == 0:
                    continue
                
                # Aplicar calibração de amplitude (mV)
                # Assumir que 10mm = 1mV (padrão ECG)
                mm_per_pixel_y = calibration.get('mm_per_pixel_y', 1.0)
                mv_per_pixel = mm_per_pixel_y / 10.0
                
                # Converter para milivolts
                signal_mv = signal_data * mv_per_pixel
                
                # Aplicar calibração temporal
                mm_per_pixel_x = calibration.get('mm_per_pixel_x', 1.0)
                # Assumir velocidade padrão de 25mm/s
                time_per_pixel = mm_per_pixel_x / 25.0  # segundos por pixel
                
                # Calcular taxa de amostragem
                sampling_rate = 1.0 / time_per_pixel if time_per_pixel > 0 else 500.0
                
                # Limitar taxa de amostragem a valores razoáveis
                sampling_rate = np.clip(sampling_rate, 100, 2000)
                
                calibrated_data[lead_name] = {
                    'signal': signal_mv.tolist(),
                    'sampling_rate': float(sampling_rate),
                    'duration': len(signal_mv) * time_per_pixel,
                    'amplitude_range': [float(np.min(signal_mv)), float(np.max(signal_mv))],
                    'calibration_applied': True
                }
            
            logger.info(f"Calibração aplicada a {len(calibrated_data)} derivações")
            return calibrated_data
            
        except Exception as e:
            logger.error(f"Erro na calibração: {str(e)}")
            # Retornar dados não calibrados
            uncalibrated = {}
            for lead_name, signal_data in leads_data.items():
                if len(signal_data) > 0:
                    uncalibrated[lead_name] = {
                        'signal': signal_data.tolist(),
                        'sampling_rate': 500.0,  # Padrão
                        'duration': len(signal_data) / 500.0,
                        'amplitude_range': [float(np.min(signal_data)), float(np.max(signal_data))],
                        'calibration_applied': False
                    }
            return uncalibrated
    
    def _estimate_sampling_rate(self, grid_info: Dict) -> float:
        """Estima taxa de amostragem baseada na grade."""
        try:
            calibration = grid_info.get('calibration', {})
            mm_per_pixel_x = calibration.get('mm_per_pixel_x', 1.0)
            
            # Velocidade padrão de papel ECG: 25mm/s
            time_per_pixel = mm_per_pixel_x / 25.0
            sampling_rate = 1.0 / time_per_pixel if time_per_pixel > 0 else 500.0
            
            # Limitar a valores razoáveis
            return float(np.clip(sampling_rate, 100, 2000))
            
        except Exception:
            return 500.0  # Padrão
    
    def _calculate_quality_score(self, calibrated_data: Dict) -> float:
        """Calcula score de qualidade da extração."""
        try:
            if not calibrated_data:
                return 0.0
            
            scores = []
            
            for lead_data in calibrated_data.values():
                signal = np.array(lead_data['signal'])
                
                if len(signal) == 0:
                    scores.append(0.0)
                    continue
                
                # Fatores de qualidade
                length_score = min(len(signal) / 1000, 1.0)  # Preferir sinais longos
                amplitude_score = min(np.std(signal) / 0.5, 1.0)  # Variação adequada
                continuity_score = 1.0 - np.sum(np.isnan(signal)) / len(signal)  # Sem NaN
                
                lead_score = (length_score + amplitude_score + continuity_score) / 3.0
                scores.append(lead_score)
            
            overall_score = np.mean(scores) if scores else 0.0
            return float(np.clip(overall_score, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Score neutro em caso de erro
    
    def _generate_preview(self, processed_image: np.ndarray, 
                         leads_data: Dict[str, np.ndarray]) -> str:
        """Gera imagem de preview da extração."""
        try:
            # Criar figura com subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Mostrar imagem processada
            axes[0].imshow(processed_image, cmap='gray')
            axes[0].set_title('Imagem Processada')
            axes[0].axis('off')
            
            # Mostrar sinais extraídos
            if leads_data:
                for i, (lead_name, signal) in enumerate(leads_data.items()):
                    if i >= 4:  # Limitar a 4 derivações no preview
                        break
                    axes[1].plot(signal, label=lead_name, alpha=0.7)
                
                axes[1].set_title('Sinais ECG Extraídos')
                axes[1].set_xlabel('Amostras')
                axes[1].set_ylabel('Amplitude')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'Nenhum sinal extraído', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Sinais ECG Extraídos')
            
            # Converter para base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            preview_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return preview_b64
            
        except Exception as e:
            logger.error(f"Erro ao gerar preview: {str(e)}")
            return ""


# Instância global do digitalizador
ecg_digitizer = ECGDigitizer()


def process_ecg_image_file(image_data: bytes, filename: str) -> Dict[str, Any]:
    """
    Função de conveniência para processar arquivo de imagem ECG.
    
    Args:
        image_data: Dados binários da imagem
        filename: Nome do arquivo
        
    Returns:
        Dict com resultados da digitalização
    """
    return ecg_digitizer.process_ecg_image(image_data, filename)

