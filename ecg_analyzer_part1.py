#!/usr/bin/env python3
"""
ECG ANALYZER AVANÇADO - PARTE 1: PIPELINE DE PRÉ-PROCESSAMENTO ROBUSTO
Implementa correção de rotação, perspectiva, calibração e detecção de layout
"""

import numpy as np
import cv2
from scipy import ndimage, signal
from skimage import morphology, transform
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedECGPreprocessor:
    """Pipeline avançado de pré-processamento para ECGs digitalizados"""
    
    def __init__(self):
        self.standard_layouts = {
            '3x4': {'rows': 3, 'cols': 4, 'order': 'standard'},
            '6x2': {'rows': 6, 'cols': 2, 'order': 'vertical'},
            '12x1': {'rows': 12, 'cols': 1, 'order': 'single'},
            '4x3_limb_chest': {'rows': 4, 'cols': 3, 'order': 'mixed'}
        }
        
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Parâmetros de calibração padrão
        self.mm_per_mv = 10  # 10mm = 1mV (padrão)
        self.mm_per_sec = 25  # 25mm = 1s (padrão)
        self.grid_size_mm = 5  # Grade de 5mm
        
    def detect_and_correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """Detecta e corrige rotação automática usando Hough transform"""
        
        # Converter para escala de cinza se necessário
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detectar bordas
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detectar linhas usando Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            # Analisar ângulos das linhas
            angles = []
            for rho, theta in lines[:, 0]:
                # Converter para graus
                angle = np.degrees(theta) - 90
                # Normalizar para [-45, 45]
                if angle > 45:
                    angle -= 90
                elif angle < -45:
                    angle += 90
                angles.append(angle)
            
            # Calcular ângulo médio (remover outliers)
            angles = np.array(angles)
            median_angle = np.median(angles)
            mad = np.median(np.abs(angles - median_angle))
            filtered_angles = angles[np.abs(angles - median_angle) < 2 * mad]
            
            if len(filtered_angles) > 0:
                rotation_angle = np.mean(filtered_angles)
                
                # Aplicar rotação se significativa
                if abs(rotation_angle) > 0.5:
                    print(f"🔄 Rotação detectada: {rotation_angle:.2f}°")
                    rotated = ndimage.rotate(image, rotation_angle, reshape=True, 
                                           mode='constant', cval=255)
                    return rotated
        
        return image
    
    def detect_perspective_and_correct(self, image: np.ndarray) -> np.ndarray:
        """Detecta e corrige distorção de perspectiva"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Threshold adaptativo para encontrar papel
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Encontrar maior contorno (provavelmente o papel)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Aproximar para polígono
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) == 4:
                # Ordenar pontos
                pts = approx.reshape(4, 2)
                rect = self._order_points(pts)
                
                # Calcular dimensões do retângulo de destino
                width = max(
                    np.linalg.norm(rect[0] - rect[1]),
                    np.linalg.norm(rect[2] - rect[3])
                )
                height = max(
                    np.linalg.norm(rect[0] - rect[3]),
                    np.linalg.norm(rect[1] - rect[2])
                )
                
                # Definir pontos de destino
                dst = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype="float32")
                
                # Calcular matriz de perspectiva
                M = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
                
                # Aplicar transformação
                warped = cv2.warpPerspective(image, M, (int(width), int(height)))
                
                print("📐 Correção de perspectiva aplicada")
                return warped
        
        return image
    
    def _order_points(self, pts):
        """Ordena pontos no formato: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Soma e diferença para encontrar corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left
        
        return rect
    
    def detect_and_remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Remove artefatos de digitalização"""
        
        # Aplicar filtro bilateral para preservar bordas
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Remover pequenos objetos (poeira, manchas)
        if len(denoised.shape) == 3:
            gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        else:
            gray = denoised
        
        # Threshold adaptativo
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Operações morfológicas
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Remover componentes pequenos
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        
        # Filtrar por área
        min_area = 100
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                cleaned[labels == i] = 0
        
        return cleaned
    
    def detect_grid_and_calibrate(self, image: np.ndarray) -> Dict:
        """Detecta grade do ECG e calibra escalas"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detectar linhas da grade usando FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Encontrar picos no espectro (frequências da grade)
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        
        # Analisar projeções para encontrar espaçamento
        h_projection = np.sum(magnitude_spectrum[center_h-10:center_h+10, :], axis=0)
        v_projection = np.sum(magnitude_spectrum[:, center_w-10:center_w+10], axis=1)
        
        # Encontrar picos
        h_peaks, _ = signal.find_peaks(h_projection[center_w+10:], distance=5)
        v_peaks, _ = signal.find_peaks(v_projection[center_h+10:], distance=5)
        
        grid_spacing_pixels = {
            'horizontal': np.mean(np.diff(h_peaks)) if len(h_peaks) > 1 else 50,
            'vertical': np.mean(np.diff(v_peaks)) if len(v_peaks) > 1 else 50
        }
        
        # Detectar marcas de calibração (1mV)
        calibration_mark = self._detect_calibration_mark(gray)
        
        if calibration_mark:
            # Calibrar baseado na marca de 1mV
            pixels_per_mv = calibration_mark['height']
            pixels_per_mm = pixels_per_mv / self.mm_per_mv
        else:
            # Usar grade detectada
            pixels_per_mm = grid_spacing_pixels['vertical'] / self.grid_size_mm
            pixels_per_mv = pixels_per_mm * self.mm_per_mv
        
        # Calcular escala temporal
        pixels_per_sec = pixels_per_mm * self.mm_per_sec
        
        calibration = {
            'pixels_per_mv': pixels_per_mv,
            'pixels_per_mm': pixels_per_mm,
            'pixels_per_sec': pixels_per_sec,
            'grid_spacing': grid_spacing_pixels,
            'calibration_mark_found': calibration_mark is not None
        }
        
        print(f"📏 Calibração: {pixels_per_mm:.1f} pixels/mm, {pixels_per_mv:.1f} pixels/mV")
        
        return calibration
    
    def _detect_calibration_mark(self, image: np.ndarray) -> Optional[Dict]:
        """Detecta marca de calibração de 1mV"""
        
        # Procurar por retângulo característico de calibração
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Aproximar contorno
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:  # Retângulo
                x, y, w, h = cv2.boundingRect(contour)
                
                # Verificar proporções típicas de marca de calibração
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 0.8 and 20 < h < 100:
                    # Possível marca de calibração
                    return {
                        'x': x, 'y': y,
                        'width': w, 'height': h,
                        'area': w * h
                    }
        
        return None
    
    def detect_layout(self, image: np.ndarray) -> Dict:
        """Detecta layout automático do ECG"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detecção baseada em projeções
        h_projection = np.mean(gray, axis=1)
        v_projection = np.mean(gray, axis=0)
        
        # Suavizar projeções
        h_smooth = signal.savgol_filter(h_projection, 51, 3)
        v_smooth = signal.savgol_filter(v_projection, 51, 3)
        
        # Encontrar vales (separadores entre sinais)
        h_valleys = signal.find_peaks(-h_smooth, prominence=20)[0]
        v_valleys = signal.find_peaks(-v_smooth, prominence=20)[0]
        
        # Filtrar vales muito próximos
        h_valleys = self._filter_close_valleys(h_valleys, min_distance=50)
        v_valleys = self._filter_close_valleys(v_valleys, min_distance=50)
        
        # Determinar número de linhas e colunas
        num_rows = len(h_valleys) + 1
        num_cols = len(v_valleys) + 1
        
        # Identificar layout
        layout_key = f"{num_rows}x{num_cols}"
        
        if layout_key in self.standard_layouts:
            layout = self.standard_layouts[layout_key].copy()
            layout['detected'] = True
        else:
            # Layout customizado
            layout = {
                'rows': num_rows,
                'cols': num_cols,
                'order': 'custom',
                'detected': True
            }
        
        # Adicionar posições dos separadores
        layout['row_separators'] = h_valleys
        layout['col_separators'] = v_valleys
        
        print(f"📋 Layout detectado: {num_rows}x{num_cols}")
        
        return layout
    
    def _filter_close_valleys(self, valleys: np.ndarray, min_distance: int) -> np.ndarray:
        """Filtra vales muito próximos"""
        if len(valleys) == 0:
            return valleys
        
        filtered = [valleys[0]]
        for v in valleys[1:]:
            if v - filtered[-1] >= min_distance:
                filtered.append(v)
        
        return np.array(filtered)
    
    def preprocess_complete(self, image_path: str, debug: bool = False) -> Dict:
        """Pipeline completo de pré-processamento"""
        
        print(f"\n🔧 PRÉ-PROCESSAMENTO AVANÇADO: {image_path}")
        
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar: {image_path}")
        
        original = image.copy()
        
        # 1. Corrigir rotação
        image = self.detect_and_correct_rotation(image)
        
        # 2. Corrigir perspectiva
        image = self.detect_perspective_and_correct(image)
        
        # 3. Remover artefatos
        cleaned = self.detect_and_remove_artifacts(image)
        
        # 4. Detectar grade e calibrar
        calibration = self.detect_grid_and_calibrate(image)
        
        # 5. Detectar layout
        layout = self.detect_layout(cleaned)
        
        # 6. Extrair regiões individuais
        regions = self._extract_lead_regions(cleaned, layout)
        
        result = {
            'original': original,
            'preprocessed': image,
            'cleaned': cleaned,
            'calibration': calibration,
            'layout': layout,
            'regions': regions,
            'success': True
        }
        
        if debug:
            self._visualize_preprocessing(result)
        
        return result
    
    def _extract_lead_regions(self, image: np.ndarray, layout: Dict) -> Dict:
        """Extrai regiões individuais para cada derivação"""
        
        h, w = image.shape[:2]
        regions = {}
        
        # Calcular limites das células
        row_bounds = [0] + list(layout['row_separators']) + [h]
        col_bounds = [0] + list(layout['col_separators']) + [w]
        
        # Mapear derivações para posições baseado no layout
        if layout['order'] == 'standard':
            # Layout 3x4 padrão
            lead_positions = {
                'I': (0, 0), 'aVR': (0, 1), 'V1': (0, 2), 'V4': (0, 3),
                'II': (1, 0), 'aVL': (1, 1), 'V2': (1, 2), 'V5': (1, 3),
                'III': (2, 0), 'aVF': (2, 1), 'V3': (2, 2), 'V6': (2, 3)
            }
        elif layout['order'] == 'vertical':
            # Layout 6x2
            lead_positions = {
                'I': (0, 0), 'V1': (0, 1),
                'II': (1, 0), 'V2': (1, 1),
                'III': (2, 0), 'V3': (2, 1),
                'aVR': (3, 0), 'V4': (3, 1),
                'aVL': (4, 0), 'V5': (4, 1),
                'aVF': (5, 0), 'V6': (5, 1)
            }
        else:
            # Layout genérico
            lead_positions = {}
            idx = 0
            for r in range(layout['rows']):
                for c in range(layout['cols']):
                    if idx < len(self.lead_names):
                        lead_positions[self.lead_names[idx]] = (r, c)
                        idx += 1
        
        # Extrair cada região
        for lead, (row, col) in lead_positions.items():
            if row < len(row_bounds) - 1 and col < len(col_bounds) - 1:
                y1, y2 = row_bounds[row], row_bounds[row + 1]
                x1, x2 = col_bounds[col], col_bounds[col + 1]
                
                # Adicionar margem
                margin = 10
                y1 += margin
                y2 -= margin
                x1 += margin
                x2 -= margin
                
                regions[lead] = {
                    'image': image[y1:y2, x1:x2],
                    'bounds': (x1, y1, x2, y2),
                    'position': (row, col)
                }
        
        return regions
    
    def _visualize_preprocessing(self, result: Dict):
        """Visualiza resultados do pré-processamento"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(cv2.cvtColor(result['original'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Corrigido
        axes[0, 1].imshow(cv2.cvtColor(result['preprocessed'], cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Rotação + Perspectiva Corrigidas')
        axes[0, 1].axis('off')
        
        # Limpo
        axes[0, 2].imshow(result['cleaned'], cmap='gray')
        axes[0, 2].set_title('Artefatos Removidos')
        axes[0, 2].axis('off')
        
        # Layout detectado
        layout_vis = result['cleaned'].copy()
        for sep in result['layout']['row_separators']:
            cv2.line(layout_vis, (0, sep), (layout_vis.shape[1], sep), 128, 2)
        for sep in result['layout']['col_separators']:
            cv2.line(layout_vis, (sep, 0), (sep, layout_vis.shape[0]), 128, 2)
        
        axes[1, 0].imshow(layout_vis, cmap='gray')
        axes[1, 0].set_title(f"Layout: {result['layout']['rows']}x{result['layout']['cols']}")
        axes[1, 0].axis('off')
        
        # Calibração
        cal_text = f"Calibração:\n"
        cal_text += f"Pixels/mm: {result['calibration']['pixels_per_mm']:.1f}\n"
        cal_text += f"Pixels/mV: {result['calibration']['pixels_per_mv']:.1f}\n"
        cal_text += f"Marca 1mV: {'Sim' if result['calibration']['calibration_mark_found'] else 'Não'}"
        
        axes[1, 1].text(0.1, 0.5, cal_text, fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Informações de Calibração')
        axes[1, 1].axis('off')
        
        # Regiões extraídas
        if result['regions']:
            sample_lead = list(result['regions'].keys())[0]
            sample_region = result['regions'][sample_lead]['image']
            axes[1, 2].imshow(sample_region, cmap='gray')
            axes[1, 2].set_title(f'Região Extraída: {sample_lead}')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Pipeline de Pré-processamento Avançado', fontsize=16)
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    preprocessor = AdvancedECGPreprocessor()
    
    # Processar ECG
    # result = preprocessor.preprocess_complete('ecg_scan.pdf', debug=True)
    
    print("\n✅ Pipeline de pré-processamento robusto implementado!")
    print("Funcionalidades:")
    print("- Correção automática de rotação")
    print("- Correção de perspectiva")
    print("- Remoção de artefatos")
    print("- Calibração baseada em grade")
    print("- Detecção automática de layout")
