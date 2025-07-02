#!/usr/bin/env python3
"""
ECG Extractor Aprimorado - Versão robusta para Google Colab
Resolve problemas de extração de derivações com múltiplas estratégias
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from pdf2image import convert_from_path
import os

class ImprovedECGExtractor:
    """Extrator de ECG aprimorado com múltiplas estratégias"""
    
    def __init__(self):
        self.target_length = 1000
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
    def extract_ecg_robust(self, file_path, debug=True):
        """Extração robusta com múltiplas estratégias"""
        
        print(f"🔍 Analisando: {os.path.basename(file_path)}")
        
        # 1. Carregar e preprocessar imagem
        image = self._load_image(file_path)
        if image is None:
            return {'success': False, 'error': 'Não foi possível carregar arquivo'}
        
        # 2. Múltiplas estratégias de preprocessamento
        strategies = [
            self._strategy_1_standard,
            self._strategy_2_adaptive,
            self._strategy_3_morphological,
            self._strategy_4_grid_detection
        ]
        
        best_result = None
        best_score = 0
        
        for i, strategy in enumerate(strategies):
            print(f"🧪 Testando estratégia {i+1}/4...")
            
            try:
                result = strategy(image.copy())
                score = self._evaluate_extraction_quality(result['signals'])
                
                print(f"   Qualidade: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_result['strategy'] = f"Estratégia {i+1}"
                    best_result['quality_score'] = score
                    
            except Exception as e:
                print(f"   ❌ Falhou: {e}")
                continue
        
        if best_result is None:
            return {'success': False, 'error': 'Todas as estratégias falharam'}
        
        # 3. Pós-processamento e limpeza
        best_result['signals'] = self._post_process_signals(best_result['signals'])
        
        # 4. Avaliação final
        final_quality = self._assess_quality_detailed(best_result['signals'])
        best_result['quality'] = final_quality
        
        if debug:
            self._visualize_extraction(image, best_result)
        
        print(f"✅ Melhor resultado: {best_result['strategy']}")
        print(f"📊 Qualidade final: {final_quality['score']:.3f}")
        print(f"🔢 Derivações válidas: {final_quality['good_leads']}/12")
        
        return {'success': True, **best_result}
    
    def _load_image(self, file_path):
        """Carrega imagem com otimizações"""
        try:
            if file_path.lower().endswith('.pdf'):
                # PDF com alta resolução
                pages = convert_from_path(file_path, dpi=600, first_page=1, last_page=1)
                image = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(file_path)
            
            # Redimensionar se muito grande (otimização)
            h, w = image.shape[:2]
            if max(h, w) > 3000:
                scale = 3000 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            return image
            
        except Exception as e:
            print(f"❌ Erro ao carregar imagem: {e}")
            return None
    
    def _strategy_1_standard(self, image):
        """Estratégia 1: Abordagem padrão melhorada"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Equalização adaptativa
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Filtro bilateral para reduzir ruído
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        signals = self._extract_grid_based(gray, 3, 4)
        return {'signals': signals, 'method': 'Standard + CLAHE'}
    
    def _strategy_2_adaptive(self, image):
        """Estratégia 2: Threshold adaptativo"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur para suavizar
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold adaptativo
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        signals = self._extract_from_binary(binary, 3, 4)
        return {'signals': signals, 'method': 'Adaptive Threshold'}
    
    def _strategy_3_morphological(self, image):
        """Estratégia 3: Operações morfológicas"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold OTSU
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Operações morfológicas para limpar
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        signals = self._extract_from_binary(binary, 3, 4)
        return {'signals': signals, 'method': 'Morphological'}
    
    def _strategy_4_grid_detection(self, image):
        """Estratégia 4: Detecção automática de grid"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar linhas para encontrar grid
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        # Se encontrou linhas suficientes, use-as para definir grid
        if lines is not None and len(lines) > 10:
            # Grid adaptativo baseado em linhas detectadas
            signals = self._extract_adaptive_grid(gray, lines)
        else:
            # Fallback para grid fixo
            signals = self._extract_grid_based(gray, 3, 4)
        
        return {'signals': signals, 'method': 'Grid Detection'}
    
    def _extract_grid_based(self, gray, rows, cols):
        """Extração baseada em grid fixo"""
        h, w = gray.shape
        signals = np.zeros((12, self.target_length))
        
        # Margens adaptativas baseadas no tamanho da imagem
        margin_h = max(20, h // 50)
        margin_w = max(20, w // 50)
        
        cell_h = h // rows
        cell_w = w // cols
        
        for i in range(min(12, rows * cols)):
            row = i // cols
            col = i % cols
            
            y1 = row * cell_h + margin_h
            y2 = (row + 1) * cell_h - margin_h
            x1 = col * cell_w + margin_w
            x2 = (col + 1) * cell_w - margin_w
            
            if y2 > y1 and x2 > x1:
                region = gray[y1:y2, x1:x2]
                signal = self._extract_signal_robust(region)
                
                if len(signal) > 0:
                    signals[i] = self._resample_signal(signal)
        
        return signals
    
    def _extract_from_binary(self, binary, rows, cols):
        """Extração de imagem já binarizada"""
        h, w = binary.shape
        signals = np.zeros((12, self.target_length))
        
        margin_h = max(10, h // 100)
        margin_w = max(10, w // 100)
        
        cell_h = h // rows
        cell_w = w // cols
        
        for i in range(min(12, rows * cols)):
            row = i // cols
            col = i % cols
            
            y1 = row * cell_h + margin_h
            y2 = (row + 1) * cell_h - margin_h
            x1 = col * cell_w + margin_w
            x2 = (col + 1) * cell_w - margin_w
            
            if y2 > y1 and x2 > x1:
                region = binary[y1:y2, x1:x2]
                signal = self._extract_signal_from_binary(region)
                
                if len(signal) > 0:
                    signals[i] = self._resample_signal(signal)
        
        return signals
    
    def _extract_adaptive_grid(self, gray, lines):
        """Grid adaptativo baseado em linhas detectadas"""
        # Implementação simplificada - usar grid padrão como fallback
        return self._extract_grid_based(gray, 3, 4)
    
    def _extract_signal_robust(self, region):
        """Extração robusta de sinal com múltiplas tentativas"""
        if region.size == 0:
            return np.array([])
        
        # Tentativa 1: Threshold adaptativo
        try:
            binary = cv2.adaptiveThreshold(
                region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 15, 10
            )
            signal1 = self._extract_signal_from_binary(binary)
            if len(signal1) > 100:  # Sinal válido
                return signal1
        except:
            pass
        
        # Tentativa 2: OTSU
        try:
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            signal2 = self._extract_signal_from_binary(binary)
            if len(signal2) > 100:
                return signal2
        except:
            pass
        
        # Tentativa 3: Threshold fixo
        try:
            mean_val = np.mean(region)
            _, binary = cv2.threshold(region, mean_val - 30, 255, cv2.THRESH_BINARY_INV)
            signal3 = self._extract_signal_from_binary(binary)
            if len(signal3) > 100:
                return signal3
        except:
            pass
        
        return np.array([])
    
    def _extract_signal_from_binary(self, binary):
        """Extrai sinal de região binarizada"""
        h, w = binary.shape
        signal = []
        
        for x in range(w):
            col = binary[:, x]
            white_pixels = np.where(col > 0)[0]
            
            if len(white_pixels) > 0:
                # Usar mediana para robustez
                y = np.median(white_pixels)
                signal.append(h/2 - y)  # Inverter coordenada Y
            elif signal:
                # Interpolação simples
                signal.append(signal[-1])
        
        return np.array(signal)
    
    def _resample_signal(self, signal):
        """Reamostra sinal com interpolação suave"""
        if len(signal) == 0:
            return np.zeros(self.target_length)
        
        # Remover outliers antes da reamostragem
        signal = self._remove_outliers(signal)
        
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, self.target_length)
        
        # Interpolação cúbica para suavidade
        f = interpolate.interp1d(x_old, signal, kind='cubic', 
                               bounds_error=False, fill_value='extrapolate')
        
        return f(x_new)
    
    def _remove_outliers(self, signal):
        """Remove outliers do sinal"""
        if len(signal) < 10:
            return signal
        
        q1, q3 = np.percentile(signal, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Substituir outliers por mediana
        median_val = np.median(signal)
        mask = (signal < lower_bound) | (signal > upper_bound)
        signal[mask] = median_val
        
        return signal
    
    def _post_process_signals(self, signals):
        """Pós-processamento dos sinais"""
        processed = np.zeros_like(signals)
        
        for i in range(12):
            signal = signals[i]
            
            # Filtro passa-baixa para suavizar
            b, a = signal.butter(3, 0.3, btype='low')
            signal = signal.filtfilt(b, a, signal)
            
            # Normalização robusta
            if np.std(signal) > 1e-6:
                signal = (signal - np.median(signal)) / (np.std(signal) + 1e-6)
            
            processed[i] = signal
        
        return processed
    
    def _evaluate_extraction_quality(self, signals):
        """Avalia qualidade da extração"""
        if signals is None or signals.size == 0:
            return 0
        
        scores = []
        for signal in signals:
            # Critérios de qualidade
            std_score = min(np.std(signal) / 2.0, 1.0)  # Variabilidade
            
            # Detecção de complexos QRS (picos)
            peaks, _ = signal.find_peaks(signal, height=0.1, distance=50)
            peak_score = min(len(peaks) / 20.0, 1.0)  # Esperado ~10-20 batimentos
            
            # Suavidade (menos ruído)
            diff_std = np.std(np.diff(signal))
            smooth_score = max(0, 1 - diff_std)
            
            lead_score = (std_score + peak_score + smooth_score) / 3
            scores.append(lead_score)
        
        return np.mean(scores)
    
    def _assess_quality_detailed(self, signals):
        """Avaliação detalhada de qualidade"""
        lead_scores = []
        
        for i, signal in enumerate(signals):
            if np.std(signal) < 0.01:
                score = 0.0
            else:
                # Múltiplos critérios
                std = np.std(signal)
                mean_abs = np.mean(np.abs(signal))
                
                # Detecção de picos
                peaks, _ = signal.find_peaks(np.abs(signal), height=0.1, distance=30)
                
                # SNR estimado
                noise_est = np.std(np.diff(signal))
                snr = std / (noise_est + 1e-10)
                
                # Score combinado
                score = min(1.0, (std + mean_abs + len(peaks)/10 + snr/10) / 4)
            
            lead_scores.append(score)
        
        return {
            'score': np.mean(lead_scores),
            'lead_scores': lead_scores,
            'good_leads': sum(s > 0.3 for s in lead_scores),
            'excellent_leads': sum(s > 0.7 for s in lead_scores)
        }
    
    def _visualize_extraction(self, original_image, result):
        """Visualiza resultado da extração"""
        signals = result['signals']
        quality = result['quality']
        
        # Plot dos sinais
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(12):
            color = 'green' if quality['lead_scores'][i] > 0.5 else 'red'
            axes[i].plot(signals[i], color=color, linewidth=1.5)
            axes[i].set_title(f"{self.lead_names[i]} (Q: {quality['lead_scores'][i]:.2f})")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(-3, 3)
        
        plt.suptitle(f"ECG Extraído - {result['method']}\n"
                    f"Qualidade: {quality['score']:.3f} | "
                    f"Derivações válidas: {quality['good_leads']}/12")
        plt.tight_layout()
        plt.show()
        
        # Mostrar imagem original
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Imagem Original do ECG")
        plt.axis('off')
        plt.show()

# Função de uso simples
def test_improved_extraction(file_path):
    """Testa o extrator aprimorado"""
    extractor = ImprovedECGExtractor()
    result = extractor.extract_ecg_robust(file_path, debug=True)
    
    if result['success']:
        print(f"\n✅ EXTRAÇÃO CONCLUÍDA!")
        print(f"📊 Qualidade: {result['quality']['score']:.3f}")
        print(f"🔢 Derivações válidas: {result['quality']['good_leads']}/12")
        print(f"⭐ Derivações excelentes: {result['quality']['excellent_leads']}/12")
        print(f"🛠️ Método: {result['method']}")
        
        return result['signals']
    else:
        print(f"❌ Falha: {result['error']}")
        return None

# Exemplo de uso:
# signals = test_improved_extraction('/path/to/your/ecg.pdf')