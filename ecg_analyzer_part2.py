#!/usr/bin/env python3
"""
ECG ANALYZER AVANÇADO - PARTE 2: DELINEAÇÃO DE ONDAS
Implementa detecção precisa de ondas P, QRS, T e medição de intervalos
"""

import numpy as np
from scipy import signal, interpolate
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
import pywt
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class AdvancedWaveDelineator:
    """Sistema avançado de delineação de ondas ECG"""
    
    def __init__(self, sampling_rate: float = 500):
        self.fs = sampling_rate
        self.ms_to_samples = self.fs / 1000
        
        # Parâmetros fisiológicos normais (em ms)
        self.normal_ranges = {
            'P_duration': (80, 120),
            'PR_interval': (120, 200),
            'QRS_duration': (80, 120),
            'QT_interval': (350, 450),  # Varia com FC
            'QTc_interval': (350, 450),
            'RR_interval': (600, 1000)  # 60-100 bpm
        }
        
        # Configurações de filtros
        self.filters = {
            'baseline': {'low': 0.5, 'high': None, 'order': 4},
            'powerline': {'low': 48, 'high': 52, 'order': 4},
            'muscle': {'low': None, 'high': 40, 'order': 4},
            'p_wave': {'low': 0.5, 'high': 15, 'order': 4},
            'qrs': {'low': 8, 'high': 20, 'order': 4},
            't_wave': {'low': 0.5, 'high': 10, 'order': 4}
        }
    
    def preprocess_signal(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Pré-processamento avançado do sinal"""
        
        # Remover baseline wander usando wavelets
        signal_clean = self._remove_baseline_wavelet(ecg_signal)
        
        # Filtrar ruído de linha (50/60 Hz)
        b, a = signal.iirnotch(50, 30, self.fs)
        signal_clean = filtfilt(b, a, signal_clean)
        
        # Filtro passa-banda geral
        b, a = butter(4, [0.5, 45], btype='band', fs=self.fs)
        signal_clean = filtfilt(b, a, signal_clean)
        
        return signal_clean
    
    def _remove_baseline_wavelet(self, signal_ecg: np.ndarray) -> np.ndarray:
        """Remove baseline drift usando decomposição wavelet"""
        
        # Decomposição wavelet
        coeffs = pywt.wavedec(signal_ecg, 'db4', level=9)
        
        # Zerar coeficientes de baixa frequência (baseline)
        coeffs[0] = np.zeros_like(coeffs[0])
        coeffs[1] = np.zeros_like(coeffs[1])
        
        # Reconstruir sinal
        signal_clean = pywt.waverec(coeffs, 'db4')
        
        # Ajustar comprimento se necessário
        if len(signal_clean) > len(signal_ecg):
            signal_clean = signal_clean[:len(signal_ecg)]
        
        return signal_clean
    
    def detect_r_peaks_advanced(self, ecg_signal: np.ndarray) -> Dict:
        """Detecção avançada de picos R usando múltiplos métodos"""
        
        # Método 1: Pan-Tompkins modificado
        r_peaks_pt = self._pan_tompkins_detector(ecg_signal)
        
        # Método 2: Baseado em wavelets
        r_peaks_wt = self._wavelet_detector(ecg_signal)
        
        # Método 3: Baseado em derivadas
        r_peaks_der = self._derivative_detector(ecg_signal)
        
        # Fusão dos resultados
        r_peaks = self._fuse_detections([r_peaks_pt, r_peaks_wt, r_peaks_der])
        
        # Refinar detecções
        r_peaks = self._refine_r_peaks(ecg_signal, r_peaks)
        
        # Calcular métricas de qualidade
        quality = self._assess_r_detection_quality(r_peaks)
        
        return {
            'positions': r_peaks,
            'count': len(r_peaks),
            'quality': quality,
            'methods': {
                'pan_tompkins': len(r_peaks_pt),
                'wavelet': len(r_peaks_wt),
                'derivative': len(r_peaks_der)
            }
        }
    
    def _pan_tompkins_detector(self, signal_ecg: np.ndarray) -> np.ndarray:
        """Implementação do algoritmo Pan-Tompkins"""
        
        # 1. Filtro passa-banda
        b, a = butter(4, [5, 15], btype='band', fs=self.fs)
        filtered = filtfilt(b, a, signal_ecg)
        
        # 2. Derivada
        diff = np.diff(filtered)
        
        # 3. Elevar ao quadrado
        squared = diff ** 2
        
        # 4. Integração com janela móvel
        window_size = int(0.150 * self.fs)  # 150ms
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # 5. Encontrar picos
        threshold = 0.3 * np.max(integrated)
        min_distance = int(0.250 * self.fs)  # 250ms (240 bpm máx)
        
        peaks, properties = find_peaks(integrated, 
                                     height=threshold,
                                     distance=min_distance)
        
        return peaks
    
    def _wavelet_detector(self, signal_ecg: np.ndarray) -> np.ndarray:
        """Detecção usando transformada wavelet"""
        
        # Decomposição wavelet
        coeffs = pywt.wavedec(signal_ecg, 'db4', level=4)
        
        # Usar níveis 3 e 4 (frequências do QRS)
        d3 = coeffs[-3]
        d4 = coeffs[-4]
        
        # Reconstruir apenas com componentes QRS
        coeffs_qrs = [np.zeros_like(c) for c in coeffs]
        coeffs_qrs[-3] = d3
        coeffs_qrs[-4] = d4
        
        qrs_enhanced = pywt.waverec(coeffs_qrs, 'db4')
        
        # Detectar picos
        qrs_squared = qrs_enhanced ** 2
        threshold = 0.4 * np.max(qrs_squared)
        min_distance = int(0.250 * self.fs)
        
        peaks, _ = find_peaks(qrs_squared[:len(signal_ecg)], 
                            height=threshold,
                            distance=min_distance)
        
        return peaks
    
    def _derivative_detector(self, signal_ecg: np.ndarray) -> np.ndarray:
        """Detecção baseada em derivadas"""
        
        # Primeira e segunda derivadas
        d1 = np.gradient(signal_ecg)
        d2 = np.gradient(d1)
        
        # Combinar informações
        combined = np.abs(d1) + 0.5 * np.abs(d2)
        
        # Suavizar
        window_size = int(0.020 * self.fs)  # 20ms
        smoothed = savgol_filter(combined, window_size, 3)
        
        # Detectar picos
        threshold = 0.5 * np.max(smoothed)
        min_distance = int(0.250 * self.fs)
        
        peaks, _ = find_peaks(smoothed, 
                            height=threshold,
                            distance=min_distance)
        
        return peaks
    
    def _fuse_detections(self, detections: List[np.ndarray], 
                        tolerance_ms: float = 50) -> np.ndarray:
        """Fusão de múltiplas detecções"""
        
        tolerance_samples = int(tolerance_ms * self.ms_to_samples)
        
        # Combinar todas detecções
        all_peaks = np.concatenate(detections)
        all_peaks = np.sort(all_peaks)
        
        # Agrupar picos próximos
        fused_peaks = []
        i = 0
        
        while i < len(all_peaks):
            # Encontrar picos dentro da tolerância
            cluster = [all_peaks[i]]
            j = i + 1
            
            while j < len(all_peaks) and all_peaks[j] - all_peaks[i] < tolerance_samples:
                cluster.append(all_peaks[j])
                j += 1
            
            # Usar mediana do cluster se tiver suporte de múltiplos detectores
            if len(cluster) >= 2:
                fused_peaks.append(int(np.median(cluster)))
            
            i = j
        
        return np.array(fused_peaks)
    
    def _refine_r_peaks(self, signal_ecg: np.ndarray, r_peaks: np.ndarray) -> np.ndarray:
        """Refina posições dos picos R"""
        
        refined_peaks = []
        window = int(0.050 * self.fs)  # 50ms
        
        for peak in r_peaks:
            # Janela ao redor do pico
            start = max(0, peak - window)
            end = min(len(signal_ecg), peak + window)
            
            # Encontrar máximo local
            local_max = np.argmax(signal_ecg[start:end])
            refined_peak = start + local_max
            
            refined_peaks.append(refined_peak)
        
        return np.array(refined_peaks)
    
    def _assess_r_detection_quality(self, r_peaks: np.ndarray) -> Dict:
        """Avalia qualidade da detecção de R"""
        
        if len(r_peaks) < 2:
            return {'score': 0, 'regularity': 0, 'plausibility': 0}
        
        # Intervalos RR
        rr_intervals = np.diff(r_peaks)
        
        # Regularidade (baixa variabilidade)
        regularity = 1 - (np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-6))
        regularity = np.clip(regularity, 0, 1)
        
        # Plausibilidade (FC entre 30-200 bpm)
        heart_rates = 60 * self.fs / rr_intervals
        plausible = np.sum((heart_rates >= 30) & (heart_rates <= 200)) / len(heart_rates)
        
        # Score geral
        score = 0.7 * regularity + 0.3 * plausible
        
        return {
            'score': score,
            'regularity': regularity,
            'plausibility': plausible,
            'mean_hr': np.mean(heart_rates),
            'std_hr': np.std(heart_rates)
        }
    
    def detect_p_waves(self, signal_ecg: np.ndarray, r_peaks: np.ndarray) -> Dict:
        """Detecta ondas P antes dos complexos QRS"""
        
        # Filtrar para realçar ondas P
        b, a = butter(4, [0.5, 15], btype='band', fs=self.fs)
        p_filtered = filtfilt(b, a, signal_ecg)
        
        p_waves = []
        p_onsets = []
        p_offsets = []
        
        for i, r_peak in enumerate(r_peaks[1:], 1):  # Skip first R
            # Janela de busca: 50-300ms antes do R
            search_start = r_peak - int(0.300 * self.fs)
            search_end = r_peak - int(0.050 * self.fs)
            
            if search_start < 0:
                continue
            
            # Região de busca
            search_region = p_filtered[search_start:search_end]
            
            if len(search_region) < 10:
                continue
            
            # Encontrar máximo local (pico da onda P)
            p_candidates, properties = find_peaks(search_region, 
                                                prominence=0.05 * np.std(p_filtered))
            
            if len(p_candidates) > 0:
                # Escolher candidato mais próximo do R
                p_idx = p_candidates[-1]
                p_peak = search_start + p_idx
                
                # Encontrar onset e offset
                onset, offset = self._find_wave_boundaries(p_filtered, p_peak, 
                                                         window_ms=100)
                
                p_waves.append(p_peak)
                p_onsets.append(onset)
                p_offsets.append(offset)
        
        return {
            'peaks': np.array(p_waves),
            'onsets': np.array(p_onsets),
            'offsets': np.array(p_offsets),
            'count': len(p_waves),
            'detection_rate': len(p_waves) / (len(r_peaks) - 1) if len(r_peaks) > 1 else 0
        }
    
    def detect_t_waves(self, signal_ecg: np.ndarray, r_peaks: np.ndarray) -> Dict:
        """Detecta ondas T após complexos QRS"""
        
        # Filtrar para realçar ondas T
        b, a = butter(4, [0.5, 10], btype='band', fs=self.fs)
        t_filtered = filtfilt(b, a, signal_ecg)
        
        t_waves = []
        t_onsets = []
        t_offsets = []
        t_morphologies = []
        
        for i in range(len(r_peaks) - 1):
            # Janela de busca: 100-400ms após o R
            search_start = r_peaks[i] + int(0.100 * self.fs)
            search_end = min(r_peaks[i] + int(0.400 * self.fs), 
                           r_peaks[i+1] - int(0.100 * self.fs))
            
            if search_end <= search_start:
                continue
            
            # Região de busca
            search_region = t_filtered[search_start:search_end]
            
            if len(search_region) < 20:
                continue
            
            # Detectar pico (positivo ou negativo)
            pos_peaks, pos_props = find_peaks(search_region, 
                                             prominence=0.1 * np.std(t_filtered))
            neg_peaks, neg_props = find_peaks(-search_region, 
                                             prominence=0.1 * np.std(t_filtered))
            
            # Escolher o mais proeminente
            if len(pos_peaks) > 0 and len(neg_peaks) > 0:
                if pos_props['prominences'][0] > neg_props['prominences'][0]:
                    t_idx = pos_peaks[0]
                    morphology = 'positive'
                else:
                    t_idx = neg_peaks[0]
                    morphology = 'negative'
            elif len(pos_peaks) > 0:
                t_idx = pos_peaks[0]
                morphology = 'positive'
            elif len(neg_peaks) > 0:
                t_idx = neg_peaks[0]
                morphology = 'negative'
            else:
                continue
            
            t_peak = search_start + t_idx
            
            # Encontrar onset e offset
            onset, offset = self._find_wave_boundaries(t_filtered, t_peak, 
                                                     window_ms=200)
            
            t_waves.append(t_peak)
            t_onsets.append(onset)
            t_offsets.append(offset)
            t_morphologies.append(morphology)
        
        return {
            'peaks': np.array(t_waves),
            'onsets': np.array(t_onsets),
            'offsets': np.array(t_offsets),
            'morphologies': t_morphologies,
            'count': len(t_waves),
            'detection_rate': len(t_waves) / (len(r_peaks) - 1) if len(r_peaks) > 1 else 0
        }
    
    def detect_qrs_boundaries(self, signal_ecg: np.ndarray, r_peaks: np.ndarray) -> Dict:
        """Detecta limites dos complexos QRS"""
        
        # Filtrar para QRS
        b, a = butter(4, [8, 20], btype='band', fs=self.fs)
        qrs_filtered = filtfilt(b, a, signal_ecg)
        
        q_points = []
        s_points = []
        qrs_onsets = []
        qrs_offsets = []
        
        for r_peak in r_peaks:
            # Buscar Q (antes de R)
            q_window_start = max(0, r_peak - int(0.080 * self.fs))
            q_region = signal_ecg[q_window_start:r_peak]
            
            if len(q_region) > 5:
                q_idx = np.argmin(q_region)
                q_point = q_window_start + q_idx
                q_points.append(q_point)
                
                # Onset do QRS
                onset = self._find_qrs_onset(qrs_filtered, q_point, r_peak)
                qrs_onsets.append(onset)
            
            # Buscar S (depois de R)
            s_window_end = min(len(signal_ecg), r_peak + int(0.080 * self.fs))
            s_region = signal_ecg[r_peak:s_window_end]
            
            if len(s_region) > 5:
                s_idx = np.argmin(s_region)
                s_point = r_peak + s_idx
                s_points.append(s_point)
                
                # Offset do QRS
                offset = self._find_qrs_offset(qrs_filtered, r_peak, s_point)
                qrs_offsets.append(offset)
        
        return {
            'q_points': np.array(q_points),
            's_points': np.array(s_points),
            'onsets': np.array(qrs_onsets),
            'offsets': np.array(qrs_offsets),
            'durations': np.array(qrs_offsets) - np.array(qrs_onsets) if qrs_onsets else np.array([])
        }
    
    def _find_wave_boundaries(self, signal_ecg: np.ndarray, peak: int, 
                            window_ms: float) -> Tuple[int, int]:
        """Encontra onset e offset de uma onda"""
        
        window = int(window_ms * self.ms_to_samples)
        
        # Janela ao redor do pico
        start = max(0, peak - window)
        end = min(len(signal_ecg), peak + window)
        
        # Derivada do sinal
        region = signal_ecg[start:end]
        if len(region) < 5:
            return peak, peak
        
        derivative = np.gradient(region)
        
        # Onset: onde a derivada começa a aumentar significativamente
        peak_local = peak - start
        onset_search = derivative[:peak_local]
        if len(onset_search) > 0:
            onset_idx = np.where(np.abs(onset_search) < 0.1 * np.max(np.abs(derivative)))[0]
            onset = start + onset_idx[-1] if len(onset_idx) > 0 else start
        else:
            onset = start
        
        # Offset: onde a derivada volta a ser pequena
        offset_search = derivative[peak_local:]
        if len(offset_search) > 0:
            offset_idx = np.where(np.abs(offset_search) < 0.1 * np.max(np.abs(derivative)))[0]
            offset = peak + offset_idx[0] if len(offset_idx) > 0 else end
        else:
            offset = end
        
        return onset, offset
    
    def _find_qrs_onset(self, signal_ecg: np.ndarray, q_point: int, r_peak: int) -> int:
        """Encontra onset específico do QRS"""
        
        # Buscar ponto de inflexão antes do Q
        search_start = max(0, q_point - int(0.040 * self.fs))
        region = signal_ecg[search_start:q_point]
        
        if len(region) < 5:
            return q_point
        
        # Segunda derivada
        d2 = np.gradient(np.gradient(region))
        
        # Ponto onde d2 cruza zero
        zero_crossings = np.where(np.diff(np.sign(d2)))[0]
        
        if len(zero_crossings) > 0:
            return search_start + zero_crossings[-1]
        
        return q_point
    
    def _find_qrs_offset(self, signal_ecg: np.ndarray, r_peak: int, s_point: int) -> int:
        """Encontra offset específico do QRS"""
        
        # Buscar ponto de inflexão após o S
        search_end = min(len(signal_ecg), s_point + int(0.040 * self.fs))
        region = signal_ecg[s_point:search_end]
        
        if len(region) < 5:
            return s_point
        
        # Segunda derivada
        d2 = np.gradient(np.gradient(region))
        
        # Ponto onde d2 cruza zero
        zero_crossings = np.where(np.diff(np.sign(d2)))[0]
        
        if len(zero_crossings) > 0:
            return s_point + zero_crossings[0]
        
        return s_point
    
    def calculate_intervals(self, waves: Dict) -> Dict:
        """Calcula todos os intervalos importantes"""
        
        intervals = {}
        
        # PR interval
        if len(waves['p_waves']['onsets']) > 0 and len(waves['qrs']['onsets']) > 0:
            # Parear ondas P com QRS subsequentes
            pr_intervals = []
            for p_onset in waves['p_waves']['onsets']:
                # Encontrar próximo QRS
                qrs_after_p = waves['qrs']['onsets'][waves['qrs']['onsets'] > p_onset]
                if len(qrs_after_p) > 0:
                    pr = (qrs_after_p[0] - p_onset) / self.ms_to_samples
                    pr_intervals.append(pr)
            
            intervals['PR'] = {
                'values': np.array(pr_intervals),
                'mean': np.mean(pr_intervals) if pr_intervals else 0,
                'std': np.std(pr_intervals) if pr_intervals else 0
            }
        
        # QRS duration
        if len(waves['qrs']['durations']) > 0:
            qrs_durations = waves['qrs']['durations'] / self.ms_to_samples
            intervals['QRS'] = {
                'values': qrs_durations,
                'mean': np.mean(qrs_durations),
                'std': np.std(qrs_durations)
            }
        
        # QT interval
        if len(waves['qrs']['onsets']) > 0 and len(waves['t_waves']['offsets']) > 0:
            qt_intervals = []
            for i, qrs_onset in enumerate(waves['qrs']['onsets']):
                # Encontrar T offset correspondente
                t_after_qrs = waves['t_waves']['offsets'][waves['t_waves']['offsets'] > qrs_onset]
                if len(t_after_qrs) > 0 and i < len(waves['r_peaks']['positions']) - 1:
                    # Verificar se T está antes do próximo QRS
                    if i + 1 < len(waves['qrs']['onsets']):
                        next_qrs = waves['qrs']['onsets'][i + 1]
                        if t_after_qrs[0] < next_qrs:
                            qt = (t_after_qrs[0] - qrs_onset) / self.ms_to_samples
                            qt_intervals.append(qt)
            
            intervals['QT'] = {
                'values': np.array(qt_intervals),
                'mean': np.mean(qt_intervals) if qt_intervals else 0,
                'std': np.std(qt_intervals) if qt_intervals else 0
            }
            
            # QTc (Bazett's formula)
            if 'RR' in intervals and len(qt_intervals) > 0:
                mean_rr = np.mean(intervals['RR']['values'])
                qtc_values = np.array(qt_intervals) / np.sqrt(mean_rr / 1000)
                intervals['QTc'] = {
                    'values': qtc_values,
                    'mean': np.mean(qtc_values),
                    'std': np.std(qtc_values)
                }
        
        # RR intervals
        if len(waves['r_peaks']['positions']) > 1:
            rr_intervals = np.diff(waves['r_peaks']['positions']) / self.ms_to_samples
            intervals['RR'] = {
                'values': rr_intervals,
                'mean': np.mean(rr_intervals),
                'std': np.std(rr_intervals),
                'heart_rate': 60000 / np.mean(rr_intervals)
            }
        
        return intervals
    
    def delineate_complete(self, ecg_signal: np.ndarray, lead_name: str = 'II') -> Dict:
        """Delineação completa do sinal ECG"""
        
        print(f"\n🔍 Delineando sinal da derivação {lead_name}")
        
        # Pré-processar
        clean_signal = self.preprocess_signal(ecg_signal)
        
        # Detectar R peaks
        r_detection = self.detect_r_peaks_advanced(clean_signal)
        
        if r_detection['count'] < 3:
            return {
                'success': False,
                'error': 'Poucos complexos QRS detectados',
                'r_count': r_detection['count']
            }
        
        # Detectar outras ondas
        waves = {
            'r_peaks': r_detection,
            'p_waves': self.detect_p_waves(clean_signal, r_detection['positions']),
            't_waves': self.detect_t_waves(clean_signal, r_detection['positions']),
            'qrs': self.detect_qrs_boundaries(clean_signal, r_detection['positions'])
        }
        
        # Calcular intervalos
        intervals = self.calculate_intervals(waves)
        
        # Avaliar qualidade geral
        quality = self._assess_delineation_quality(waves, intervals)
        
        result = {
            'success': True,
            'signal': clean_signal,
            'waves': waves,
            'intervals': intervals,
            'quality': quality,
            'lead': lead_name
        }
        
        return result
    
    def _assess_delineation_quality(self, waves: Dict, intervals: Dict) -> Dict:
        """Avalia qualidade da delineação"""
        
        # Taxa de detecção de cada onda
        p_rate = waves['p_waves']['detection_rate']
        t_rate = waves['t_waves']['detection_rate']
        
        # Plausibilidade dos intervalos
        plausibility_scores = []
        
        if 'PR' in intervals and len(intervals['PR']['values']) > 0:
            pr_mean = intervals['PR']['mean']
            pr_ok = self.normal_ranges['PR_interval'][0] <= pr_mean <= self.normal_ranges['PR_interval'][1]
            plausibility_scores.append(1.0 if pr_ok else 0.5)
        
        if 'QRS' in intervals and len(intervals['QRS']['values']) > 0:
            qrs_mean = intervals['QRS']['mean']
            qrs_ok = self.normal_ranges['QRS_duration'][0] <= qrs_mean <= self.normal_ranges['QRS_duration'][1]
            plausibility_scores.append(1.0 if qrs_ok else 0.5)
        
        if 'QTc' in intervals and len(intervals['QTc']['values']) > 0:
            qtc_mean = intervals['QTc']['mean']
            qtc_ok = self.normal_ranges['QTc_interval'][0] <= qtc_mean <= self.normal_ranges['QTc_interval'][1]
            plausibility_scores.append(1.0 if qtc_ok else 0.5)
        
        # Score geral
        overall_score = np.mean([
            waves['r_peaks']['quality']['score'],
            p_rate,
            t_rate,
            np.mean(plausibility_scores) if plausibility_scores else 0.5
        ])
        
        return {
            'overall_score': overall_score,
            'r_quality': waves['r_peaks']['quality']['score'],
            'p_detection_rate': p_rate,
            't_detection_rate': t_rate,
            'interval_plausibility': np.mean(plausibility_scores) if plausibility_scores else 0.5
        }
    
    def visualize_delineation(self, result: Dict, save_path: Optional[str] = None):
        """Visualiza resultado da delineação"""
        
        if not result['success']:
            print("❌ Não é possível visualizar - delineação falhou")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        signal_ecg = result['signal']
        waves = result['waves']
        
        # Plot do sinal com marcações
        time = np.arange(len(signal_ecg)) / self.fs
        
        ax1.plot(time, signal_ecg, 'b-', linewidth=0.5, label='ECG')
        
        # Marcar ondas
        if len(waves['r_peaks']['positions']) > 0:
            ax1.plot(time[waves['r_peaks']['positions']], 
                    signal_ecg[waves['r_peaks']['positions']], 
                    'ro', markersize=8, label='R peaks')
        
        if len(waves['p_waves']['peaks']) > 0:
            ax1.plot(time[waves['p_waves']['peaks']], 
                    signal_ecg[waves['p_waves']['peaks']], 
                    'go', markersize=6, label='P waves')
        
        if len(waves['t_waves']['peaks']) > 0:
            ax1.plot(time[waves['t_waves']['peaks']], 
                    signal_ecg[waves['t_waves']['peaks']], 
                    'mo', markersize=6, label='T waves')
        
        if len(waves['qrs']['q_points']) > 0:
            ax1.plot(time[waves['qrs']['q_points']], 
                    signal_ecg[waves['qrs']['q_points']], 
                    'co', markersize=4, label='Q points')
        
        if len(waves['qrs']['s_points']) > 0:
            ax1.plot(time[waves['qrs']['s_points']], 
                    signal_ecg[waves['qrs']['s_points']], 
                    'yo', markersize=4, label='S points')
        
        # Marcar intervalos
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        # PR intervals
        if 'p_waves' in waves and 'qrs' in waves:
            for i, p_onset in enumerate(waves['p_waves']['onsets'][:5]):  # Primeiros 5
                qrs_after = waves['qrs']['onsets'][waves['qrs']['onsets'] > p_onset]
                if len(qrs_after) > 0:
                    ax1.axvspan(time[p_onset], time[qrs_after[0]], 
                              alpha=0.2, color=colors[0], label='PR' if i == 0 else '')
        
        # QRS durations
        if 'qrs' in waves:
            for i, (onset, offset) in enumerate(zip(waves['qrs']['onsets'][:5], 
                                                   waves['qrs']['offsets'][:5])):
                ax1.axvspan(time[onset], time[offset], 
                          alpha=0.3, color=colors[1], label='QRS' if i == 0 else '')
        
        ax1.set_ylabel('Amplitude (mV)')
        ax1.set_title(f'Delineação ECG - Derivação {result["lead"]}')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot dos intervalos
        intervals = result['intervals']
        interval_names = []
        interval_means = []
        interval_stds = []
        
        for name, data in intervals.items():
            if 'mean' in data and data['mean'] > 0:
                interval_names.append(name)
                interval_means.append(data['mean'])
                interval_stds.append(data['std'])
        
        x = np.arange(len(interval_names))
        ax2.bar(x, interval_means, yerr=interval_stds, capsize=5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(interval_names)
        ax2.set_ylabel('Duração (ms)')
        ax2.set_title('Intervalos Medidos')
        ax2.grid(True, alpha=0.3)
        
        # Adicionar faixas normais
        for i, name in enumerate(interval_names):
            if name in self.normal_ranges:
                normal_min, normal_max = self.normal_ranges[name]
                ax2.axhspan(normal_min, normal_max, alpha=0.2, color='green', 
                          xmin=i/len(interval_names), xmax=(i+1)/len(interval_names))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualização salva em: {save_path}")
        
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Criar delineador
    delineator = AdvancedWaveDelineator(sampling_rate=500)
    
    # Simular sinal ECG
    # ecg_signal = load_ecg_signal()  # Carregar sinal real
    
    # Delinear
    # result = delineator.delineate_complete(ecg_signal, lead_name='II')
    
    # Visualizar
    # delineator.visualize_delineation(result)
    
    print("\n✅ Sistema de delineação de ondas implementado!")
    print("Funcionalidades:")
    print("- Detecção multi-método de picos R")
    print("- Detecção de ondas P e T")
    print("- Delineação completa do QRS")
    print("- Cálculo de todos os intervalos")
    print("- Avaliação de qualidade")
