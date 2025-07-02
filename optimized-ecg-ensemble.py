#!/usr/bin/env python3
"""
Sistema Ensemble ECG Otimizado com Análise Paramétrica Completa
Combina Deep Learning com extração de parâmetros clínicos tradicionais
para diagnóstico mais preciso e interpretável
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
from collections import defaultdict, OrderedDict
from datetime import datetime
import warnings
from tqdm import tqdm
from dataclasses import dataclass
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from scipy import signal as scipy_signal
from scipy.signal import find_peaks, butter, filtfilt, hilbert
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew
import pywt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, multilabel_confusion_matrix
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PARÂMETROS ECG CLÍNICOS ====================

@dataclass
class ECGParameters:
    """Parâmetros clínicos extraídos do ECG"""
    # Frequência Cardíaca
    heart_rate: float
    heart_rate_variability: float
    rhythm_regularity: float
    
    # Amplitudes (em mV)
    p_amplitude: Dict[str, float]  # Por derivação
    qrs_amplitude: Dict[str, float]
    t_amplitude: Dict[str, float]
    st_level: Dict[str, float]
    
    # Durações (em ms)
    p_duration: float
    pr_interval: float
    qrs_duration: float
    qt_interval: float
    qtc_interval: float  # QT corrigido
    
    # Morfologia
    p_morphology: Dict[str, str]  # Normal, bifásica, invertida, etc.
    qrs_morphology: Dict[str, str]  # rS, qR, QS, etc.
    t_morphology: Dict[str, str]  # Normal, invertida, achatada, etc.
    
    # Eixos
    p_axis: float
    qrs_axis: float
    t_axis: float
    
    # Características especiais
    q_waves: Dict[str, bool]  # Ondas Q patológicas
    delta_waves: bool  # WPW
    epsilon_waves: bool  # ARVD
    j_point_elevation: Dict[str, float]
    
    # Variabilidade e complexidade
    qrs_variability: float
    rr_entropy: float
    
    # Índices derivados
    sokolow_lyon_index: float
    cornell_index: float
    romhilt_estes_score: int

# ==================== DETECTOR DE ONDAS AVANÇADO ====================

class AdvancedWaveDetector:
    """Detector avançado de ondas P, QRS e T com análise morfológica"""
    
    def __init__(self, sampling_rate=100):
        self.fs = sampling_rate
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    def detect_r_peaks(self, ecg_signal):
        """Detecção robusta de picos R usando múltiplos métodos"""
        # Método 1: Pan-Tompkins modificado
        r_peaks_pt = self._pan_tompkins_detector(ecg_signal)
        
        # Método 2: Baseado em wavelets
        r_peaks_wt = self._wavelet_detector(ecg_signal)
        
        # Método 3: Baseado em energia
        r_peaks_energy = self._energy_detector(ecg_signal)
        
        # Consenso entre métodos
        r_peaks = self._consensus_peaks(r_peaks_pt, r_peaks_wt, r_peaks_energy)
        
        # Refinar com busca local
        r_peaks = self._refine_r_peaks(ecg_signal, r_peaks)
        
        return r_peaks
    
    def _pan_tompkins_detector(self, signal):
        """Implementação otimizada do Pan-Tompkins"""
        # Filtro passa-banda
        nyquist = self.fs / 2
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = butter(2, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        # Derivada
        diff = np.diff(filtered)
        
        # Elevar ao quadrado
        squared = diff ** 2
        
        # Integração com janela móvel
        window = int(0.15 * self.fs)  # 150ms
        integrated = np.convolve(squared, np.ones(window)/window, mode='same')
        
        # Encontrar picos
        peaks, properties = find_peaks(
            integrated,
            distance=int(0.2 * self.fs),  # Mínimo 200ms entre picos
            height=np.percentile(integrated, 75)
        )
        
        return peaks
    
    def _wavelet_detector(self, signal):
        """Detecção usando transformada wavelet"""
        # Decomposição wavelet
        coeffs = pywt.swt(signal, 'db4', level=4, trim_approx=False)
        
        # Reconstruir apenas com escalas relevantes para QRS
        d3 = coeffs[2][1]
        d4 = coeffs[3][1]
        
        # Combinar escalas
        combined = np.abs(d3) + np.abs(d4)
        
        # Detectar picos
        peaks, _ = find_peaks(
            combined,
            distance=int(0.2 * self.fs),
            height=np.percentile(combined, 80)
        )
        
        return peaks
    
    def _energy_detector(self, signal):
        """Detecção baseada em energia do sinal"""
        # Transformada de Hilbert para envelope
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        
        # Suavizar envelope
        window = int(0.1 * self.fs)
        envelope_smooth = np.convolve(envelope, np.ones(window)/window, mode='same')
        
        # Detectar picos
        peaks, _ = find_peaks(
            envelope_smooth,
            distance=int(0.2 * self.fs),
            height=np.mean(envelope_smooth) + np.std(envelope_smooth)
        )
        
        return peaks
    
    def _consensus_peaks(self, peaks1, peaks2, peaks3):
        """Encontra consenso entre diferentes detectores"""
        all_peaks = []
        tolerance = int(0.05 * self.fs)  # 50ms de tolerância
        
        # Para cada pico no detector 1
        for p1 in peaks1:
            count = 1
            # Verificar se existe em outros detectores
            if any(abs(p2 - p1) < tolerance for p2 in peaks2):
                count += 1
            if any(abs(p3 - p1) < tolerance for p3 in peaks3):
                count += 1
            
            # Se pelo menos 2 detectores concordam
            if count >= 2:
                all_peaks.append(p1)
        
        return np.array(sorted(set(all_peaks)))
    
    def _refine_r_peaks(self, signal, r_peaks):
        """Refina posição dos picos R"""
        refined_peaks = []
        window = int(0.05 * self.fs)  # 50ms
        
        for peak in r_peaks:
            start = max(0, peak - window)
            end = min(len(signal), peak + window)
            
            # Encontrar máximo local
            local_max = start + np.argmax(np.abs(signal[start:end]))
            refined_peaks.append(local_max)
        
        return np.array(refined_peaks)
    
    def detect_p_waves(self, signal, r_peaks):
        """Detecta ondas P antes dos complexos QRS"""
        p_waves = []
        p_onsets = []
        p_offsets = []
        
        for r_peak in r_peaks:
            # Janela de busca: 50-300ms antes do R
            start = max(0, r_peak - int(0.3 * self.fs))
            end = max(0, r_peak - int(0.05 * self.fs))
            
            if start >= end:
                continue
            
            segment = signal[start:end]
            
            # Suavizar segmento
            if len(segment) > 5:
                segment_smooth = scipy_signal.savgol_filter(segment, 5, 2)
                
                # Encontrar pico P
                peaks, properties = find_peaks(segment_smooth, prominence=0.02)
                
                if len(peaks) > 0:
                    # Pegar o pico mais próximo do R
                    p_peak = peaks[-1] + start
                    
                    # Encontrar onset e offset
                    onset, offset = self._find_wave_boundaries(signal, p_peak, 'P')
                    
                    p_waves.append(p_peak)
                    p_onsets.append(onset)
                    p_offsets.append(offset)
        
        return np.array(p_waves), np.array(p_onsets), np.array(p_offsets)
    
    def detect_t_waves(self, signal, r_peaks):
        """Detecta ondas T após complexos QRS"""
        t_waves = []
        t_onsets = []
        t_offsets = []
        
        for i, r_peak in enumerate(r_peaks[:-1]):
            # Janela de busca: 100-400ms após o R
            start = r_peak + int(0.1 * self.fs)
            end = min(r_peak + int(0.4 * self.fs), r_peaks[i+1] - int(0.1 * self.fs))
            
            if start >= end or end > len(signal):
                continue
            
            segment = signal[start:end]
            
            if len(segment) > 5:
                # Suavizar
                segment_smooth = scipy_signal.savgol_filter(segment, 5, 2)
                
                # Encontrar pico T (positivo ou negativo)
                peaks_pos, _ = find_peaks(segment_smooth, prominence=0.05)
                peaks_neg, _ = find_peaks(-segment_smooth, prominence=0.05)
                
                # Escolher o maior
                if len(peaks_pos) > 0 and len(peaks_neg) > 0:
                    if segment_smooth[peaks_pos[0]] > abs(segment_smooth[peaks_neg[0]]):
                        t_peak = peaks_pos[0] + start
                    else:
                        t_peak = peaks_neg[0] + start
                elif len(peaks_pos) > 0:
                    t_peak = peaks_pos[0] + start
                elif len(peaks_neg) > 0:
                    t_peak = peaks_neg[0] + start
                else:
                    continue
                
                # Boundaries
                onset, offset = self._find_wave_boundaries(signal, t_peak, 'T')
                
                t_waves.append(t_peak)
                t_onsets.append(onset)
                t_offsets.append(offset)
        
        return np.array(t_waves), np.array(t_onsets), np.array(t_offsets)
    
    def _find_wave_boundaries(self, signal, peak, wave_type):
        """Encontra início e fim de uma onda"""
        window = int(0.1 * self.fs) if wave_type == 'P' else int(0.15 * self.fs)
        
        # Onset: buscar para trás até encontrar linha de base
        onset = peak
        baseline = np.median(signal)
        threshold = 0.1 * abs(signal[peak] - baseline)
        
        for i in range(peak, max(peak - window, 0), -1):
            if abs(signal[i] - baseline) < threshold:
                onset = i
                break
        
        # Offset: buscar para frente
        offset = peak
        for i in range(peak, min(peak + window, len(signal))):
            if abs(signal[i] - baseline) < threshold:
                offset = i
                break
        
        return onset, offset
    
    def analyze_morphology(self, signal, wave_peaks, wave_type):
        """Analisa morfologia das ondas"""
        morphologies = []
        
        for peak in wave_peaks:
            window = int(0.1 * self.fs)
            start = max(0, peak - window)
            end = min(len(signal), peak + window)
            
            segment = signal[start:end]
            
            # Características morfológicas
            if wave_type == 'P':
                morph = self._classify_p_morphology(segment)
            elif wave_type == 'QRS':
                morph = self._classify_qrs_morphology(segment)
            elif wave_type == 'T':
                morph = self._classify_t_morphology(segment)
            else:
                morph = 'unknown'
            
            morphologies.append(morph)
        
        return morphologies
    
    def _classify_p_morphology(self, segment):
        """Classifica morfologia da onda P"""
        # Verificar se é bifásica
        peaks_pos, _ = find_peaks(segment)
        peaks_neg, _ = find_peaks(-segment)
        
        if len(peaks_pos) > 0 and len(peaks_neg) > 0:
            return 'bifásica'
        elif len(peaks_neg) > 0 and len(peaks_pos) == 0:
            return 'invertida'
        elif np.max(segment) > 2.5:  # >2.5mm
            return 'aumentada'
        else:
            return 'normal'
    
    def _classify_qrs_morphology(self, segment):
        """Classifica morfologia do QRS"""
        # Simplified classification
        # Procurar ondas Q, R e S
        baseline = np.mean(segment[:5])
        
        # Primeira deflexão negativa = Q
        q_present = any(segment[:len(segment)//3] < baseline - 0.1)
        
        # Deflexão positiva = R
        r_amplitude = np.max(segment) - baseline
        
        # Deflexão negativa após R = S
        r_pos = np.argmax(segment)
        s_present = any(segment[r_pos:] < baseline - 0.1)
        
        if q_present and r_amplitude > 0.5 and s_present:
            return 'qRs'
        elif q_present and r_amplitude < 0.5:
            return 'QS'
        elif not q_present and s_present:
            return 'Rs'
        elif r_amplitude > 1.0:
            return 'R'
        else:
            return 'rS'
    
    def _classify_t_morphology(self, segment):
        """Classifica morfologia da onda T"""
        peak_val = np.max(np.abs(segment))
        peak_pos = np.argmax(np.abs(segment))
        
        if segment[peak_pos] < -0.1:
            return 'invertida'
        elif peak_val < 0.05:
            return 'achatada'
        elif peak_val > 1.0:
            return 'aumentada'
        else:
            # Verificar simetria
            first_half = segment[:peak_pos]
            second_half = segment[peak_pos:]
            
            if len(first_half) > 0 and len(second_half) > 0:
                asymmetry = abs(len(first_half) - len(second_half)) / len(segment)
                if asymmetry > 0.3:
                    return 'assimétrica'
            
            return 'normal'

# ==================== EXTRATOR DE PARÂMETROS CLÍNICOS ====================

class ClinicalParameterExtractor:
    """Extrai todos os parâmetros clínicos relevantes do ECG"""
    
    def __init__(self, sampling_rate=100):
        self.fs = sampling_rate
        self.wave_detector = AdvancedWaveDetector(sampling_rate)
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    def extract_all_parameters(self, ecg_12lead):
        """
        Extrai todos os parâmetros clínicos de um ECG 12 derivações
        
        Args:
            ecg_12lead: Array (12, n_samples) com ECG 12 derivações
            
        Returns:
            ECGParameters: Objeto com todos os parâmetros extraídos
        """
        # Usar derivação II para detecção principal (melhor para ritmo)
        lead_ii = ecg_12lead[1]
        
        # Detectar ondas
        r_peaks = self.wave_detector.detect_r_peaks(lead_ii)
        p_waves, p_onsets, p_offsets = self.wave_detector.detect_p_waves(lead_ii, r_peaks)
        t_waves, t_onsets, t_offsets = self.wave_detector.detect_t_waves(lead_ii, r_peaks)
        
        # Extrair parâmetros
        params = ECGParameters(
            # Frequência cardíaca
            heart_rate=self._calculate_heart_rate(r_peaks),
            heart_rate_variability=self._calculate_hrv(r_peaks),
            rhythm_regularity=self._calculate_rhythm_regularity(r_peaks),
            
            # Amplitudes
            p_amplitude=self._measure_wave_amplitudes(ecg_12lead, p_waves, 'P'),
            qrs_amplitude=self._measure_wave_amplitudes(ecg_12lead, r_peaks, 'QRS'),
            t_amplitude=self._measure_wave_amplitudes(ecg_12lead, t_waves, 'T'),
            st_level=self._measure_st_levels(ecg_12lead, r_peaks),
            
            # Durações
            p_duration=self._calculate_p_duration(p_onsets, p_offsets),
            pr_interval=self._calculate_pr_interval(p_onsets, r_peaks),
            qrs_duration=self._calculate_qrs_duration(ecg_12lead, r_peaks),
            qt_interval=self._calculate_qt_interval(r_peaks, t_offsets),
            qtc_interval=self._calculate_qtc(r_peaks, t_offsets),
            
            # Morfologia
            p_morphology=self._analyze_p_morphology(ecg_12lead, p_waves),
            qrs_morphology=self._analyze_qrs_morphology(ecg_12lead, r_peaks),
            t_morphology=self._analyze_t_morphology(ecg_12lead, t_waves),
            
            # Eixos
            p_axis=self._calculate_axis(ecg_12lead, p_waves, 'P'),
            qrs_axis=self._calculate_axis(ecg_12lead, r_peaks, 'QRS'),
            t_axis=self._calculate_axis(ecg_12lead, t_waves, 'T'),
            
            # Características especiais
            q_waves=self._detect_pathological_q_waves(ecg_12lead, r_peaks),
            delta_waves=self._detect_delta_waves(ecg_12lead, p_waves, r_peaks),
            epsilon_waves=self._detect_epsilon_waves(ecg_12lead, r_peaks),
            j_point_elevation=self._measure_j_point(ecg_12lead, r_peaks),
            
            # Variabilidade
            qrs_variability=self._calculate_qrs_variability(ecg_12lead, r_peaks),
            rr_entropy=self._calculate_rr_entropy(r_peaks),
            
            # Índices
            sokolow_lyon_index=self._calculate_sokolow_lyon(ecg_12lead, r_peaks),
            cornell_index=self._calculate_cornell_index(ecg_12lead, r_peaks),
            romhilt_estes_score=self._calculate_romhilt_estes(ecg_12lead, r_peaks)
        )
        
        return params
    
    def _calculate_heart_rate(self, r_peaks):
        """Calcula frequência cardíaca média"""
        if len(r_peaks) < 2:
            return 0
        
        rr_intervals = np.diff(r_peaks) / self.fs  # Em segundos
        heart_rate = 60 / np.mean(rr_intervals)
        
        return heart_rate
    
    def _calculate_hrv(self, r_peaks):
        """Calcula variabilidade da frequência cardíaca (SDNN)"""
        if len(r_peaks) < 3:
            return 0
        
        rr_intervals = np.diff(r_peaks) / self.fs * 1000  # Em ms
        hrv = np.std(rr_intervals)
        
        return hrv
    
    def _calculate_rhythm_regularity(self, r_peaks):
        """Calcula regularidade do ritmo (0-1, 1=regular)"""
        if len(r_peaks) < 3:
            return 1.0
        
        rr_intervals = np.diff(r_peaks)
        cv = np.std(rr_intervals) / np.mean(rr_intervals)  # Coeficiente de variação
        
        # Converter para escala 0-1
        regularity = 1 / (1 + cv)
        
        return regularity
    
    def _measure_wave_amplitudes(self, ecg_12lead, wave_peaks, wave_type):
        """Mede amplitudes das ondas em todas as derivações"""
        amplitudes = {}
        
        for i, lead_name in enumerate(self.lead_names):
            lead_signal = ecg_12lead[i]
            
            if len(wave_peaks) > 0:
                # Calcular baseline
                baseline = np.median(lead_signal)
                
                # Medir amplitudes
                wave_amplitudes = []
                for peak in wave_peaks:
                    if 0 <= peak < len(lead_signal):
                        amplitude = abs(lead_signal[peak] - baseline)
                        wave_amplitudes.append(amplitude)
                
                amplitudes[lead_name] = np.mean(wave_amplitudes) if wave_amplitudes else 0
            else:
                amplitudes[lead_name] = 0
        
        return amplitudes
    
    def _measure_st_levels(self, ecg_12lead, r_peaks):
        """Mede níveis do segmento ST"""
        st_levels = {}
        
        for i, lead_name in enumerate(self.lead_names):
            lead_signal = ecg_12lead[i]
            st_measurements = []
            
            for r_peak in r_peaks:
                # Ponto J: 40ms após o pico R
                j_point = r_peak + int(0.04 * self.fs)
                # ST medido 80ms após o ponto J
                st_point = j_point + int(0.08 * self.fs)
                
                if st_point < len(lead_signal):
                    # Baseline: segmento PR
                    pr_start = max(0, r_peak - int(0.08 * self.fs))
                    pr_end = max(0, r_peak - int(0.04 * self.fs))
                    baseline = np.mean(lead_signal[pr_start:pr_end])
                    
                    st_level = lead_signal[st_point] - baseline
                    st_measurements.append(st_level)
            
            st_levels[lead_name] = np.mean(st_measurements) if st_measurements else 0
        
        return st_levels
    
    def _calculate_p_duration(self, p_onsets, p_offsets):
        """Calcula duração média da onda P"""
        if len(p_onsets) == 0 or len(p_offsets) == 0:
            return 0
        
        durations = []
        for onset, offset in zip(p_onsets, p_offsets):
            duration = (offset - onset) / self.fs * 1000  # Em ms
            if 0 < duration < 200:  # Sanity check
                durations.append(duration)
        
        return np.mean(durations) if durations else 0
    
    def _calculate_pr_interval(self, p_onsets, r_peaks):
        """Calcula intervalo PR médio"""
        if len(p_onsets) == 0 or len(r_peaks) == 0:
            return 0
        
        intervals = []
        for p_onset in p_onsets:
            # Encontrar próximo R
            next_r = r_peaks[r_peaks > p_onset]
            if len(next_r) > 0:
                pr = (next_r[0] - p_onset) / self.fs * 1000  # Em ms
                if 50 < pr < 400:  # Sanity check
                    intervals.append(pr)
        
        return np.mean(intervals) if intervals else 0
    
    def _calculate_qrs_duration(self, ecg_12lead, r_peaks):
        """Calcula duração do complexo QRS"""
        if len(r_peaks) == 0:
            return 0
        
        durations = []
        
        # Usar múltiplas derivações para melhor precisão
        for r_peak in r_peaks[:min(10, len(r_peaks))]:  # Primeiros 10 complexos
            # Combinar várias derivações
            combined = np.abs(ecg_12lead[1]) + np.abs(ecg_12lead[5])  # II + aVF
            
            # Encontrar início e fim do QRS
            window = int(0.1 * self.fs)
            start = max(0, r_peak - window)
            end = min(len(combined), r_peak + window)
            
            segment = combined[start:end]
            threshold = 0.1 * np.max(segment)
            
            # Onset
            qrs_onset = start
            for i in range(r_peak - start, 0, -1):
                if segment[i] < threshold:
                    qrs_onset = start + i
                    break
            
            # Offset
            qrs_offset = end
            for i in range(r_peak - start, len(segment)):
                if segment[i] < threshold:
                    qrs_offset = start + i
                    break
            
            duration = (qrs_offset - qrs_onset) / self.fs * 1000  # Em ms
            if 40 < duration < 200:  # Sanity check
                durations.append(duration)
        
        return np.mean(durations) if durations else 0
    
    def _calculate_qt_interval(self, r_peaks, t_offsets):
        """Calcula intervalo QT"""
        if len(r_peaks) == 0 or len(t_offsets) == 0:
            return 0
        
        intervals = []
        for r_peak in r_peaks:
            # Encontrar próximo T offset
            next_t = t_offsets[t_offsets > r_peak]
            if len(next_t) > 0:
                qt = (next_t[0] - r_peak) / self.fs * 1000  # Em ms
                if 200 < qt < 600:  # Sanity check
                    intervals.append(qt)
        
        return np.mean(intervals) if intervals else 0
    
    def _calculate_qtc(self, r_peaks, t_offsets):
        """Calcula QT corrigido (fórmula de Bazett)"""
        qt = self._calculate_qt_interval(r_peaks, t_offsets)
        hr = self._calculate_heart_rate(r_peaks)
        
        if qt > 0 and hr > 0:
            rr = 60 / hr  # RR em segundos
            qtc = qt / np.sqrt(rr)
            return qtc
        
        return 0
    
    def _analyze_p_morphology(self, ecg_12lead, p_waves):
        """Analisa morfologia da onda P em cada derivação"""
        morphology = {}
        
        for i, lead_name in enumerate(self.lead_names):
            if len(p_waves) > 0:
                lead_signal = ecg_12lead[i]
                morphs = self.wave_detector.analyze_morphology(
                    lead_signal, p_waves[:5], 'P'
                )
                # Morfologia predominante
                morphology[lead_name] = max(set(morphs), key=morphs.count) if morphs else 'normal'
            else:
                morphology[lead_name] = 'ausente'
        
        return morphology
    
    def _analyze_qrs_morphology(self, ecg_12lead, r_peaks):
        """Analisa morfologia do QRS"""
        morphology = {}
        
        for i, lead_name in enumerate(self.lead_names):
            if len(r_peaks) > 0:
                lead_signal = ecg_12lead[i]
                morphs = self.wave_detector.analyze_morphology(
                    lead_signal, r_peaks[:5], 'QRS'
                )
                morphology[lead_name] = max(set(morphs), key=morphs.count) if morphs else 'normal'
            else:
                morphology[lead_name] = 'ausente'
        
        return morphology
    
    def _analyze_t_morphology(self, ecg_12lead, t_waves):
        """Analisa morfologia da onda T"""
        morphology = {}
        
        for i, lead_name in enumerate(self.lead_names):
            if len(t_waves) > 0:
                lead_signal = ecg_12lead[i]
                morphs = self.wave_detector.analyze_morphology(
                    lead_signal, t_waves[:5], 'T'
                )
                morphology[lead_name] = max(set(morphs), key=morphs.count) if morphs else 'normal'
            else:
                morphology[lead_name] = 'ausente'
        
        return morphology
    
    def _calculate_axis(self, ecg_12lead, wave_peaks, wave_type):
        """Calcula eixo elétrico"""
        if len(wave_peaks) == 0:
            return 0
        
        # Usar derivações I e aVF
        lead_i = ecg_12lead[0]
        lead_avf = ecg_12lead[5]
        
        # Calcular área sob a curva para as ondas
        area_i = []
        area_avf = []
        
        for peak in wave_peaks[:5]:
            window = int(0.05 * self.fs)
            start = max(0, peak - window)
            end = min(len(lead_i), peak + window)
            
            area_i.append(np.trapz(lead_i[start:end]))
            area_avf.append(np.trapz(lead_avf[start:end]))
        
        mean_i = np.mean(area_i) if area_i else 0
        mean_avf = np.mean(area_avf) if area_avf else 0
        
        # Calcular ângulo
        axis_degrees = np.degrees(np.arctan2(mean_avf, mean_i))
        
        return axis_degrees
    
    def _detect_pathological_q_waves(self, ecg_12lead, r_peaks):
        """Detecta ondas Q patológicas"""
        q_waves = {}
        
        for i, lead_name in enumerate(self.lead_names):
            lead_signal = ecg_12lead[i]
            has_pathological_q = False
            
            for r_peak in r_peaks[:5]:
                # Buscar Q antes do R
                window = int(0.04 * self.fs)
                start = max(0, r_peak - window)
                
                segment = lead_signal[start:r_peak]
                if len(segment) > 0:
                    q_depth = np.min(segment)
                    r_height = lead_signal[r_peak]
                    
                    # Q patológica: >25% da altura R ou >40ms de duração
                    if abs(q_depth) > 0.25 * abs(r_height):
                        has_pathological_q = True
                        break
            
            q_waves[lead_name] = has_pathological_q
        
        return q_waves
    
    def _detect_delta_waves(self, ecg_12lead, p_waves, r_peaks):
        """Detecta ondas delta (WPW)"""
        if len(p_waves) == 0 or len(r_peaks) == 0:
            return False
        
        # Verificar PR curto e início lento do QRS
        pr_intervals = []
        for p in p_waves:
            next_r = r_peaks[r_peaks > p]
            if len(next_r) > 0:
                pr = (next_r[0] - p) / self.fs * 1000
                pr_intervals.append(pr)
        
        # Delta wave presente se PR < 120ms
        return np.mean(pr_intervals) < 120 if pr_intervals else False
    
    def _detect_epsilon_waves(self, ecg_12lead, r_peaks):
        """Detecta ondas epsilon (ARVD)"""
        # Buscar em V1-V3
        for i in range(6, 9):  # V1-V3
            lead_signal = ecg_12lead[i]
            
            for r_peak in r_peaks[:5]:
                # Buscar após QRS
                start = r_peak + int(0.04 * self.fs)
                end = r_peak + int(0.15 * self.fs)
                
                if end < len(lead_signal):
                    segment = lead_signal[start:end]
                    
                    # Procurar pequenas deflexões
                    if len(segment) > 10:
                        # Filtro passa-alta para realçar altas frequências
                        b, a = butter(2, 40/(self.fs/2), 'high')
                        filtered = filtfilt(b, a, segment)
                        
                        # Epsilon waves têm pequenas deflexões de alta frequência
                        if np.max(np.abs(filtered)) > 0.05:
                            return True
        
        return False
    
    def _measure_j_point(self, ecg_12lead, r_peaks):
        """Mede elevação do ponto J"""
        j_points = {}
        
        for i, lead_name in enumerate(self.lead_names):
            lead_signal = ecg_12lead[i]
            j_elevations = []
            
            for r_peak in r_peaks[:5]:
                # J point: fim do QRS
                j_point = r_peak + int(0.04 * self.fs)
                
                if j_point < len(lead_signal):
                    # Baseline
                    baseline_start = max(0, r_peak - int(0.08 * self.fs))
                    baseline_end = max(0, r_peak - int(0.04 * self.fs))
                    baseline = np.mean(lead_signal[baseline_start:baseline_end])
                    
                    j_elevation = lead_signal[j_point] - baseline
                    j_elevations.append(j_elevation)
            
            j_points[lead_name] = np.mean(j_elevations) if j_elevations else 0
        
        return j_points
    
    def _calculate_qrs_variability(self, ecg_12lead, r_peaks):
        """Calcula variabilidade da morfologia do QRS"""
        if len(r_peaks) < 3:
            return 0
        
        # Extrair templates QRS
        templates = []
        lead_ii = ecg_12lead[1]
        
        for r_peak in r_peaks[:10]:
            window = int(0.1 * self.fs)
            start = max(0, r_peak - window)
            end = min(len(lead_ii), r_peak + window)
            
            template = lead_ii[start:end]
            if len(template) == 2 * window:
                templates.append(template)
        
        if len(templates) < 2:
            return 0
        
        # Calcular variabilidade entre templates
        templates = np.array(templates)
        mean_template = np.mean(templates, axis=0)
        
        variabilities = []
        for template in templates:
            # Correlação com template médio
            corr = np.corrcoef(template, mean_template)[0, 1]
            variabilities.append(1 - corr)
        
        return np.mean(variabilities)
    
    def _calculate_rr_entropy(self, r_peaks):
        """Calcula entropia dos intervalos RR"""
        if len(r_peaks) < 4:
            return 0
        
        rr_intervals = np.diff(r_peaks)
        
        # Discretizar em bins
        n_bins = min(10, len(rr_intervals) // 2)
        hist, _ = np.histogram(rr_intervals, bins=n_bins)
        
        # Calcular entropia
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]  # Remover zeros
        
        entropy = -np.sum(probs * np.log2(probs))
        
        return entropy
    
    def _calculate_sokolow_lyon(self, ecg_12lead, r_peaks):
        """Calcula índice de Sokolow-Lyon para HVE"""
        if len(r_peaks) == 0:
            return 0
        
        # S em V1 + R em V5 ou V6
        s_v1 = np.mean([abs(np.min(ecg_12lead[6][r-10:r+10])) for r in r_peaks[:5]])
        r_v5 = np.mean([np.max(ecg_12lead[10][r-10:r+10]) for r in r_peaks[:5]])
        r_v6 = np.mean([np.max(ecg_12lead[11][r-10:r+10]) for r in r_peaks[:5]])
        
        return s_v1 + max(r_v5, r_v6)
    
    def _calculate_cornell_index(self, ecg_12lead, r_peaks):
        """Calcula índice de Cornell para HVE"""
        if len(r_peaks) == 0:
            return 0
        
        # R em aVL + S em V3
        r_avl = np.mean([np.max(ecg_12lead[4][r-10:r+10]) for r in r_peaks[:5]])
        s_v3 = np.mean([abs(np.min(ecg_12lead[8][r-10:r+10])) for r in r_peaks[:5]])
        
        return r_avl + s_v3
    
    def _calculate_romhilt_estes(self, ecg_12lead, r_peaks):
        """Calcula score de Romhilt-Estes para HVE"""
        score = 0
        
        # Critérios simplificados
        # 1. Amplitude (3 pontos)
        sokolow = self._calculate_sokolow_lyon(ecg_12lead, r_peaks)
        if sokolow > 35:
            score += 3
        
        # 2. Padrão strain (3 pontos)
        # ST depression + T inversion em V5-V6
        st_v5 = self._measure_st_levels(ecg_12lead, r_peaks).get('V5', 0)
        if st_v5 < -0.1:
            score += 3
        
        # 3. Aumento atrial esquerdo (3 pontos)
        p_duration = self._calculate_p_duration([], [])  # Simplificado
        if p_duration > 110:
            score += 3
        
        # 4. Desvio do eixo (2 pontos)
        qrs_axis = self._calculate_axis(ecg_12lead, r_peaks, 'QRS')
        if qrs_axis < -30:
            score += 2
        
        # 5. Duração QRS (1 ponto)
        qrs_dur = self._calculate_qrs_duration(ecg_12lead, r_peaks)
        if qrs_dur > 90:
            score += 1
        
        return score

# ==================== DATASET OTIMIZADO COM PARÂMETROS ====================

class OptimizedECGDataset(Dataset):
    """Dataset que combina sinais ECG com parâmetros clínicos extraídos"""
    
    def __init__(self, X, Y, config, parameter_cache=None, is_training=True):
        self.X = X
        self.Y = Y
        self.config = config
        self.is_training = is_training
        
        # Cache de parâmetros para evitar recálculo
        self.parameter_cache = parameter_cache or {}
        
        # Extrator de parâmetros
        self.param_extractor = ClinicalParameterExtractor(config.sampling_rate)
        
        # Normalização
        self.signal_mean = np.mean(X, axis=(0, 2), keepdims=True)
        self.signal_std = np.std(X, axis=(0, 2), keepdims=True) + 1e-8
        
        logger.info(f"Dataset criado: {len(X)} amostras")
        if not parameter_cache:
            logger.info("Extraindo parâmetros clínicos... (isso pode demorar)")
            self._precompute_parameters()
    
    def _precompute_parameters(self):
        """Pré-calcula parâmetros para todas as amostras"""
        for idx in tqdm(range(len(self.X)), desc="Extraindo parâmetros"):
            if idx not in self.parameter_cache:
                ecg = self.X[idx]
                params = self.param_extractor.extract_all_parameters(ecg)
                self.parameter_cache[idx] = self._params_to_vector(params)
    
    def _params_to_vector(self, params):
        """Converte ECGParameters para vetor de features"""
        features = []
        
        # Frequência cardíaca e variabilidade
        features.extend([
            params.heart_rate,
            params.heart_rate_variability,
            params.rhythm_regularity
        ])
        
        # Amplitudes médias
        features.extend([
            np.mean(list(params.p_amplitude.values())),
            np.mean(list(params.qrs_amplitude.values())),
            np.mean(list(params.t_amplitude.values())),
            np.mean(list(params.st_level.values()))
        ])
        
        # Durações
        features.extend([
            params.p_duration,
            params.pr_interval,
            params.qrs_duration,
            params.qt_interval,
            params.qtc_interval
        ])
        
        # Eixos
        features.extend([
            params.p_axis,
            params.qrs_axis,
            params.t_axis
        ])
        
        # Características especiais (binárias)
        features.extend([
            float(any(params.q_waves.values())),
            float(params.delta_waves),
            float(params.epsilon_waves),
            np.mean(list(params.j_point_elevation.values()))
        ])
        
        # Variabilidade e complexidade
        features.extend([
            params.qrs_variability,
            params.rr_entropy
        ])
        
        # Índices
        features.extend([
            params.sokolow_lyon_index,
            params.cornell_index,
            float(params.romhilt_estes_score)
        ])
        
        return np.array(features, dtype=np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # ECG signal
        ecg = self.X[idx].copy()
        ecg = (ecg - self.signal_mean[idx]) / self.signal_std[idx]
        
        # Parâmetros clínicos
        if idx in self.parameter_cache:
            params = self.parameter_cache[idx]
        else:
            ecg_params = self.param_extractor.extract_all_parameters(self.X[idx])
            params = self._params_to_vector(ecg_params)
            self.parameter_cache[idx] = params
        
        # Augmentação durante treino
        if self.is_training and np.random.rand() < 0.7:
            ecg = self._apply_augmentation(ecg)
        
        # Converter para tensores
        ecg_tensor = torch.FloatTensor(ecg)
        params_tensor = torch.FloatTensor(params)
        
        if self.config.use_multilabel:
            label = torch.FloatTensor(self.Y[idx])
        else:
            label = torch.LongTensor([self.Y[idx]])
        
        return ecg_tensor, params_tensor, label
    
    def _apply_augmentation(self, ecg):
        """Aplica augmentação preservando características clínicas"""
        # Amplitude scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            ecg = ecg * scale
        
        # Baseline wander
        if np.random.rand() < 0.3:
            t = np.linspace(0, 10, ecg.shape[1])
            for i in range(ecg.shape[0]):
                baseline = 0.05 * np.sin(2 * np.pi * 0.3 * t + np.random.rand() * 2 * np.pi)
                ecg[i] += baseline
        
        # Ruído gaussiano leve
        if np.random.rand() < 0.3:
            noise = 0.01 * np.random.randn(*ecg.shape)
            ecg = ecg + noise
        
        return ecg

# ==================== MODELOS OTIMIZADOS COM FUSÃO DE PARÂMETROS ====================

class ParameterAwareECGModel(nn.Module):
    """Modelo base que integra parâmetros clínicos com deep learning"""
    
    def __init__(self, num_classes, num_parameters=26):
        super().__init__()
        self.num_classes = num_classes
        self.num_parameters = num_parameters
        
        # Processamento de parâmetros clínicos
        self.param_processor = nn.Sequential(
            nn.Linear(num_parameters, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # Gating mechanism para fusão adaptativa
        self.fusion_gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def process_parameters(self, params):
        """Processa parâmetros clínicos"""
        return self.param_processor(params)
    
    def adaptive_fusion(self, deep_features, param_features):
        """Fusão adaptativa de features de deep learning e parâmetros"""
        # Gate para cada amostra
        gate = self.fusion_gate(param_features)
        
        # Fusão ponderada
        fused = gate * param_features + (1 - gate) * deep_features
        
        return fused

class OptimizedRhythmAnalyzer(ParameterAwareECGModel):
    """Analisador de ritmo otimizado com parâmetros"""
    
    def __init__(self, num_classes):
        super().__init__(num_classes)
        
        # CNN para análise de ritmo
        self.rhythm_cnn = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=50, stride=2, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=10, stride=2, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Atenção temporal para padrões de ritmo
        self.temporal_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        
        # RNN para sequências de RR
        self.rr_lstm = nn.LSTM(256, 128, num_layers=2, 
                              batch_first=True, bidirectional=True)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classificador com fusão
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 512),  # Deep features + param features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, ecg, params):
        # Processar sinais
        x = self.rhythm_cnn(ecg)
        
        # Atenção temporal
        x_att = x.transpose(1, 2)
        x_att, _ = self.temporal_attention(x_att, x_att, x_att)
        x = x + x_att.transpose(1, 2)
        
        # LSTM para padrões sequenciais
        x_seq = x.transpose(1, 2)
        x_lstm, _ = self.rr_lstm(x_seq)
        
        # Pool
        x_pooled = self.global_pool(x).squeeze(-1)
        x_lstm_pooled = x_lstm.mean(dim=1)
        
        # Combinar features CNN e LSTM
        deep_features = x_pooled + x_lstm_pooled[:, :256]
        
        # Processar parâmetros
        param_features = self.process_parameters(params)
        
        # Fusão adaptativa
        fused_features = torch.cat([deep_features, param_features], dim=1)
        
        # Classificação
        output = self.classifier(fused_features)
        
        return output

class OptimizedMorphologyAnalyzer(ParameterAwareECGModel):
    """Analisador morfológico com Inception e parâmetros"""
    
    def __init__(self, num_classes):
        super().__init__(num_classes)
        
        # Inception blocks para diferentes escalas
        self.inception1 = self._make_inception_block(12, 128)
        self.inception2 = self._make_inception_block(128, 256)
        self.inception3 = self._make_inception_block(256, 512)
        
        # Atenção por derivação
        self.lead_attention = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 12, 1),
            nn.Sigmoid()
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256, 768),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(768, num_classes)
        )
    
    def _make_inception_block(self, in_channels, out_channels):
        return nn.ModuleDict({
            'conv1x1': nn.Sequential(
                nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
                nn.BatchNorm1d(out_channels//4),
                nn.ReLU()
            ),
            'conv3x3': nn.Sequential(
                nn.Conv1d(in_channels, out_channels//4, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels//4),
                nn.ReLU()
            ),
            'conv5x5': nn.Sequential(
                nn.Conv1d(in_channels, out_channels//4, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels//4),
                nn.ReLU()
            ),
            'conv7x7': nn.Sequential(
                nn.Conv1d(in_channels, out_channels//4, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_channels//4),
                nn.ReLU()
            )
        })
    
    def forward(self, ecg, params):
        # Inception processing
        x = ecg
        for inception in [self.inception1, self.inception2, self.inception3]:
            outputs = []
            for conv in inception.values():
                outputs.append(conv(x))
            x = torch.cat(outputs, dim=1)
        
        # Atenção por derivação
        att_weights = self.lead_attention(x)
        
        # Aplicar atenção multiplicando pelos pesos originais do ECG
        x_weighted = x * att_weights
        
        # Global pooling
        deep_features = self.global_pool(x_weighted).squeeze(-1)
        
        # Processar parâmetros
        param_features = self.process_parameters(params)
        
        # Fusão
        fused_features = torch.cat([deep_features, param_features], dim=1)
        
        # Classificação
        output = self.classifier(fused_features)
        
        return output

class OptimizedGlobalIntegrator(ParameterAwareECGModel):
    """Integrador global com Transformer e parâmetros clínicos"""
    
    def __init__(self, num_classes):
        super().__init__(num_classes)
        
        # Embedding inicial
        self.input_projection = nn.Conv1d(12, 256, kernel_size=1)
        
        # Positional encoding learnable
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, 256))
        
        # Transformer com cross-attention para parâmetros
        self.transformer_encoder = nn.ModuleList([
            TransformerBlockWithCrossAttention(256, 8, 1024)
            for _ in range(4)
        ])
        
        # Projeção de parâmetros para cross-attention
        self.param_projection = nn.Linear(256, 256)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.layer_norm = nn.LayerNorm(256)
        
        # Classificador final
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, ecg, params):
        # Projeção inicial
        x = self.input_projection(ecg)
        x = x.transpose(1, 2)  # (B, L, C)
        
        # Adicionar positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Processar parâmetros
        param_features = self.process_parameters(params)
        param_projected = self.param_projection(param_features).unsqueeze(1)
        
        # Transformer com cross-attention
        for transformer in self.transformer_encoder:
            x = transformer(x, param_projected)
        
        # Pooling
        x = x.transpose(1, 2)
        deep_features = self.global_pool(x).squeeze(-1)
        deep_features = self.layer_norm(deep_features)
        
        # Fusão final
        fused_features = torch.cat([deep_features, param_features], dim=1)
        
        # Classificação
        output = self.classifier(fused_features)
        
        return output

class TransformerBlockWithCrossAttention(nn.Module):
    """Bloco Transformer com cross-attention para parâmetros"""
    
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, params):
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention com parâmetros
        cross_out, _ = self.cross_attention(x, params, params)
        x = self.norm2(x + self.dropout(cross_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x

# ==================== ENSEMBLE MÉDICO OTIMIZADO ====================

class OptimizedMedicalEnsemble(nn.Module):
    """Ensemble otimizado com análise paramétrica completa"""
    
    def __init__(self, num_classes, config):
        super().__init__()
        self.num_classes = num_classes
        self.config = config
        
        # Modelos especializados otimizados
        self.models = nn.ModuleDict({
            'rhythm': OptimizedRhythmAnalyzer(num_classes),
            'morphology': OptimizedMorphologyAnalyzer(num_classes),
            'global': OptimizedGlobalIntegrator(num_classes)
        })
        
        # Analisador de parâmetros dedicado
        self.parameter_analyzer = nn.Sequential(
            nn.Linear(26, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )
        
        # Meta-learner otimizado
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 4, 512),  # 3 modelos + param analyzer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Interpretador clínico
        self.clinical_interpreter = ClinicalInterpreter(num_classes)
    
    def forward(self, ecg, params, return_interpretation=False):
        # Predições dos modelos especializados
        outputs = {}
        outputs['rhythm'] = self.models['rhythm'](ecg, params)
        outputs['morphology'] = self.models['morphology'](ecg, params)
        outputs['global'] = self.models['global'](ecg, params)
        
        # Análise direta de parâmetros
        outputs['parameters'] = self.parameter_analyzer(params)
        
        # Meta-learning
        all_outputs = torch.cat(list(outputs.values()), dim=1)
        meta_output = self.meta_learner(all_outputs)
        
        # Combinação final
        final_output = 0.4 * meta_output + 0.2 * outputs['rhythm'] + \
                      0.2 * outputs['morphology'] + 0.1 * outputs['global'] + \
                      0.1 * outputs['parameters']
        
        if return_interpretation:
            interpretation = self.clinical_interpreter(
                final_output, params, outputs
            )
            return final_output, interpretation
        
        return final_output

class ClinicalInterpreter(nn.Module):
    """Módulo para interpretação clínica dos resultados"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Redes para gerar explicações
        self.explanation_generator = nn.ModuleDict({
            'rhythm': nn.Linear(num_classes, 128),
            'morphology': nn.Linear(num_classes, 128),
            'conduction': nn.Linear(num_classes, 128),
            'ischemia': nn.Linear(num_classes, 128)
        })
        
        self.explanation_combiner = nn.Sequential(
            nn.Linear(512 + 26, 256),  # Explanations + parameters
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, predictions, params, model_outputs):
        """Gera interpretação clínica estruturada"""
        explanations = {}
        
        # Gerar explicações por categoria
        probs = torch.sigmoid(predictions)
        
        for category, generator in self.explanation_generator.items():
            explanations[category] = generator(probs)
        
        # Combinar com parâmetros
        all_explanations = torch.cat(list(explanations.values()), dim=1)
        combined = torch.cat([all_explanations, params], dim=1)
        
        interpretation_vector = self.explanation_combiner(combined)
        
        # Estruturar interpretação
        interpretation = {
            'confidence_scores': probs,
            'explanation_vector': interpretation_vector,
            'parameter_contributions': self._analyze_parameter_contributions(params),
            'critical_findings': self._identify_critical_findings(probs, params)
        }
        
        return interpretation
    
    def _analyze_parameter_contributions(self, params):
        """Analisa contribuição de cada parâmetro para o diagnóstico"""
        # Implementação simplificada
        contributions = {
            'heart_rate': params[:, 0],
            'pr_interval': params[:, 8],
            'qrs_duration': params[:, 9],
            'qt_interval': params[:, 10],
            'st_levels': params[:, 6]
        }
        return contributions
    
    def _identify_critical_findings(self, probs, params):
        """Identifica achados críticos que requerem atenção imediata"""
        critical = []
        
        # Verificar condições críticas baseadas em probabilidades
        critical_conditions = {
            2: 'Fibrilação Ventricular',  # Assumindo índice 2
            3: 'Taquicardia Ventricular',
            14: 'STEMI',
            26: 'Bloqueio AV Completo'
        }
        
        for idx, condition in critical_conditions.items():
            if idx < probs.shape[1] and probs[0, idx] > 0.8:
                critical.append(condition)
        
        # Verificar parâmetros críticos
        hr = params[0, 0].item()
        if hr < 40 or hr > 150:
            critical.append(f'Frequência cardíaca anormal: {hr:.0f} bpm')
        
        qtc = params[0, 11].item()
        if qtc > 500:
            critical.append(f'QTc prolongado: {qtc:.0f} ms')
        
        return critical

# ==================== TRAINER OTIMIZADO ====================

class OptimizedMedicalTrainer:
    """Trainer otimizado com validação clínica aprimorada"""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Otimizador com diferentes learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': model.models.parameters(), 'lr': config.learning_rate},
            {'params': model.parameter_analyzer.parameters(), 'lr': config.learning_rate * 2},
            {'params': model.meta_learner.parameters(), 'lr': config.learning_rate * 0.5}
        ], weight_decay=1e-5)
        
        # Scheduler adaptativo
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate * 10,
            epochs=config.num_epochs,
            steps_per_epoch=100  # Será ajustado
        )
        
        # Loss otimizada
        self.criterion = ClinicallyWeightedLoss(device)
        
        # Métricas
        self.metrics_tracker = defaultdict(list)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        predictions_all = []
        targets_all = []
        
        for ecg, params, labels in tqdm(train_loader, desc="Training"):
            ecg = ecg.to(self.device)
            params = params.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            with autocast():
                outputs = self.model(ecg, params)
                loss = self.criterion(outputs, labels, params)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            predictions_all.append(torch.sigmoid(outputs).detach().cpu())
            targets_all.append(labels.cpu())
        
        # Métricas
        predictions_all = torch.cat(predictions_all)
        targets_all = torch.cat(targets_all)
        
        metrics = self.calculate_comprehensive_metrics(predictions_all, targets_all)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        predictions_all = []
        targets_all = []
        interpretations = []
        
        with torch.no_grad():
            for ecg, params, labels in tqdm(val_loader, desc="Validation"):
                ecg = ecg.to(self.device)
                params = params.to(self.device)
                labels = labels.to(self.device)
                
                # Forward com interpretação
                outputs, interpretation = self.model(ecg, params, return_interpretation=True)
                loss = self.criterion(outputs, labels, params)
                
                total_loss += loss.item()
                predictions_all.append(torch.sigmoid(outputs).cpu())
                targets_all.append(labels.cpu())
                interpretations.append(interpretation)
        
        # Métricas
        predictions_all = torch.cat(predictions_all)
        targets_all = torch.cat(targets_all)
        
        metrics = self.calculate_comprehensive_metrics(predictions_all, targets_all)
        metrics['loss'] = total_loss / len(val_loader)
        
        # Análise de interpretações
        self.analyze_interpretations(interpretations, metrics)
        
        return metrics
    
    def calculate_comprehensive_metrics(self, predictions, targets):
        """Calcula métricas médicas abrangentes"""
        metrics = {}
        
        # Threshold otimizado por classe
        thresholds = self.optimize_thresholds_medical(predictions, targets)
        binary_predictions = predictions > thresholds.unsqueeze(0)
        
        # Métricas globais
        metrics['accuracy'] = accuracy_score(targets.flatten(), binary_predictions.flatten())
        metrics['macro_f1'] = f1_score(targets, binary_predictions, average='macro')
        metrics['weighted_f1'] = f1_score(targets, binary_predictions, average='weighted')
        
        # Métricas por severidade
        severity_metrics = self.calculate_severity_metrics(predictions, targets, binary_predictions)
        metrics.update(severity_metrics)
        
        # Métricas clínicas específicas
        clinical_metrics = self.calculate_clinical_metrics(predictions, targets)
        metrics.update(clinical_metrics)
        
        return metrics
    
    def optimize_thresholds_medical(self, predictions, targets):
        """Otimiza thresholds considerando importância clínica"""
        thresholds = []
        
        for idx in range(predictions.shape[1]):
            if targets[:, idx].sum() > 0:
                # Para condições graves, priorizar sensibilidade
                if idx in [2, 3, 14, 26]:  # Condições críticas
                    # Encontrar threshold para 95% de sensibilidade
                    sorted_preds = torch.sort(predictions[targets[:, idx] == 1, idx])[0]
                    if len(sorted_preds) > 0:
                        threshold = sorted_preds[int(0.05 * len(sorted_preds))]
                    else:
                        threshold = 0.3
                else:
                    # Para outras, otimizar F1
                    best_f1 = 0
                    best_threshold = 0.5
                    
                    for t in torch.linspace(0.1, 0.9, 20):
                        binary = predictions[:, idx] > t
                        f1 = f1_score(targets[:, idx], binary)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = t.item()
                    
                    threshold = best_threshold
            else:
                threshold = 0.5
            
            thresholds.append(threshold)
        
        return torch.tensor(thresholds)
    
    def calculate_severity_metrics(self, predictions, targets, binary_predictions):
        """Calcula métricas por nível de severidade"""
        metrics = {}
        
        # Agrupar por severidade
        from comprehensive_ecg_ensemble import COMPLETE_ECG_PATHOLOGIES
        
        severity_groups = defaultdict(list)
        for idx, (code, info) in enumerate(COMPLETE_ECG_PATHOLOGIES.items()):
            if idx < predictions.shape[1]:
                severity = info['severity']
                severity_groups[severity].append(idx)
        
        # Calcular métricas por grupo
        for severity, indices in severity_groups.items():
            if indices:
                group_targets = targets[:, indices]
                group_preds = binary_predictions[:, indices]
                
                metrics[f'severity_{severity}_f1'] = f1_score(
                    group_targets.numpy(),
                    group_preds.numpy(),
                    average='macro'
                )
                
                # Para condições graves, calcular sensibilidade média
                if severity >= 4:
                    sensitivities = []
                    for idx in indices:
                        tp = ((group_preds[:, indices.index(idx)] == 1) & 
                             (group_targets[:, indices.index(idx)] == 1)).sum()
                        fn = ((group_preds[:, indices.index(idx)] == 0) & 
                             (group_targets[:, indices.index(idx)] == 1)).sum()
                        
                        if tp + fn > 0:
                            sensitivities.append(tp.item() / (tp.item() + fn.item()))
                    
                    if sensitivities:
                        metrics[f'severity_{severity}_sensitivity'] = np.mean(sensitivities)
        
        return metrics
    
    def calculate_clinical_metrics(self, predictions, targets):
        """Calcula métricas de relevância clínica direta"""
        metrics = {}
        
        # Taxa de detecção de emergências
        emergency_conditions = [2, 3, 14, 26]  # VF, VT, STEMI, AVB3
        emergency_detected = 0
        emergency_total = 0
        
        for idx in emergency_conditions:
            if idx < predictions.shape[1]:
                detected = (predictions[:, idx] > 0.5) & (targets[:, idx] == 1)
                emergency_detected += detected.sum().item()
                emergency_total += targets[:, idx].sum().item()
        
        if emergency_total > 0:
            metrics['emergency_detection_rate'] = emergency_detected / emergency_total
        
        # Taxa de falsos alarmes
        false_alarms = 0
        total_negatives = 0
        
        for idx in emergency_conditions:
            if idx < predictions.shape[1]:
                fp = (predictions[:, idx] > 0.5) & (targets[:, idx] == 0)
                false_alarms += fp.sum().item()
                total_negatives += (targets[:, idx] == 0).sum().item()
        
        if total_negatives > 0:
            metrics['false_alarm_rate'] = false_alarms / total_negatives
        
        return metrics
    
    def analyze_interpretations(self, interpretations, metrics):
        """Analisa qualidade das interpretações clínicas"""
        if not interpretations:
            return
        
        # Agregar achados críticos
        all_critical = []
        for interp in interpretations:
            if 'critical_findings' in interp:
                all_critical.extend(interp['critical_findings'])
        
        # Contar achados mais comuns
        from collections import Counter
        critical_counts = Counter(all_critical)
        
        metrics['top_critical_findings'] = dict(critical_counts.most_common(5))
        metrics['total_critical_findings'] = len(all_critical)

class ClinicallyWeightedLoss(nn.Module):
    """Loss function que considera importância clínica e parâmetros"""
    
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Pesos baseados em severidade clínica
        from comprehensive_ecg_ensemble import COMPLETE_ECG_PATHOLOGIES
        
        severity_weights = []
        for info in COMPLETE_ECG_PATHOLOGIES.values():
            weight = (info['severity'] + 1) ** 1.5  # Peso exponencial
            severity_weights.append(weight)
        
        self.severity_weights = torch.tensor(severity_weights).float().to(device)
    
    def forward(self, predictions, targets, params):
        # Loss base
        base_loss = self.base_loss(predictions, targets)
        
        # Aplicar pesos de severidade
        weighted_loss = base_loss * self.severity_weights[:predictions.shape[1]]
        
        # Penalização adicional baseada em parâmetros
        param_penalty = self.calculate_parameter_penalty(predictions, targets, params)
        
        total_loss = weighted_loss.mean() + 0.1 * param_penalty
        
        return total_loss
    
    def calculate_parameter_penalty(self, predictions, targets, params):
        """Penaliza predições inconsistentes com parâmetros"""
        penalty = 0
        
        # Exemplo: Se HR > 150, deveria detectar taquicardia
        hr = params[:, 0]  # Heart rate
        tach_idx = 4  # Assumindo índice para taquicardia
        
        if tach_idx < predictions.shape[1]:
            should_detect_tach = hr > 150
            tach_probs = torch.sigmoid(predictions[:, tach_idx])
            
            # Penalizar se não detectar quando deveria
            penalty += torch.mean(should_detect_tach.float() * (1 - tach_probs))
        
        # Adicionar outras regras clínicas conforme necessário
        
        return penalty

# ==================== FUNÇÃO PRINCIPAL OTIMIZADA ====================

def main():
    """Função principal para treinar o sistema otimizado"""
    parser = argparse.ArgumentParser(description="Sistema ECG Otimizado com Análise Paramétrica")
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cache-params', action='store_true',
                       help='Carregar parâmetros pré-calculados do cache')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("SISTEMA ECG OTIMIZADO COM ANÁLISE PARAMÉTRICA COMPLETA")
    logger.info("="*70)
    
    # Configuração
    config = MedicalECGConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Carregar dados
    logger.info("Carregando dados...")
    data_path = Path(args.data_path)
    
    X = np.load(data_path / 'X.npy')
    Y_multi = np.load(data_path / 'Y_multilabel.npy')
    
    logger.info(f"Dados: X={X.shape}, Y={Y_multi.shape}")
    
    # Cache de parâmetros
    param_cache_file = data_path / 'parameter_cache.pkl'
    parameter_cache = {}
    
    if args.cache_params and param_cache_file.exists():
        logger.info("Carregando parâmetros do cache...")
        with open(param_cache_file, 'rb') as f:
            parameter_cache = pickle.load(f)
        logger.info(f"Cache carregado: {len(parameter_cache)} amostras")
    
    # Dividir dados
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y_multi, test_size=0.2, random_state=42
    )
    
    # Criar datasets otimizados
    logger.info("Criando datasets...")
    train_dataset = OptimizedECGDataset(
        X_train, Y_train, config, 
        parameter_cache={i: parameter_cache.get(i) for i in range(len(X_train)) if i in parameter_cache},
        is_training=True
    )
    
    val_dataset = OptimizedECGDataset(
        X_val, Y_val, config,
        parameter_cache={i: parameter_cache.get(i + len(X_train)) for i in range(len(X_val)) if (i + len(X_train)) in parameter_cache},
        is_training=False
    )
    
    # Salvar cache atualizado
    if not args.cache_params:
        logger.info("Salvando cache de parâmetros...")
        all_cache = {}
        all_cache.update(train_dataset.parameter_cache)
        all_cache.update({i + len(X_train): v for i, v in val_dataset.parameter_cache.items()})
        
        with open(param_cache_file, 'wb') as f:
            pickle.dump(all_cache, f)
        logger.info("Cache salvo!")
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Criar modelo otimizado
    logger.info("Criando modelo otimizado...")
    model = OptimizedMedicalEnsemble(Y_multi.shape[1], config)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total de parâmetros: {total_params:,}")
    
    # Trainer
    trainer = OptimizedMedicalTrainer(model, config, device)
    
    # Ajustar steps do scheduler
    trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        trainer.optimizer,
        max_lr=config.learning_rate * 10,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Treinar
    logger.info("\nIniciando treinamento otimizado...")
    best_val_score = 0
    
    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Treinar
        train_metrics = trainer.train_epoch(train_loader)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Macro F1: {train_metrics['macro_f1']:.4f}, "
                   f"Weighted F1: {train_metrics['weighted_f1']:.4f}")
        
        # Validar
        val_metrics = trainer.validate(val_loader)
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                   f"Macro F1: {val_metrics['macro_f1']:.4f}, "
                   f"Emergency Detection: {val_metrics.get('emergency_detection_rate', 0):.4f}")
        
        # Score composto
        score = val_metrics['macro_f1'] + \
                0.3 * val_metrics.get('emergency_detection_rate', 0) - \
                0.1 * val_metrics.get('false_alarm_rate', 0)
        
        if score > best_val_score:
            best_val_score = score
            
            # Salvar modelo
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': val_metrics,
                'config': config.__dict__
            }
            torch.save(checkpoint, 'optimized_ecg_ensemble_best.pth')
            logger.info(f"Novo melhor modelo! Score: {score:.4f}")
            
            # Salvar relatório detalhado
            save_clinical_report(val_metrics, epoch)
        
        # Scheduler
        trainer.scheduler.step()
    
    logger.info("\n" + "="*70)
    logger.info("TREINAMENTO CONCLUÍDO!")
    logger.info("="*70)
    logger.info("Modelo salvo: optimized_ecg_ensemble_best.pth")
    logger.info("Relatório clínico: clinical_validation_report_optimized.json")

def save_clinical_report(metrics, epoch):
    """Salva relatório clínico detalhado"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'epoch': epoch,
        'overall_metrics': {
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'accuracy': metrics['accuracy']
        },
        'clinical_metrics': {
            'emergency_detection_rate': metrics.get('emergency_detection_rate', 0),
            'false_alarm_rate': metrics.get('false_alarm_rate', 0)
        },
        'severity_analysis': {
            k: v for k, v in metrics.items() if 'severity' in k
        },
        'critical_findings': metrics.get('top_critical_findings', {})
    }
    
    with open('clinical_validation_report_optimized.json', 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
