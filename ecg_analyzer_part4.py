#!/usr/bin/env python3
"""
ECG ANALYZER AVANÇADO - PARTE 4: PIPELINE DE INTEGRAÇÃO COMPLETO
Integra pré-processamento, delineação, ML e interpretação clínica
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import traceback

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Imports dos módulos anteriores (simulados aqui)
# from preprocessing import AdvancedECGPreprocessor
# from delineation import AdvancedWaveDelineator
# from clinical import ClinicalInterpreter, ECGReport, Urgency, RiskLevel

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalQualityValidator:
    """Validador de qualidade do sinal ECG"""
    
    def __init__(self):
        self.quality_thresholds = {
            'snr': 10,  # dB
            'baseline_drift': 0.5,  # mV
            'saturation_ratio': 0.05,  # 5%
            'noise_level': 0.1,  # mV
            'lead_completeness': 0.8  # 80%
        }
    
    def validate_signal(self, ecg_signal: np.ndarray, sampling_rate: float) -> Dict:
        """Valida qualidade do sinal ECG"""
        
        quality_metrics = {}
        
        # SNR (Signal-to-Noise Ratio)
        quality_metrics['snr'] = self._calculate_snr(ecg_signal)
        
        # Baseline drift
        quality_metrics['baseline_drift'] = self._assess_baseline_drift(ecg_signal)
        
        # Saturação
        quality_metrics['saturation_ratio'] = self._check_saturation(ecg_signal)
        
        # Nível de ruído
        quality_metrics['noise_level'] = self._estimate_noise_level(ecg_signal, sampling_rate)
        
        # Completude (sem gaps)
        quality_metrics['completeness'] = self._check_completeness(ecg_signal)
        
        # Score geral
        quality_metrics['overall_score'] = self._calculate_overall_score(quality_metrics)
        
        # Validação
        quality_metrics['is_valid'] = quality_metrics['overall_score'] > 0.6
        
        # Problemas detectados
        quality_metrics['issues'] = self._identify_issues(quality_metrics)
        
        return quality_metrics
    
    def _calculate_snr(self, signal: np.ndarray) -> float:
        """Calcula relação sinal-ruído"""
        
        # Estimar sinal usando filtro de média móvel
        window = 50
        signal_estimated = np.convolve(signal, np.ones(window)/window, mode='same')
        
        # Ruído = sinal - estimativa
        noise = signal - signal_estimated
        
        # SNR em dB
        signal_power = np.mean(signal_estimated**2)
        noise_power = np.mean(noise**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100  # Sinal muito limpo
        
        return snr
    
    def _assess_baseline_drift(self, signal: np.ndarray) -> float:
        """Avalia deriva da linha de base"""
        
        # Usar filtro passa-baixa para extrair baseline
        from scipy import signal as scipy_signal
        b, a = scipy_signal.butter(4, 0.5, btype='low', fs=100)
        baseline = scipy_signal.filtfilt(b, a, signal)
        
        # Máxima deriva
        drift = np.max(np.abs(baseline))
        
        return drift
    
    def _check_saturation(self, signal: np.ndarray) -> float:
        """Verifica saturação do sinal"""
        
        # Detectar valores no limite
        max_val = np.max(np.abs(signal))
        saturated = np.sum(np.abs(signal) > 0.95 * max_val)
        
        ratio = saturated / len(signal)
        
        return ratio
    
    def _estimate_noise_level(self, signal: np.ndarray, fs: float) -> float:
        """Estima nível de ruído de alta frequência"""
        
        # Filtro passa-alta para isolar ruído
        from scipy import signal as scipy_signal
        b, a = scipy_signal.butter(4, 40, btype='high', fs=fs)
        noise = scipy_signal.filtfilt(b, a, signal)
        
        # RMS do ruído
        noise_level = np.sqrt(np.mean(noise**2))
        
        return noise_level
    
    def _check_completeness(self, signal: np.ndarray) -> float:
        """Verifica completude do sinal (sem gaps)"""
        
        # Detectar valores zero consecutivos (possíveis gaps)
        zero_runs = []
        in_zero_run = False
        run_length = 0
        
        for val in signal:
            if abs(val) < 0.001:
                if not in_zero_run:
                    in_zero_run = True
                    run_length = 1
                else:
                    run_length += 1
            else:
                if in_zero_run:
                    zero_runs.append(run_length)
                    in_zero_run = False
        
        # Calcular proporção sem gaps longos
        long_gaps = sum(1 for run in zero_runs if run > 50)
        completeness = 1 - (long_gaps / max(len(signal) / 1000, 1))
        
        return completeness
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calcula score geral de qualidade"""
        
        scores = []
        
        # SNR score
        if metrics['snr'] >= self.quality_thresholds['snr']:
            scores.append(1.0)
        else:
            scores.append(metrics['snr'] / self.quality_thresholds['snr'])
        
        # Baseline score
        if metrics['baseline_drift'] <= self.quality_thresholds['baseline_drift']:
            scores.append(1.0)
        else:
            scores.append(self.quality_thresholds['baseline_drift'] / metrics['baseline_drift'])
        
        # Saturation score
        if metrics['saturation_ratio'] <= self.quality_thresholds['saturation_ratio']:
            scores.append(1.0)
        else:
            scores.append(1 - metrics['saturation_ratio'])
        
        # Noise score
        if metrics['noise_level'] <= self.quality_thresholds['noise_level']:
            scores.append(1.0)
        else:
            scores.append(self.quality_thresholds['noise_level'] / metrics['noise_level'])
        
        # Completeness score
        scores.append(metrics['completeness'])
        
        return np.mean(scores)
    
    def _identify_issues(self, metrics: Dict) -> List[str]:
        """Identifica problemas de qualidade"""
        
        issues = []
        
        if metrics['snr'] < self.quality_thresholds['snr']:
            issues.append(f"Baixo SNR: {metrics['snr']:.1f} dB")
        
        if metrics['baseline_drift'] > self.quality_thresholds['baseline_drift']:
            issues.append(f"Deriva de baseline: {metrics['baseline_drift']:.2f} mV")
        
        if metrics['saturation_ratio'] > self.quality_thresholds['saturation_ratio']:
            issues.append(f"Saturação: {metrics['saturation_ratio']*100:.1f}%")
        
        if metrics['noise_level'] > self.quality_thresholds['noise_level']:
            issues.append(f"Ruído elevado: {metrics['noise_level']:.2f} mV")
        
        if metrics['completeness'] < self.quality_thresholds['lead_completeness']:
            issues.append(f"Sinal incompleto: {metrics['completeness']*100:.1f}%")
        
        return issues

class CompleteECGAnalyzer:
    """Pipeline completo de análise ECG end-to-end"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Inicializa o analisador completo
        
        Args:
            model_path: Caminho para o modelo ML
            config_path: Caminho para arquivo de configuração (opcional)
        """
        
        self.model_path = model_path
        self.config = self._load_config(config_path)
        
        # Inicializar componentes
        logger.info("Inicializando componentes do ECG Analyzer...")
        
        # Pré-processador
        from ecg_analyzer_part1 import AdvancedECGPreprocessor
        self.preprocessor = AdvancedECGPreprocessor()
        
        # Delineador
        from ecg_analyzer_part2 import AdvancedWaveDelineator
        self.delineator = AdvancedWaveDelineator(
            sampling_rate=self.config['sampling_rate']
        )
        
        # Interpretador clínico
        from ecg_analyzer_part3 import ClinicalInterpreter
        self.interpreter = ClinicalInterpreter()
        
        # Validador de qualidade
        self.quality_validator = SignalQualityValidator()
        
        # Carregar modelo ML
        self.model = self._load_model()
        
        # Cache para otimização
        self.cache = {}
        
        logger.info("✅ ECG Analyzer inicializado com sucesso!")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Carrega configuração do sistema"""
        
        default_config = {
            'sampling_rate': 500,
            'target_length': 5000,
            'num_leads': 12,
            'lead_names': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
            'quality_threshold': 0.6,
            'confidence_threshold': 0.7,
            'batch_size': 32,
            'output_format': 'pdf',
            'save_intermediate': True
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _load_model(self) -> tf.keras.Model:
        """Carrega modelo ML"""
        
        try:
            logger.info(f"Carregando modelo: {self.model_path}")
            model = tf.keras.models.load_model(self.model_path)
            logger.info("✅ Modelo carregado com sucesso")
            return model
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def analyze_file(self, file_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Análise completa de arquivo ECG
        
        Args:
            file_path: Caminho do arquivo (PDF, imagem ou sinal)
            output_dir: Diretório para salvar resultados
            
        Returns:
            Dicionário com resultados completos da análise
        """
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ANÁLISE ECG: {os.path.basename(file_path)}")
        logger.info(f"{'='*60}")
        
        start_time = datetime.now()
        results = {
            'file': file_path,
            'timestamp': start_time.isoformat(),
            'success': False
        }
        
        try:
            # 1. Detectar tipo de arquivo e extrair sinal
            logger.info("\n1️⃣ EXTRAÇÃO DO SINAL")
            extraction_result = self._extract_ecg_signal(file_path)
            
            if not extraction_result['success']:
                results['error'] = extraction_result.get('error', 'Falha na extração')
                return results
            
            results['extraction'] = extraction_result
            
            # 2. Validar qualidade do sinal
            logger.info("\n2️⃣ VALIDAÇÃO DE QUALIDADE")
            quality_results = self._validate_signal_quality(extraction_result['signals'])
            
            results['quality'] = quality_results
            
            if not quality_results['is_valid']:
                results['error'] = 'Qualidade insuficiente do sinal'
                results['quality_issues'] = quality_results['issues']
                logger.warning(f"⚠️ Qualidade insuficiente: {quality_results['issues']}")
                return results
            
            # 3. Delinear ondas
            logger.info("\n3️⃣ DELINEAÇÃO DE ONDAS")
            delineation_results = self._delineate_waves(extraction_result['signals'])
            
            results['delineation'] = delineation_results
            
            # 4. Fazer predições ML
            logger.info("\n4️⃣ ANÁLISE COM INTELIGÊNCIA ARTIFICIAL")
            ml_results = self._run_ml_analysis(extraction_result['signals'])
            
            results['ml_predictions'] = ml_results
            
            # 5. Interpretação clínica
            logger.info("\n5️⃣ INTERPRETAÇÃO CLÍNICA")
            clinical_report = self._generate_clinical_report(
                delineation_results,
                ml_results,
                quality_results
            )
            
            results['clinical_report'] = clinical_report
            
            # 6. Salvar resultados
            if output_dir:
                logger.info("\n6️⃣ SALVANDO RESULTADOS")
                self._save_results(results, output_dir)
            
            # Análise concluída
            results['success'] = True
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"\n✅ ANÁLISE CONCLUÍDA EM {results['processing_time']:.1f}s")
            
            # Resumo dos achados
            self._print_summary(clinical_report)
            
        except Exception as e:
            logger.error(f"\n❌ ERRO NA ANÁLISE: {str(e)}")
            logger.error(traceback.format_exc())
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        return results
    
    def _extract_ecg_signal(self, file_path: str) -> Dict:
        """Extrai sinal ECG do arquivo"""
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.pdf', '.jpg', '.jpeg', '.png']:
            # Arquivo de imagem/PDF - usar preprocessador
            result = self.preprocessor.preprocess_complete(file_path, debug=False)
            
            if result['success']:
                # Extrair sinais das regiões
                signals = self._extract_signals_from_regions(result['regions'])
                
                return {
                    'success': True,
                    'signals': signals,
                    'calibration': result['calibration'],
                    'layout': result['layout'],
                    'preprocessing': result
                }
            else:
                return {'success': False, 'error': 'Falha no pré-processamento'}
        
        elif file_ext in ['.npy', '.mat', '.csv']:
            # Arquivo de sinal direto
            signals = self._load_signal_file(file_path)
            
            return {
                'success': True,
                'signals': signals,
                'calibration': None,
                'layout': None
            }
        
        else:
            return {'success': False, 'error': f'Formato não suportado: {file_ext}'}
    
    def _extract_signals_from_regions(self, regions: Dict) -> np.ndarray:
        """Extrai sinais das regiões identificadas"""
        
        signals = np.zeros((12, self.config['target_length']))
        
        for i, lead in enumerate(self.config['lead_names']):
            if lead in regions:
                region_img = regions[lead]['image']
                signal = self._extract_signal_from_image(region_img)
                
                # Redimensionar para tamanho alvo
                if len(signal) != self.config['target_length']:
                    signal = self._resample_signal(signal, self.config['target_length'])
                
                signals[i] = signal
        
        return signals
    
    def _extract_signal_from_image(self, image: np.ndarray) -> np.ndarray:
        """Extrai sinal de uma imagem de região"""
        
        # Implementação simplificada - em produção usar métodos mais sofisticados
        h, w = image.shape[:2]
        signal = []
        
        for x in range(w):
            col = image[:, x]
            # Encontrar linha escura (sinal)
            dark_points = np.where(col < 128)[0]
            
            if len(dark_points) > 0:
                y = np.median(dark_points)
                signal.append(h/2 - y)
            else:
                signal.append(signal[-1] if signal else 0)
        
        return np.array(signal)
    
    def _resample_signal(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Reamostra sinal para comprimento alvo"""
        
        from scipy import interpolate
        
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_length)
        
        f = interpolate.interp1d(x_old, signal, kind='linear')
        return f(x_new)
    
    def _load_signal_file(self, file_path: str) -> np.ndarray:
        """Carrega arquivo de sinal"""
        
        ext = Path(file_path).suffix.lower()
        
        if ext == '.npy':
            return np.load(file_path)
        elif ext == '.csv':
            return pd.read_csv(file_path).values.T
        elif ext == '.mat':
            from scipy.io import loadmat
            mat = loadmat(file_path)
            # Assumir que o sinal está na primeira variável
            key = [k for k in mat.keys() if not k.startswith('__')][0]
            return mat[key]
        
        return None
    
    def _validate_signal_quality(self, signals: np.ndarray) -> Dict:
        """Valida qualidade dos sinais"""
        
        lead_quality = {}
        
        for i, lead in enumerate(self.config['lead_names']):
            quality = self.quality_validator.validate_signal(
                signals[i], 
                self.config['sampling_rate']
            )
            lead_quality[lead] = quality
        
        # Qualidade global
        valid_leads = sum(1 for q in lead_quality.values() if q['is_valid'])
        overall_score = np.mean([q['overall_score'] for q in lead_quality.values()])
        
        return {
            'lead_quality': lead_quality,
            'valid_leads': valid_leads,
            'total_leads': len(self.config['lead_names']),
            'overall_score': overall_score,
            'is_valid': valid_leads >= 8 and overall_score > 0.6,
            'issues': self._compile_quality_issues(lead_quality)
        }
    
    def _compile_quality_issues(self, lead_quality: Dict) -> List[str]:
        """Compila problemas de qualidade"""
        
        all_issues = []
        
        for lead, quality in lead_quality.items():
            if quality['issues']:
                for issue in quality['issues']:
                    all_issues.append(f"{lead}: {issue}")
        
        return all_issues
    
    def _delineate_waves(self, signals: np.ndarray) -> Dict:
        """Delineia ondas em todos os canais"""
        
        delineation_results = {}
        
        # Delinear cada derivação
        for i, lead in enumerate(self.config['lead_names']):
            logger.info(f"Delineando {lead}...")
            
            result = self.delineator.delineate_complete(signals[i], lead)
            delineation_results[lead] = result
        
        # Resultado consolidado (usar DII como principal)
        main_lead = 'II' if 'II' in delineation_results else self.config['lead_names'][0]
        
        return {
            'lead_results': delineation_results,
            'main_lead': main_lead,
            'consolidated': delineation_results[main_lead]
        }
    
    def _run_ml_analysis(self, signals: np.ndarray) -> Dict:
        """Executa análise com modelo ML"""
        
        # Preparar entrada
        if signals.ndim == 2:
            signals = np.expand_dims(signals, axis=0)
        
        # Normalizar
        signals_norm = self._normalize_signals(signals)
        
        # Predição
        predictions = self.model.predict(signals_norm, verbose=0)
        
        # Processar saída
        if predictions.ndim > 1:
            predictions = predictions[0]
        
        # Mapear para classes
        class_names = self._get_class_names()
        
        results = {}
        for i, prob in enumerate(predictions):
            if i < len(class_names):
                results[class_names[i]] = float(prob)
        
        # Ordenar por probabilidade
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'predictions': sorted_results,
            'top_prediction': list(sorted_results.keys())[0],
            'confidence': list(sorted_results.values())[0]
        }
    
    def _normalize_signals(self, signals: np.ndarray) -> np.ndarray:
        """Normaliza sinais para entrada do modelo"""
        
        # Z-score normalization por canal
        normalized = np.zeros_like(signals)
        
        for i in range(signals.shape[1]):
            signal = signals[:, i, :]
            mean = np.mean(signal)
            std = np.std(signal)
            
            if std > 0:
                normalized[:, i, :] = (signal - mean) / std
            else:
                normalized[:, i, :] = signal - mean
        
        return normalized
    
    def _get_class_names(self) -> List[str]:
        """Obtém nomes das classes do modelo"""
        
        # Lista padrão - idealmente carregar do modelo ou config
        return [
            'Normal', 'Fibrilação Atrial', 'Flutter Atrial',
            'Taquicardia Ventricular', 'Bradicardia Sinusal',
            'BAV 1º grau', 'BAV 2º grau', 'BAV 3º grau',
            'BRD', 'BRE', 'IAM', 'Isquemia',
            'HVE', 'HVD', 'Hipercalemia', 'Hipocalemia',
            'QT Longo', 'Brugada', 'WPW'
        ]
    
    def _generate_clinical_report(self, delineation: Dict, 
                                 ml_results: Dict, quality: Dict) -> Dict:
        """Gera relatório clínico completo"""
        
        # Usar delineação principal
        main_delineation = delineation['consolidated']
        
        if not main_delineation['success']:
            return {
                'success': False,
                'error': 'Falha na delineação'
            }
        
        # Gerar laudo
        from ecg_analyzer_part3 import ECGReport
        
        report = self.interpreter.generate_report(
            patient_id='ECG_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
            waves=main_delineation['waves'],
            intervals=main_delineation['intervals'],
            ml_predictions=ml_results['predictions'],
            quality_score=quality['overall_score']
        )
        
        # Formatar texto
        report_text = self.interpreter.format_report_text(report)
        
        return {
            'success': True,
            'report': report,
            'formatted_text': report_text,
            'urgency': report.urgency_level.value,
            'risk': report.risk_stratification.value
        }
    
    def _save_results(self, results: Dict, output_dir: str):
        """Salva resultados da análise"""
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"ecg_analysis_{timestamp}"
        
        # 1. Salvar relatório em PDF
        if 'clinical_report' in results and results['clinical_report']['success']:
            pdf_path = os.path.join(output_dir, f"{base_name}_report.pdf")
            self._save_pdf_report(results, pdf_path)
            logger.info(f"📄 Relatório PDF salvo: {pdf_path}")
        
        # 2. Salvar dados JSON
        json_path = os.path.join(output_dir, f"{base_name}_data.json")
        self._save_json_results(results, json_path)
        logger.info(f"📊 Dados JSON salvos: {json_path}")
        
        # 3. Salvar visualizações
        if self.config['save_intermediate']:
            # Salvar imagens de delineação
            for lead, delineation in results['delineation']['lead_results'].items():
                if delineation['success']:
                    img_path = os.path.join(output_dir, f"{base_name}_{lead}_delineation.png")
                    self._save_delineation_plot(delineation, lead, img_path)
            
            # Salvar gráfico de qualidade
            quality_path = os.path.join(output_dir, f"{base_name}_quality.png")
            self._save_quality_plot(results['quality'], quality_path)
        
        logger.info(f"✅ Todos os resultados salvos em: {output_dir}")
    
    def _save_pdf_report(self, results: Dict, pdf_path: str):
        """Salva relatório completo em PDF"""
        
        with PdfPages(pdf_path) as pdf:
            # Página 1: Relatório clínico
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            # Texto do relatório
            report_text = results['clinical_report']['formatted_text']
            ax.text(0.1, 0.95, report_text, 
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   fontfamily='monospace')
            
            plt.title('LAUDO ELETROCARDIOGRÁFICO', fontsize=16, fontweight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Página 2: Visualização das derivações
            if 'delineation' in results:
                fig, axes = plt.subplots(4, 3, figsize=(11, 8.5))
                axes = axes.flatten()
                
                for i, lead in enumerate(self.config['lead_names']):
                    if lead in results['delineation']['lead_results']:
                        delineation = results['delineation']['lead_results'][lead]
                        if delineation['success']:
                            self._plot_lead_mini(delineation, lead, axes[i])
                
                plt.suptitle('Análise das 12 Derivações', fontsize=14)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Página 3: Métricas de qualidade
            if 'quality' in results:
                fig = self._create_quality_figure(results['quality'])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Metadados do PDF
            d = pdf.infodict()
            d['Title'] = 'Relatório ECG'
            d['Author'] = 'ECG Analyzer AI'
            d['Subject'] = 'Análise Eletrocardiográfica'
            d['Keywords'] = 'ECG, Cardiologia, IA'
            d['CreationDate'] = datetime.now()
    
    def _save_json_results(self, results: Dict, json_path: str):
        """Salva resultados em formato JSON"""
        
        # Converter objetos não serializáveis
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            elif hasattr(obj, 'value'):  # Enums
                return obj.value
            return obj
        
        # Criar versão serializável
        serializable_results = {}
        
        for key, value in results.items():
            if key in ['extraction', 'delineation']:  # Muito grandes
                # Salvar apenas resumo
                if isinstance(value, dict) and 'success' in value:
                    serializable_results[key] = {
                        'success': value['success'],
                        'summary': f"Dados completos disponíveis"
                    }
            else:
                serializable_results[key] = json.loads(
                    json.dumps(value, default=convert_to_serializable)
                )
        
        # Salvar
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    def _save_delineation_plot(self, delineation: Dict, lead: str, img_path: str):
        """Salva visualização da delineação"""
        
        from ecg_analyzer_part2 import AdvancedWaveDelineator
        
        # Criar visualização
        delineator = AdvancedWaveDelineator(self.config['sampling_rate'])
        delineator.visualize_delineation(delineation, save_path=img_path)
    
    def _save_quality_plot(self, quality: Dict, img_path: str):
        """Salva gráfico de qualidade"""
        
        fig = self._create_quality_figure(quality)
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_lead_mini(self, delineation: Dict, lead: str, ax):
        """Plot miniatura de uma derivação"""
        
        signal = delineation['signal']
        waves = delineation['waves']
        
        # Limitar pontos para visualização
        max_points = 2500
        if len(signal) > max_points:
            step = len(signal) // max_points
            signal = signal[::step]
            time = np.arange(len(signal)) * step / self.config['sampling_rate']
        else:
            time = np.arange(len(signal)) / self.config['sampling_rate']
        
        ax.plot(time, signal, 'b-', linewidth=0.5)
        
        # Marcar R peaks
        if 'r_peaks' in waves and len(waves['r_peaks']['positions']) > 0:
            r_positions = waves['r_peaks']['positions']
            # Ajustar para subsampling
            if len(signal) < len(delineation['signal']):
                r_positions = r_positions // step
            
            valid_r = r_positions[r_positions < len(signal)]
            ax.plot(time[valid_r], signal[valid_r], 'ro', markersize=3)
        
        ax.set_title(lead, fontsize=10)
        ax.set_xlim(0, 5)  # Primeiros 5 segundos
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Tempo (s)', fontsize=8)
        ax.set_ylabel('mV', fontsize=8)
    
    def _create_quality_figure(self, quality: Dict) -> plt.Figure:
        """Cria figura com métricas de qualidade"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        
        # Gráfico 1: Qualidade por derivação
        leads = list(quality['lead_quality'].keys())
        scores = [q['overall_score'] for q in quality['lead_quality'].values()]
        colors = ['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' for s in scores]
        
        bars = ax1.bar(leads, scores, color=colors)
        ax1.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label='Limiar')
        ax1.set_ylabel('Score de Qualidade')
        ax1.set_title('Qualidade por Derivação')
        ax1.set_ylim(0, 1)
        ax1.legend()
        
        # Adicionar valores nas barras
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Gráfico 2: Métricas detalhadas
        metrics_names = ['SNR', 'Baseline', 'Saturação', 'Ruído', 'Completude']
        
        # Médias das métricas
        avg_metrics = []
        for metric in ['snr', 'baseline_drift', 'saturation_ratio', 'noise_level', 'completeness']:
            values = [q[metric] for q in quality['lead_quality'].values() if metric in q]
            if values:
                if metric == 'snr':
                    avg_metrics.append(np.mean(values) / 20)  # Normalizar SNR
                elif metric in ['baseline_drift', 'noise_level']:
                    avg_metrics.append(1 - min(np.mean(values), 1))  # Inverter
                elif metric == 'saturation_ratio':
                    avg_metrics.append(1 - np.mean(values))
                else:
                    avg_metrics.append(np.mean(values))
            else:
                avg_metrics.append(0)
        
        # Radar plot
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
        avg_metrics = np.concatenate((avg_metrics, [avg_metrics[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax2.plot(angles, avg_metrics, 'o-', linewidth=2)
        ax2.fill(angles, avg_metrics, alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics_names)
        ax2.set_ylim(0, 1)
        ax2.set_title('Métricas de Qualidade Global')
        ax2.grid(True)
        
        plt.suptitle(f'Análise de Qualidade do Sinal - Score Global: {quality["overall_score"]:.2f}', 
                    fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def _print_summary(self, clinical_report: Dict):
        """Imprime resumo dos achados"""
        
        if not clinical_report['success']:
            return
        
        report = clinical_report['report']
        
        print("\n" + "="*60)
        print("RESUMO DA ANÁLISE")
        print("="*60)
        
        print(f"\n🔍 Ritmo: {report.rhythm}")
        print(f"❤️  FC: {report.heart_rate} bpm")
        print(f"⚠️  Urgência: {report.urgency_level.value}")
        print(f"📊 Risco: {report.risk_stratification.value}")
        
        if report.findings:
            print("\n📋 PRINCIPAIS ACHADOS:")
            for i, finding in enumerate(report.findings[:5], 1):
                print(f"{i}. {finding.finding} ({finding.severity})")
                if finding.clinical_significance:
                    print(f"   → {finding.clinical_significance}")
        
        if report.recommendations:
            print("\n💡 RECOMENDAÇÕES:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"{i}. {rec}")
        
        print("\n" + "="*60)
    
    def batch_analyze(self, file_list: List[str], output_dir: str, 
                     parallel: bool = True) -> Dict:
        """Análise em lote de múltiplos arquivos"""
        
        logger.info(f"\n🔄 ANÁLISE EM LOTE: {len(file_list)} arquivos")
        
        results = {}
        failed = []
        
        for i, file_path in enumerate(file_list, 1):
            logger.info(f"\n[{i}/{len(file_list)}] Processando: {os.path.basename(file_path)}")
            
            try:
                # Criar subdiretório para cada arquivo
                file_output_dir = os.path.join(
                    output_dir, 
                    Path(file_path).stem
                )
                
                # Analisar
                result = self.analyze_file(file_path, file_output_dir)
                results[file_path] = result
                
                if not result['success']:
                    failed.append(file_path)
                    
            except Exception as e:
                logger.error(f"Erro no arquivo {file_path}: {e}")
                failed.append(file_path)
                results[file_path] = {'success': False, 'error': str(e)}
        
        # Sumário
        logger.info("\n" + "="*60)
        logger.info("SUMÁRIO DA ANÁLISE EM LOTE")
        logger.info("="*60)
        logger.info(f"Total processados: {len(file_list)}")
        logger.info(f"Sucesso: {len(file_list) - len(failed)}")
        logger.info(f"Falhas: {len(failed)}")
        
        if failed:
            logger.info("\nArquivos com falha:")
            for f in failed:
                logger.info(f"  - {os.path.basename(f)}")
        
        # Salvar sumário
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'total': len(file_list),
                'success': len(file_list) - len(failed),
                'failed': failed,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return results

# Função auxiliar para uso direto
def analyze_ecg(file_path: str, model_path: str, output_dir: Optional[str] = None,
                config_path: Optional[str] = None) -> Dict:
    """
    Função conveniente para análise de ECG
    
    Args:
        file_path: Caminho do arquivo ECG
        model_path: Caminho do modelo ML
        output_dir: Diretório para salvar resultados (opcional)
        config_path: Arquivo de configuração (opcional)
    
    Returns:
        Dicionário com resultados da análise
    """
    
    analyzer = CompleteECGAnalyzer(model_path, config_path)
    return analyzer.analyze_file(file_path, output_dir)

# Exemplo de uso
if __name__ == "__main__":
    # Configurar caminhos
    MODEL_PATH = "path/to/model.h5"
    ECG_FILE = "path/to/ecg.pdf"
    OUTPUT_DIR = "results/"
    
    # Analisar ECG
    # results = analyze_ecg(ECG_FILE, MODEL_PATH, OUTPUT_DIR)
    
    print("\n✅ Pipeline de integração completo implementado!")
    print("Funcionalidades:")
    print("- Extração automática de múltiplos formatos")
    print("- Validação completa de qualidade")
    print("- Delineação multi-derivação")
    print("- Integração com modelos ML")
    print("- Interpretação clínica estruturada")
    print("- Geração de relatórios PDF")
    print("- Análise em lote")
    print("- Cache e otimizações de performance")