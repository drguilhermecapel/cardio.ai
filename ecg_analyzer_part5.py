#!/usr/bin/env python3
"""
ECG ANALYZER AVANÇADO - PARTE 5: ANÁLISE MULTIMODAL E TEMPORAL
Implementa análise de múltiplos ECGs, comparação temporal e integração multimodal
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class TemporalECGAnalyzer:
    """Análise temporal de ECGs seriados"""
    
    def __init__(self):
        self.temporal_thresholds = {
            'st_change': 0.1,  # 1mm mudança ST
            'qt_change': 40,   # 40ms mudança QT
            'qrs_change': 20,  # 20ms mudança QRS
            'hr_change': 20,   # 20 bpm mudança FC
            'rhythm_stability': 0.8  # 80% consistência
        }
        
    def analyze_temporal_changes(self, ecg_series: List[Dict]) -> Dict:
        """
        Analisa mudanças temporais em série de ECGs
        
        Args:
            ecg_series: Lista de análises ECG com timestamps
            
        Returns:
            Análise de tendências e mudanças significativas
        """
        
        # Ordenar por data
        ecg_series = sorted(ecg_series, key=lambda x: x['timestamp'])
        
        if len(ecg_series) < 2:
            return {
                'success': False,
                'error': 'Necessário pelo menos 2 ECGs para análise temporal'
            }
        
        # Extrair séries temporais
        time_series = self._extract_time_series(ecg_series)
        
        # Analisar tendências
        trends = self._analyze_trends(time_series)
        
        # Detectar mudanças significativas
        changes = self._detect_significant_changes(time_series)
        
        # Analisar estabilidade do ritmo
        rhythm_analysis = self._analyze_rhythm_stability(ecg_series)
        
        # Progressão de patologias
        pathology_progression = self._analyze_pathology_progression(ecg_series)
        
        # Gerar sumário
        summary = self._generate_temporal_summary(
            trends, changes, rhythm_analysis, pathology_progression
        )
        
        return {
            'success': True,
            'time_series': time_series,
            'trends': trends,
            'significant_changes': changes,
            'rhythm_analysis': rhythm_analysis,
            'pathology_progression': pathology_progression,
            'summary': summary,
            'risk_trajectory': self._calculate_risk_trajectory(ecg_series)
        }
    
    def _extract_time_series(self, ecg_series: List[Dict]) -> pd.DataFrame:
        """Extrai séries temporais dos parâmetros ECG"""
        
        data = []
        
        for ecg in ecg_series:
            row = {
                'timestamp': ecg['timestamp'],
                'heart_rate': ecg.get('heart_rate', np.nan),
                'pr_interval': ecg.get('intervals', {}).get('PR', {}).get('mean', np.nan),
                'qrs_duration': ecg.get('intervals', {}).get('QRS', {}).get('mean', np.nan),
                'qt_interval': ecg.get('intervals', {}).get('QT', {}).get('mean', np.nan),
                'qtc_interval': ecg.get('intervals', {}).get('QTc', {}).get('mean', np.nan),
                'st_elevation': ecg.get('st_analysis', {}).get('max_elevation', 0),
                'st_depression': ecg.get('st_analysis', {}).get('max_depression', 0),
                'rhythm': ecg.get('rhythm', 'Unknown'),
                'risk_score': ecg.get('risk_score', 0)
            }
            
            # Adicionar contagens de achados
            findings = ecg.get('findings', [])
            row['arrhythmia_count'] = sum(1 for f in findings if 'arritmia' in f.lower())
            row['ischemia_markers'] = sum(1 for f in findings if any(
                term in f.lower() for term in ['isquemia', 'infarto', 'st']
            ))
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        return df
    
    def _analyze_trends(self, time_series: pd.DataFrame) -> Dict:
        """Analisa tendências nos parâmetros"""
        
        trends = {}
        
        # Parâmetros numéricos para análise de tendência
        numeric_params = [
            'heart_rate', 'pr_interval', 'qrs_duration', 
            'qt_interval', 'qtc_interval', 'risk_score'
        ]
        
        for param in numeric_params:
            if param in time_series.columns:
                values = time_series[param].dropna()
                
                if len(values) >= 2:
                    # Regressão linear simples
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    # Calcular mudança percentual
                    if values.iloc[0] != 0:
                        percent_change = ((values.iloc[-1] - values.iloc[0]) / values.iloc[0]) * 100
                    else:
                        percent_change = 0
                    
                    # Determinar tendência
                    if p_value < 0.05:  # Estatisticamente significativo
                        if slope > 0:
                            trend = 'increasing'
                        else:
                            trend = 'decreasing'
                    else:
                        trend = 'stable'
                    
                    trends[param] = {
                        'trend': trend,
                        'slope': slope,
                        'p_value': p_value,
                        'r_squared': r_value**2,
                        'percent_change': percent_change,
                        'first_value': values.iloc[0],
                        'last_value': values.iloc[-1],
                        'mean': values.mean(),
                        'std': values.std()
                    }
        
        return trends
    
    def _detect_significant_changes(self, time_series: pd.DataFrame) -> List[Dict]:
        """Detecta mudanças significativas entre exames"""
        
        changes = []
        
        # Comparar exames consecutivos
        for i in range(1, len(time_series)):
            current = time_series.iloc[i]
            previous = time_series.iloc[i-1]
            
            # Mudanças em intervalos
            interval_changes = []
            
            # QRS
            if not pd.isna(current['qrs_duration']) and not pd.isna(previous['qrs_duration']):
                qrs_change = current['qrs_duration'] - previous['qrs_duration']
                if abs(qrs_change) > self.temporal_thresholds['qrs_change']:
                    interval_changes.append({
                        'parameter': 'QRS duration',
                        'change': qrs_change,
                        'significance': 'Possível progressão de distúrbio de condução'
                    })
            
            # QT
            if not pd.isna(current['qtc_interval']) and not pd.isna(previous['qtc_interval']):
                qt_change = current['qtc_interval'] - previous['qtc_interval']
                if abs(qt_change) > self.temporal_thresholds['qt_change']:
                    interval_changes.append({
                        'parameter': 'QTc interval',
                        'change': qt_change,
                        'significance': 'Mudança significativa - avaliar medicações/eletrólitos'
                    })
            
            # FC
            if not pd.isna(current['heart_rate']) and not pd.isna(previous['heart_rate']):
                hr_change = current['heart_rate'] - previous['heart_rate']
                if abs(hr_change) > self.temporal_thresholds['hr_change']:
                    interval_changes.append({
                        'parameter': 'Heart rate',
                        'change': hr_change,
                        'significance': 'Variação significativa da FC'
                    })
            
            # Mudança de ritmo
            if current['rhythm'] != previous['rhythm']:
                interval_changes.append({
                    'parameter': 'Rhythm',
                    'change': f"{previous['rhythm']} → {current['rhythm']}",
                    'significance': 'MUDANÇA DE RITMO DETECTADA'
                })
            
            # Novos achados isquêmicos
            if current['ischemia_markers'] > previous['ischemia_markers']:
                interval_changes.append({
                    'parameter': 'Ischemia markers',
                    'change': f"+{current['ischemia_markers'] - previous['ischemia_markers']}",
                    'significance': 'Novos marcadores de isquemia'
                })
            
            if interval_changes:
                changes.append({
                    'from_date': previous.name,
                    'to_date': current.name,
                    'days_between': (current.name - previous.name).days,
                    'changes': interval_changes
                })
        
        return changes
    
    def _analyze_rhythm_stability(self, ecg_series: List[Dict]) -> Dict:
        """Analisa estabilidade do ritmo ao longo do tempo"""
        
        rhythms = [ecg.get('rhythm', 'Unknown') for ecg in ecg_series]
        
        # Contar frequências
        rhythm_counts = pd.Series(rhythms).value_counts()
        dominant_rhythm = rhythm_counts.index[0]
        
        # Calcular estabilidade
        stability_score = rhythm_counts.iloc[0] / len(rhythms)
        
        # Detectar episódios de arritmia
        arrhythmia_episodes = []
        normal_rhythms = ['Ritmo Sinusal', 'Normal Sinus Rhythm', 'NSR']
        
        for i, (ecg, rhythm) in enumerate(zip(ecg_series, rhythms)):
            if rhythm not in normal_rhythms and rhythm != 'Unknown':
                arrhythmia_episodes.append({
                    'index': i,
                    'timestamp': ecg['timestamp'],
                    'rhythm': rhythm,
                    'heart_rate': ecg.get('heart_rate', 'N/A')
                })
        
        # Padrões de ocorrência
        patterns = self._detect_rhythm_patterns(rhythms, [ecg['timestamp'] for ecg in ecg_series])
        
        return {
            'dominant_rhythm': dominant_rhythm,
            'stability_score': stability_score,
            'rhythm_distribution': rhythm_counts.to_dict(),
            'arrhythmia_episodes': arrhythmia_episodes,
            'episode_count': len(arrhythmia_episodes),
            'patterns': patterns,
            'variability': 1 - stability_score  # Quanto maior, mais variável
        }
    
    def _detect_rhythm_patterns(self, rhythms: List[str], timestamps: List[datetime]) -> Dict:
        """Detecta padrões temporais nas arritmias"""
        
        patterns = {
            'circadian': None,
            'weekly': None,
            'clustering': None
        }
        
        if len(rhythms) < 3:
            return patterns
        
        # Converter para DataFrame para análise
        df = pd.DataFrame({
            'rhythm': rhythms,
            'timestamp': pd.to_datetime(timestamps),
            'is_arrhythmia': [r not in ['Ritmo Sinusal', 'Normal'] for r in rhythms]
        })
        
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.dayofweek
        
        # Padrão circadiano
        if len(df) >= 10:
            hourly_arrhythmia = df.groupby('hour')['is_arrhythmia'].mean()
            if hourly_arrhythmia.std() > 0.2:  # Variação significativa
                peak_hours = hourly_arrhythmia.nlargest(3).index.tolist()
                patterns['circadian'] = {
                    'detected': True,
                    'peak_hours': peak_hours,
                    'variation': hourly_arrhythmia.std()
                }
        
        # Clustering temporal (episódios agrupados)
        if sum(df['is_arrhythmia']) >= 3:
            # Calcular intervalos entre arritmias
            arrhythmia_times = df[df['is_arrhythmia']]['timestamp']
            intervals = np.diff(arrhythmia_times)
            
            if len(intervals) > 0:
                mean_interval = np.mean(intervals)
                cv = np.std(intervals) / mean_interval if mean_interval > 0 else 0
                
                patterns['clustering'] = {
                    'detected': cv < 0.5,  # Baixo CV indica clustering
                    'mean_interval_days': mean_interval.days if hasattr(mean_interval, 'days') else 0,
                    'coefficient_variation': cv
                }
        
        return patterns
    
    def _analyze_pathology_progression(self, ecg_series: List[Dict]) -> Dict:
        """Analisa progressão de patologias específicas"""
        
        progressions = {}
        
        # Rastrear condições ao longo do tempo
        conditions_timeline = {}
        
        for ecg in ecg_series:
            timestamp = ecg['timestamp']
            
            # Processar predições ML
            ml_predictions = ecg.get('ml_predictions', {})
            for condition, probability in ml_predictions.items():
                if probability > 0.5:  # Threshold
                    if condition not in conditions_timeline:
                        conditions_timeline[condition] = []
                    conditions_timeline[condition].append({
                        'timestamp': timestamp,
                        'probability': probability
                    })
        
        # Analisar progressão de cada condição
        for condition, timeline in conditions_timeline.items():
            if len(timeline) >= 2:
                # Ordenar por tempo
                timeline = sorted(timeline, key=lambda x: x['timestamp'])
                
                # Calcular tendência de probabilidade
                probs = [t['probability'] for t in timeline]
                times = [t['timestamp'] for t in timeline]
                
                # Progressão
                first_prob = probs[0]
                last_prob = probs[-1]
                progression_rate = (last_prob - first_prob) / len(probs)
                
                # Status
                if last_prob > first_prob + 0.1:
                    status = 'worsening'
                elif last_prob < first_prob - 0.1:
                    status = 'improving'
                else:
                    status = 'stable'
                
                progressions[condition] = {
                    'status': status,
                    'first_detection': times[0],
                    'last_assessment': times[-1],
                    'initial_probability': first_prob,
                    'current_probability': last_prob,
                    'progression_rate': progression_rate,
                    'assessments': len(timeline),
                    'timeline': timeline
                }
        
        # Identificar novas condições
        new_conditions = []
        if len(ecg_series) >= 2:
            latest_conditions = set(ecg_series[-1].get('ml_predictions', {}).keys())
            previous_conditions = set()
            for ecg in ecg_series[:-1]:
                previous_conditions.update(ecg.get('ml_predictions', {}).keys())
            
            new_conditions = list(latest_conditions - previous_conditions)
        
        return {
            'tracked_conditions': progressions,
            'new_conditions': new_conditions,
            'improving': [c for c, p in progressions.items() if p['status'] == 'improving'],
            'worsening': [c for c, p in progressions.items() if p['status'] == 'worsening'],
            'stable': [c for c, p in progressions.items() if p['status'] == 'stable']
        }
    
    def _calculate_risk_trajectory(self, ecg_series: List[Dict]) -> Dict:
        """Calcula trajetória de risco cardiovascular"""
        
        risk_scores = []
        timestamps = []
        
        for ecg in ecg_series:
            # Calcular score de risco composto
            risk_score = 0
            
            # Fatores de risco
            if ecg.get('rhythm') != 'Ritmo Sinusal':
                risk_score += 2
            
            hr = ecg.get('heart_rate', 70)
            if hr > 100 or hr < 50:
                risk_score += 1
            
            qtc = ecg.get('intervals', {}).get('QTc', {}).get('mean', 440)
            if qtc > 460:
                risk_score += 2
            elif qtc > 500:
                risk_score += 4
            
            # Achados isquêmicos
            ischemia = ecg.get('ischemia_markers', 0)
            risk_score += ischemia * 3
            
            # Condições de alto risco
            high_risk_conditions = [
                'Taquicardia Ventricular', 'Fibrilação Atrial',
                'BAV 3º grau', 'IAM'
            ]
            
            for condition in high_risk_conditions:
                if condition in ecg.get('findings', []):
                    risk_score += 5
            
            risk_scores.append(risk_score)
            timestamps.append(ecg['timestamp'])
        
        # Calcular tendência
        if len(risk_scores) >= 2:
            x = np.arange(len(risk_scores))
            slope, _, _, p_value, _ = stats.linregress(x, risk_scores)
            
            if p_value < 0.05:
                if slope > 0.5:
                    trajectory = 'deteriorating'
                elif slope < -0.5:
                    trajectory = 'improving'
                else:
                    trajectory = 'stable'
            else:
                trajectory = 'stable'
        else:
            trajectory = 'insufficient_data'
            slope = 0
            p_value = 1
        
        return {
            'risk_scores': risk_scores,
            'timestamps': timestamps,
            'current_risk': risk_scores[-1] if risk_scores else 0,
            'trajectory': trajectory,
            'slope': slope,
            'p_value': p_value,
            'max_risk': max(risk_scores) if risk_scores else 0,
            'min_risk': min(risk_scores) if risk_scores else 0,
            'risk_category': self._categorize_risk(risk_scores[-1] if risk_scores else 0)
        }
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categoriza nível de risco"""
        
        if risk_score >= 15:
            return 'VERY_HIGH'
        elif risk_score >= 10:
            return 'HIGH'
        elif risk_score >= 5:
            return 'MODERATE'
        elif risk_score >= 2:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _generate_temporal_summary(self, trends: Dict, changes: List[Dict],
                                  rhythm: Dict, pathology: Dict) -> Dict:
        """Gera sumário da análise temporal"""
        
        summary = {
            'key_findings': [],
            'recommendations': [],
            'follow_up': 'routine'
        }
        
        # Analisar tendências preocupantes
        concerning_trends = []
        
        for param, trend_data in trends.items():
            if trend_data['trend'] == 'increasing' and param in ['qtc_interval', 'risk_score']:
                concerning_trends.append(f"{param} aumentando ({trend_data['percent_change']:.1f}%)")
            elif trend_data['trend'] == 'decreasing' and param == 'heart_rate':
                if trend_data['last_value'] < 50:
                    concerning_trends.append(f"FC diminuindo (atual: {trend_data['last_value']:.0f})")
        
        if concerning_trends:
            summary['key_findings'].append(f"Tendências preocupantes: {', '.join(concerning_trends)}")
            summary['follow_up'] = 'urgent'
        
        # Mudanças significativas
        if changes:
            total_changes = sum(len(c['changes']) for c in changes)
            summary['key_findings'].append(f"{total_changes} mudanças significativas detectadas")
            
            # Mudanças críticas
            critical_changes = []
            for change_event in changes:
                for change in change_event['changes']:
                    if 'MUDANÇA DE RITMO' in change.get('significance', ''):
                        critical_changes.append(change)
            
            if critical_changes:
                summary['follow_up'] = 'urgent'
        
        # Estabilidade do ritmo
        if rhythm['stability_score'] < 0.7:
            summary['key_findings'].append(
                f"Ritmo instável (estabilidade: {rhythm['stability_score']:.1%})"
            )
            summary['recommendations'].append("Considerar Holter 24h para melhor caracterização")
        
        # Progressão de patologias
        if pathology['worsening']:
            summary['key_findings'].append(
                f"Progressão detectada em: {', '.join(pathology['worsening'])}"
            )
            summary['follow_up'] = 'urgent'
            summary['recommendations'].append("Reavaliação cardiológica urgente")
        
        if pathology['new_conditions']:
            summary['key_findings'].append(
                f"Novas condições: {', '.join(pathology['new_conditions'])}"
            )
        
        # Recomendações baseadas em padrões
        if rhythm.get('patterns', {}).get('circadian', {}).get('detected'):
            summary['recommendations'].append(
                "Padrão circadiano detectado - avaliar triggers temporais"
            )
        
        # Se não há achados significativos
        if not summary['key_findings']:
            summary['key_findings'].append("Parâmetros estáveis ao longo do período")
            summary['recommendations'].append("Manter acompanhamento regular")
        
        return summary
    
    def visualize_temporal_analysis(self, analysis: Dict, save_path: Optional[str] = None):
        """Visualiza análise temporal"""
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        time_series = analysis['time_series']
        
        # 1. Tendência da FC
        ax = axes[0]
        if 'heart_rate' in time_series.columns:
            time_series['heart_rate'].plot(ax=ax, marker='o', linewidth=2)
            ax.set_title('Frequência Cardíaca ao Longo do Tempo')
            ax.set_ylabel('bpm')
            ax.grid(True, alpha=0.3)
        
        # 2. Intervalos
        ax = axes[1]
        intervals = ['pr_interval', 'qrs_duration', 'qtc_interval']
        for interval in intervals:
            if interval in time_series.columns:
                time_series[interval].plot(ax=ax, marker='o', label=interval, linewidth=2)
        ax.set_title('Evolução dos Intervalos')
        ax.set_ylabel('ms')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Score de risco
        ax = axes[2]
        risk_traj = analysis['risk_trajectory']
        ax.plot(risk_traj['timestamps'], risk_traj['risk_scores'], 
                'ro-', linewidth=2, markersize=8)
        ax.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Alto risco')
        ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Risco moderado')
        ax.set_title('Trajetória de Risco Cardiovascular')
        ax.set_ylabel('Score de Risco')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Distribuição de ritmos
        ax = axes[3]
        rhythm_data = analysis['rhythm_analysis']['rhythm_distribution']
        if rhythm_data:
            pd.Series(rhythm_data).plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title('Distribuição de Ritmos')
        
        # 5. Heatmap de mudanças
        ax = axes[4]
        if len(time_series) > 1:
            # Criar matriz de mudanças
            params = ['heart_rate', 'qtc_interval', 'qrs_duration']
            change_matrix = []
            
            for i in range(1, len(time_series)):
                row = []
                for param in params:
                    if param in time_series.columns:
                        current = time_series[param].iloc[i]
                        previous = time_series[param].iloc[i-1]
                        if not pd.isna(current) and not pd.isna(previous):
                            change = ((current - previous) / previous) * 100
                            row.append(change)
                        else:
                            row.append(0)
                    else:
                        row.append(0)
                change_matrix.append(row)
            
            if change_matrix:
                sns.heatmap(change_matrix, annot=True, fmt='.1f', 
                           xticklabels=params, cmap='RdBu_r', center=0,
                           ax=ax, cbar_kws={'label': '% Mudança'})
                ax.set_title('Matriz de Mudanças (%)')
                ax.set_ylabel('Intervalo entre Exames')
        
        # 6. Timeline de condições
        ax = axes[5]
        pathology = analysis['pathology_progression']['tracked_conditions']
        
        if pathology:
            y_pos = 0
            colors = plt.cm.Set3(np.linspace(0, 1, len(pathology)))
            
            for (condition, data), color in zip(pathology.items(), colors):
                times = [t['timestamp'] for t in data['timeline']]
                probs = [t['probability'] for t in data['timeline']]
                
                ax.scatter(times, [y_pos] * len(times), s=np.array(probs)*200,
                          c=[color], alpha=0.6, label=condition[:20])
                
                # Linha de tendência
                ax.plot(times, [y_pos] * len(times), color=color, alpha=0.3)
                
                y_pos += 1
            
            ax.set_ylim(-0.5, y_pos - 0.5)
            ax.set_yticks(range(y_pos))
            ax.set_yticklabels([c[:20] for c in pathology.keys()])
            ax.set_title('Timeline de Condições (tamanho = probabilidade)')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Análise Temporal ECG', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class MultimodalIntegrator:
    """Integração de dados multimodais com ECG"""
    
    def __init__(self):
        self.modality_weights = {
            'ecg': 0.4,
            'clinical': 0.3,
            'laboratory': 0.2,
            'imaging': 0.1
        }
    
    def integrate_multimodal_data(self, ecg_data: Dict, 
                                 clinical_data: Optional[Dict] = None,
                                 lab_data: Optional[Dict] = None,
                                 imaging_data: Optional[Dict] = None) -> Dict:
        """
        Integra dados de múltiplas modalidades para análise holística
        """
        
        integrated_analysis = {
            'modalities': ['ecg'],
            'risk_factors': [],
            'clinical_context': {},
            'integrated_risk_score': 0,
            'recommendations': []
        }
        
        # Processar dados ECG
        ecg_risk = self._process_ecg_data(ecg_data)
        integrated_analysis['ecg_analysis'] = ecg_risk
        
        # Integrar dados clínicos
        if clinical_data:
            clinical_risk = self._process_clinical_data(clinical_data)
            integrated_analysis['clinical_analysis'] = clinical_risk
            integrated_analysis['modalities'].append('clinical')
        
        # Integrar dados laboratoriais
        if lab_data:
            lab_risk = self._process_lab_data(lab_data)
            integrated_analysis['lab_analysis'] = lab_risk
            integrated_analysis['modalities'].append('laboratory')
        
        # Integrar dados de imagem
        if imaging_data:
            imaging_risk = self._process_imaging_data(imaging_data)
            integrated_analysis['imaging_analysis'] = imaging_risk
            integrated_analysis['modalities'].append('imaging')
        
        # Calcular score integrado
        integrated_analysis['integrated_risk_score'] = self._calculate_integrated_risk(
            integrated_analysis
        )
        
        # Gerar recomendações integradas
        integrated_analysis['recommendations'] = self._generate_integrated_recommendations(
            integrated_analysis
        )
        
        # Análise de interações
        integrated_analysis['interactions'] = self._analyze_modality_interactions(
            integrated_analysis
        )
        
        return integrated_analysis
    
    def _process_ecg_data(self, ecg_data: Dict) -> Dict:
        """Processa dados ECG para integração"""
        
        risk_factors = []
        risk_score = 0
        
        # Avaliar achados ECG
        findings = ecg_data.get('findings', [])
        
        high_risk_ecg = [
            'Fibrilação Atrial', 'Taquicardia Ventricular',
            'BAV 3º grau', 'IAMCSST', 'QT Longo'
        ]
        
        for finding in findings:
            if any(risk in finding for risk in high_risk_ecg):
                risk_factors.append(finding)
                risk_score += 3
        
        # Intervalos anormais
        intervals = ecg_data.get('intervals', {})
        
        if intervals.get('QTc', {}).get('mean', 0) > 460:
            risk_factors.append('QT prolongado')
            risk_score += 2
        
        if intervals.get('PR', {}).get('mean', 0) > 200:
            risk_factors.append('PR prolongado')
            risk_score += 1
        
        return {
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'rhythm': ecg_data.get('rhythm', 'Unknown'),
            'heart_rate': ecg_data.get('heart_rate', 0)
        }
    
    def _process_clinical_data(self, clinical_data: Dict) -> Dict:
        """Processa dados clínicos"""
        
        risk_factors = []
        risk_score = 0
        
        # Fatores de risco cardiovascular
        cv_risk_factors = {
            'hypertension': 2,
            'diabetes': 2,
            'smoking': 2,
            'dyslipidemia': 1,
            'family_history': 1,
            'obesity': 1
        }
        
        for factor, weight in cv_risk_factors.items():
            if clinical_data.get(factor):
                risk_factors.append(factor)
                risk_score += weight
        
        # Sintomas
        symptoms = clinical_data.get('symptoms', [])
        concerning_symptoms = ['chest_pain', 'dyspnea', 'syncope', 'palpitations']
        
        for symptom in concerning_symptoms:
            if symptom in symptoms:
                risk_factors.append(f"Symptom: {symptom}")
                risk_score += 2
        
        # Idade
        age = clinical_data.get('age', 0)
        if age > 65:
            risk_factors.append(f"Age > 65 ({age})")
            risk_score += 1
        
        return {
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'symptoms': symptoms,
            'comorbidities': clinical_data.get('comorbidities', [])
        }
    
    def _process_lab_data(self, lab_data: Dict) -> Dict:
        """Processa dados laboratoriais"""
        
        risk_factors = []
        risk_score = 0
        abnormal_values = {}
        
        # Biomarcadores cardíacos
        if 'troponin' in lab_data:
            value = lab_data['troponin']
            if value > 0.04:  # ng/mL
                risk_factors.append(f"Troponin elevado: {value}")
                risk_score += 4
                abnormal_values['troponin'] = value
        
        if 'bnp' in lab_data:
            value = lab_data['bnp']
            if value > 100:  # pg/mL
                risk_factors.append(f"BNP elevado: {value}")
                risk_score += 2
                abnormal_values['bnp'] = value
        
        # Eletrólitos
        electrolyte_ranges = {
            'potassium': (3.5, 5.0),
            'sodium': (135, 145),
            'calcium': (8.5, 10.5),
            'magnesium': (1.7, 2.2)
        }
        
        for electrolyte, (low, high) in electrolyte_ranges.items():
            if electrolyte in lab_data:
                value = lab_data[electrolyte]
                if value < low or value > high:
                    risk_factors.append(f"{electrolyte} anormal: {value}")
                    risk_score += 1
                    abnormal_values[electrolyte] = value
        
        return {
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'abnormal_values': abnormal_values
        }
    
    def _process_imaging_data(self, imaging_data: Dict) -> Dict:
        """Processa dados de imagem"""
        
        risk_factors = []
        risk_score = 0
        
        # Ecocardiograma
        if 'echo' in imaging_data:
            echo = imaging_data['echo']
            
            # Fração de ejeção
            ef = echo.get('ejection_fraction', 100)
            if ef < 50:
                risk_factors.append(f"FE reduzida: {ef}%")
                risk_score += 3
            
            # Hipertrofia
            if echo.get('lvh'):
                risk_factors.append("HVE no eco")
                risk_score += 1
            
            # Valvopatias
            valve_disease = echo.get('valve_disease', [])
            for valve in valve_disease:
                risk_factors.append(f"Valvopatia: {valve}")
                risk_score += 2
        
        # Angio TC / Cateterismo
        if 'coronary' in imaging_data:
            coronary = imaging_data['coronary']
            stenosis = coronary.get('max_stenosis', 0)
            
            if stenosis > 70:
                risk_factors.append(f"Estenose coronária significativa: {stenosis}%")
                risk_score += 4
            elif stenosis > 50:
                risk_factors.append(f"Estenose coronária moderada: {stenosis}%")
                risk_score += 2
        
        return {
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'findings': imaging_data
        }
    
    def _calculate_integrated_risk(self, analysis: Dict) -> float:
        """Calcula score de risco integrado"""
        
        total_risk = 0
        total_weight = 0
        
        # ECG
        if 'ecg_analysis' in analysis:
            ecg_risk = analysis['ecg_analysis']['risk_score']
            total_risk += ecg_risk * self.modality_weights['ecg']
            total_weight += self.modality_weights['ecg']
        
        # Clínico
        if 'clinical_analysis' in analysis:
            clinical_risk = analysis['clinical_analysis']['risk_score']
            total_risk += clinical_risk * self.modality_weights['clinical']
            total_weight += self.modality_weights['clinical']
        
        # Laboratório
        if 'lab_analysis' in analysis:
            lab_risk = analysis['lab_analysis']['risk_score']
            total_risk += lab_risk * self.modality_weights['laboratory']
            total_weight += self.modality_weights['laboratory']
        
        # Imagem
        if 'imaging_analysis' in analysis:
            imaging_risk = analysis['imaging_analysis']['risk_score']
            total_risk += imaging_risk * self.modality_weights['imaging']
            total_weight += self.modality_weights['imaging']
        
        # Normalizar pelo peso total
        if total_weight > 0:
            integrated_risk = total_risk / total_weight
        else:
            integrated_risk = 0
        
        # Aplicar fatores multiplicadores para combinações de alto risco
        multiplier = 1.0
        
        # ECG + Troponina elevada
        if ('ecg_analysis' in analysis and 'lab_analysis' in analysis):
            if any('IAM' in f for f in analysis['ecg_analysis']['risk_factors']):
                if 'troponin' in analysis['lab_analysis'].get('abnormal_values', {}):
                    multiplier *= 1.5
        
        # Múltiplos fatores de risco
        all_risk_factors = []
        for key in ['ecg_analysis', 'clinical_analysis', 'lab_analysis', 'imaging_analysis']:
            if key in analysis:
                all_risk_factors.extend(analysis[key]['risk_factors'])
        
        if len(all_risk_factors) > 5:
            multiplier *= 1.2
        
        return integrated_risk * multiplier
    
    def _generate_integrated_recommendations(self, analysis: Dict) -> List[str]:
        """Gera recomendações baseadas em análise integrada"""
        
        recommendations = []
        risk_score = analysis['integrated_risk_score']
        
        # Recomendações por nível de risco
        if risk_score > 10:
            recommendations.extend([
                "AVALIAÇÃO CARDIOLÓGICA URGENTE",
                "Considerar internação para monitorização",
                "ECG seriado a cada 6-12h"
            ])
        elif risk_score > 5:
            recommendations.extend([
                "Consulta cardiológica em 24-48h",
                "Repetir ECG e biomarcadores",
                "Iniciar/otimizar terapia cardiovascular"
            ])
        else:
            recommendations.extend([
                "Acompanhamento ambulatorial",
                "Controle de fatores de risco"
            ])
        
        # Recomendações específicas por achado
        
        # ECG + Clínico
        if 'ecg_analysis' in analysis and 'clinical_analysis' in analysis:
            ecg = analysis['ecg_analysis']
            clinical = analysis['clinical_analysis']
            
            if 'Fibrilação Atrial' in ecg['risk_factors'] and 'hypertension' in clinical['risk_factors']:
                recommendations.append("Iniciar anticoagulação (calcular CHA2DS2-VASc)")
            
            if 'chest_pain' in clinical.get('symptoms', []) and any('ST' in f for f in ecg['risk_factors']):
                recommendations.append("Protocolo de dor torácica - considerar cateterismo")
        
        # Lab anormal
        if 'lab_analysis' in analysis:
            lab = analysis['lab_analysis']
            
            if 'potassium' in lab.get('abnormal_values', {}):
                k_value = lab['abnormal_values']['potassium']
                if k_value < 3.5:
                    recommendations.append(f"Repor potássio urgente (K: {k_value})")
                elif k_value > 5.0:
                    recommendations.append(f"Tratar hipercalemia (K: {k_value})")
            
            if 'troponin' in lab.get('abnormal_values', {}):
                recommendations.append("Curva de troponina - repetir em 3-6h")
        
        # Imagem
        if 'imaging_analysis' in analysis:
            imaging = analysis['imaging_analysis']
            
            if any('estenose' in f.lower() for f in imaging['risk_factors']):
                recommendations.append("Avaliar indicação de revascularização")
            
            if any('FE reduzida' in f for f in imaging['risk_factors']):
                recommendations.append("Otimizar terapia para IC")
        
        # Remover duplicatas mantendo ordem
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _analyze_modality_interactions(self, analysis: Dict) -> Dict:
        """Analisa interações entre modalidades"""
        
        interactions = {
            'synergistic': [],
            'contradictory': [],
            'confirmatory': []
        }
        
        # ECG + Lab
        if 'ecg_analysis' in analysis and 'lab_analysis' in analysis:
            ecg = analysis['ecg_analysis']
            lab = analysis['lab_analysis']
            
            # QT longo + distúrbio eletrolítico
            if 'QT prolongado' in ecg['risk_factors']:
                if any(e in lab.get('abnormal_values', {}) for e in ['potassium', 'magnesium']):
                    interactions['synergistic'].append(
                        "QT prolongado + distúrbio eletrolítico (alto risco de arritmia)"
                    )
            
            # IAM suspeito + troponina
            if any('IAM' in f or 'infarto' in f.lower() for f in ecg['risk_factors']):
                if 'troponin' in lab.get('abnormal_values', {}):
                    interactions['confirmatory'].append(
                        "Alterações isquêmicas ECG confirmadas por troponina elevada"
                    )
                elif 'troponin' in lab_data and lab_data['troponin'] < 0.04:
                    interactions['contradictory'].append(
                        "Alterações ECG sugestivas de IAM mas troponina normal"
                    )
        
        # ECG + Imagem
        if 'ecg_analysis' in analysis and 'imaging_analysis' in analysis:
            ecg = analysis['ecg_analysis']
            imaging = analysis['imaging_analysis']
            
            # HVE
            if 'HVE' in ecg['risk_factors'] and 'HVE no eco' in imaging['risk_factors']:
                interactions['confirmatory'].append(
                    "HVE confirmada por ECG e ecocardiograma"
                )
        
        return interactions
    
    def generate_integrated_report(self, analysis: Dict) -> str:
        """Gera relatório integrado"""
        
        report = f"""
RELATÓRIO DE ANÁLISE CARDIOVASCULAR INTEGRADA
{'='*60}

DATA: {datetime.now().strftime('%d/%m/%Y %H:%M')}
MODALIDADES ANALISADAS: {', '.join(analysis['modalities'])}

SCORE DE RISCO INTEGRADO: {analysis['integrated_risk_score']:.1f}
CATEGORIA: {self._categorize_integrated_risk(analysis['integrated_risk_score'])}

ACHADOS POR MODALIDADE:
"""
        