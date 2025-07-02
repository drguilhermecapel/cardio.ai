
import numpy as np
import tensorflow as tf
from hybrid_ecg_digitizer import HybridECGDigitizer
from typing import Union, Dict, List, Optional
import os
import json
from datetime import datetime

class ECGInterpreterHybrid:
    """
    Interpretador de ECG com digitizer híbrido para máxima compatibilidade
    """

    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Inicializa interpretador

        Args:
            model_path: Caminho para o modelo
            config: Configurações opcionais
        """
        # Configurações padrão
        self.config = {
            'target_length': 1000,
            'verbose': True,
            'save_digitized': False,
            'quality_threshold': 0.5
        }
        if config:
            self.config.update(config)

        # Carregar modelo
        print(f"📊 Carregando modelo de {model_path}...")
        self.model = tf.keras.models.load_model(model_path)

        # Inicializar digitizer híbrido
        self.digitizer = HybridECGDigitizer(
            target_length=self.config['target_length'],
            verbose=self.config['verbose']
        )

        # Classes do modelo (ajuste conforme necessário)
        self.class_names = [
            'Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB',
            'PAC', 'PVC', 'STD', 'STE'
        ]

        # Estatísticas
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'methods_used': {}
        }

    def analyze(self, input_source: Union[str, np.ndarray],
                save_report: bool = False) -> Dict:
        """
        Analisa ECG com relatório detalhado

        Args:
            input_source: Arquivo ou array numpy
            save_report: Se deve salvar relatório JSON

        Returns:
            Dicionário com resultados completos
        """
        start_time = datetime.now()

        try:
            # Digitalizar se necessário
            if isinstance(input_source, str):
                digitization_result = self.digitizer.digitize(input_source)
                ecg_data = digitization_result['data']
                metadata = digitization_result
            else:
                ecg_data = input_source
                metadata = {'method': 'direct_array', 'quality': {'score': 1.0}}

            # Validar qualidade
            if metadata['quality']['score'] < self.config['quality_threshold']:
                return self._create_error_response(
                    "Qualidade de digitalização insuficiente",
                    metadata
                )

            # Preparar para modelo
            ecg_input = self._prepare_for_model(ecg_data)

            # Predição
            predictions = self.model.predict(ecg_input, verbose=0)

            # Processar resultados
            results = self._process_predictions(
                predictions[0],
                ecg_data,
                metadata
            )

            # Adicionar metadados
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            results['timestamp'] = datetime.now().isoformat()

            # Atualizar estatísticas
            self._update_stats(True, metadata.get('method', 'unknown'))

            # Salvar relatório se solicitado
            if save_report:
                self._save_report(results)

            return results

        except Exception as e:
            self._update_stats(False, 'error')
            return self._create_error_response(str(e), {})

    def analyze_batch(self, file_list: List[str]) -> List[Dict]:
        """Analisa múltiplos ECGs"""
        results = []

        print(f"\n📋 Processando {len(file_list)} ECGs...")

        for i, file_path in enumerate(file_list):
            print(f"\n[{i+1}/{len(file_list)}] {os.path.basename(file_path)}")

            result = self.analyze(file_path)
            result['file'] = file_path
            results.append(result)

            # Resumo
            if result['success']:
                print(f"✅ {result['diagnosis']} ({result['confidence']:.1%})")
            else:
                print(f"❌ {result['error']}")

        # Estatísticas finais
        self._print_stats()

        return results

    def _prepare_for_model(self, ecg_data: np.ndarray) -> np.ndarray:
        """Prepara dados para o modelo"""
        # Adicionar dimensão batch
        if ecg_data.ndim == 2:
            ecg_data = np.expand_dims(ecg_data, axis=0)

        # Adicionar dimensão de canal se necessário
        if ecg_data.ndim == 3 and self.model.input_shape[-1] == 1:
            ecg_data = np.expand_dims(ecg_data, axis=-1)

        return ecg_data

    def _process_predictions(self, predictions: np.ndarray,
                           ecg_data: np.ndarray,
                           metadata: Dict) -> Dict:
        """Processa predições com análise detalhada"""
        # Top diagnóstico
        top_idx = np.argmax(predictions)
        top_conf = float(predictions[top_idx])

        # Todos os diagnósticos
        all_predictions = {}
        for i, class_name in enumerate(self.class_names):
            all_predictions[class_name] = float(predictions[i])

        # Diagnósticos significativos (>10%)
        significant = {k: v for k, v in all_predictions.items() if v > 0.1}

        # Análise adicional
        analysis = self._perform_additional_analysis(ecg_data)

        # Criar resposta completa
        result = {
            'success': True,
            'diagnosis': self.class_names[top_idx],
            'confidence': top_conf,
            'all_predictions': all_predictions,
            'significant_findings': significant,
            'digitization': {
                'method': metadata.get('method', 'unknown'),
                'quality_score': metadata.get('quality', {}).get('score', 0),
                'quality_issues': metadata.get('quality', {}).get('issues', [])
            },
            'ecg_analysis': analysis,
            'interpretation': self._generate_interpretation(
                self.class_names[top_idx],
                top_conf,
                significant
            ),
            'recommendations': self._generate_recommendations(
                self.class_names[top_idx],
                top_conf
            )
        }

        # Adicionar dados digitalizados se configurado
        if self.config['save_digitized']:
            result['digitized_data'] = ecg_data.tolist()

        return result

    def _perform_additional_analysis(self, ecg_data: np.ndarray) -> Dict:
        """Análise adicional do ECG"""
        analysis = {}

        # Frequência cardíaca estimada (usando lead II)
        lead_ii = ecg_data[1] if ecg_data.shape[0] > 1 else ecg_data[0]

        # Detectar picos R (simplificado)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(lead_ii, height=np.std(lead_ii))

        if len(peaks) > 1:
            # Assumindo 10 segundos de ECG
            samples_per_second = self.config['target_length'] / 10
            bpm = 60 / (avg_rr / samples_per_second)

            analysis['heart_rate_bpm'] = int(np.clip(bpm, 30, 200))
            analysis['rhythm_regularity'] = 1 - (np.std(rr_intervals) / avg_rr)
        else:
            analysis['heart_rate_bpm'] = 'Não detectado'
            analysis['rhythm_regularity'] = 'Não calculado'

        return analysis

    def _generate_interpretation(self, diagnosis: str,
                               confidence: float,
                               significant: Dict) -> str:
        """Gera interpretação textual"""
        interpretations = {
            'Normal': "ECG dentro dos parâmetros normais",
            'AF': "Fibrilação atrial detectada - ritmo irregular",
            'I-AVB': "Bloqueio atrioventricular de primeiro grau",
            'LBBB': "Bloqueio de ramo esquerdo",
            'RBBB': "Bloqueio de ramo direito",
            'PAC': "Contração atrial prematura",
            'PVC': "Contração ventricular prematura",
            'STD': "Depressão do segmento ST - possível isquemia",
            'STE': "Elevação do segmento ST - possível infarto agudo"
        }

        base_interp = interpretations.get(diagnosis, "Anormalidade detectada")

        # Adicionar nível de confiança
        if confidence > 0.9:
            conf_text = "com alta confiança"
        elif confidence > 0.7:
            conf_text = "com boa confiança"
        else:
            conf_text = "com confiança moderada - confirmar com especialista"

        interpretation = f"{base_interp} ({conf_text} - {confidence:.1%})"

        # Mencionar outros achados significativos
        other_findings = [k for k, v in significant.items()
                         if k != diagnosis and v > 0.2]

        if other_findings:
            interpretation += f"\n\nOutros achados possíveis: {', '.join(other_findings)}"

        return interpretation

    def _generate_recommendations(self, diagnosis: str, confidence: float) -> List[str]:
        """Gera recomendações baseadas no diagnóstico"""
        urgent_conditions = ['AF', 'STE', 'STD']

        recommendations = []

        if diagnosis in urgent_conditions:
            recommendations.append("⚠️ Procurar atendimento médico imediato")

        if confidence < 0.7:
            recommendations.append("📋 Repetir ECG para confirmação")

        if diagnosis != 'Normal':
            recommendations.append("👨‍⚕️ Consultar cardiologista para avaliação completa")

        recommendations.append("📁 Manter registro deste ECG para comparações futuras")

        return recommendations

    def _create_error_response(self, error_msg: str, metadata: Dict) -> Dict:
        """Cria resposta de erro padronizada"""
        return {
            'success': False,
            'error': error_msg,
            'digitization': metadata,
            'timestamp': datetime.now().isoformat()
        }

    def _update_stats(self, success: bool, method: str):
        """Atualiza estatísticas"""
        self.stats['total_processed'] += 1

        if success:
            self.stats['successful'] += 1
            self.stats['methods_used'][method] = \
                self.stats['methods_used'].get(method, 0) + 1
        else:
            self.stats['failed'] += 1

    def _print_stats(self):
        """Imprime estatísticas"""
        print(f"\n📊 Estatísticas:")
        print(f"Total processado: {self.stats['total_processed']}")
        print(f"Sucesso: {self.stats['successful']} ({self.stats['successful']/self.stats['total_processed']*100:.1f}%)")
        print(f"Falhas: {self.stats['failed']}")

        if self.stats['methods_used']:
            print(f"\nMétodos utilizados:")
            for method, count in self.stats['methods_used'].items():
                print(f"  {method}: {count}")

    def _save_report(self, results: Dict):
        """Salva relatório em JSON"""
        filename = f"ecg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"📄 Relatório salvo: {filename}")
