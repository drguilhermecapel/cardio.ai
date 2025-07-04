# bias_correction_techniques.py
"""
Técnicas Avançadas de Correção de Bias para Pesquisa
Implementa múltiplas estratégias para mitigar bias do modelo PTB-XL
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

logger = logging.getLogger(__name__)

class BiasCorrector:
    """Implementa técnicas avançadas de correção de bias."""
    
    def __init__(self, num_classes: int = 71):
        self.num_classes = num_classes
        self.correction_methods = {}
        self.bias_statistics = {}
        self.calibration_curves = {}
        
    def analyze_bias_patterns(self, predictions: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analisa padrões de bias nas predições do modelo.
        
        Args:
            predictions: Array de predições (N, num_classes)
            true_labels: Labels verdadeiros opcionais (N,)
            
        Returns:
            Dicionário com análise detalhada de bias
        """
        analysis = {
            'class_frequency': {},
            'prediction_entropy': [],
            'confidence_distribution': {},
            'bias_severity': {},
            'problematic_classes': []
        }
        
        # Análise de frequência por classe
        predicted_classes = np.argmax(predictions, axis=1)
        unique, counts = np.unique(predicted_classes, return_counts=True)
        
        for class_id, count in zip(unique, counts):
            frequency = count / len(predictions)
            analysis['class_frequency'][int(class_id)] = frequency
            
            # Identificar classes com bias extremo (>30% das predições)
            if frequency > 0.3:
                analysis['problematic_classes'].append(int(class_id))
        
        # Análise de entropia (diversidade de predições)
        for i, pred in enumerate(predictions):
            entropy = -np.sum(pred * np.log(pred + 1e-10))
            analysis['prediction_entropy'].append(entropy)
        
        # Estatísticas de confiança por classe
        for class_id in range(self.num_classes):
            class_confidences = predictions[:, class_id]
            analysis['confidence_distribution'][class_id] = {
                'mean': float(np.mean(class_confidences)),
                'std': float(np.std(class_confidences)),
                'max': float(np.max(class_confidences)),
                'q95': float(np.percentile(class_confidences, 95))
            }
        
        # Calcular severidade do bias
        expected_frequency = 1.0 / self.num_classes  # Distribuição uniforme esperada
        for class_id, frequency in analysis['class_frequency'].items():
            bias_ratio = frequency / expected_frequency
            analysis['bias_severity'][class_id] = bias_ratio
        
        self.bias_statistics = analysis
        return analysis
    
    def temperature_scaling(self, predictions: np.ndarray, temperature: float = 1.5) -> np.ndarray:
        """
        Aplica temperature scaling para calibrar probabilidades.
        
        Args:
            predictions: Predições originais
            temperature: Parâmetro de temperatura (>1 reduz confiança)
            
        Returns:
            Predições calibradas
        """
        # Aplicar temperature scaling
        scaled_logits = np.log(predictions + 1e-10) / temperature
        
        # Renormalizar para obter probabilidades
        exp_logits = np.exp(scaled_logits)
        calibrated_predictions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return calibrated_predictions
    
    def label_smoothing(self, predictions: np.ndarray, smoothing: float = 0.1) -> np.ndarray:
        """
        Aplica label smoothing para reduzir overconfidence.
        
        Args:
            predictions: Predições originais
            smoothing: Fator de smoothing (0.1 = 10% de smoothing)
            
        Returns:
            Predições suavizadas
        """
        # Label smoothing formula: y_smooth = (1-α) * y + α/K
        uniform_dist = np.ones_like(predictions) / self.num_classes
        smoothed_predictions = (1 - smoothing) * predictions + smoothing * uniform_dist
        
        return smoothed_predictions
    
    def frequency_rebalancing(self, predictions: np.ndarray, target_distribution: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Rebalanceia predições baseado na frequência desejada das classes.
        
        Args:
            predictions: Predições originais
            target_distribution: Distribuição alvo (padrão: uniforme)
            
        Returns:
            Predições rebalanceadas
        """
        if target_distribution is None:
            # Distribuição uniforme
            target_distribution = np.ones(self.num_classes) / self.num_classes
        
        # Calcular frequência atual
        current_frequencies = np.mean(predictions, axis=0)
        
        # Calcular fatores de correção
        correction_factors = target_distribution / (current_frequencies + 1e-10)
        
        # Aplicar correção
        corrected_predictions = predictions * correction_factors
        
        # Renormalizar
        corrected_predictions = corrected_predictions / np.sum(corrected_predictions, axis=1, keepdims=True)
        
        return corrected_predictions
    
    def confidence_thresholding(self, predictions: np.ndarray, confidence_threshold: float = 0.7) -> np.ndarray:
        """
        Aplica thresholding baseado em confiança para reduzir predições incertas.
        
        Args:
            predictions: Predições originais
            confidence_threshold: Limiar mínimo de confiança
            
        Returns:
            Predições filtradas por confiança
        """
        max_confidences = np.max(predictions, axis=1)
        
        # Criar máscara para predições com alta confiança
        high_confidence_mask = max_confidences >= confidence_threshold
        
        # Para predições de baixa confiança, usar distribuição uniforme
        uniform_predictions = np.ones_like(predictions) / self.num_classes
        
        # Combinar predições
        thresholded_predictions = np.where(
            high_confidence_mask[:, np.newaxis],
            predictions,
            uniform_predictions
        )
        
        return thresholded_predictions
    
    def ensemble_correction(self, predictions_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Combina múltiplas correções usando ensemble.
        
        Args:
            predictions_list: Lista de predições corrigidas
            weights: Pesos para cada método (padrão: uniforme)
            
        Returns:
            Predições ensemble
        """
        if weights is None:
            weights = [1.0 / len(predictions_list)] * len(predictions_list)
        
        # Normalizar pesos
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Combinar predições
        ensemble_predictions = np.zeros_like(predictions_list[0])
        for pred, weight in zip(predictions_list, weights):
            ensemble_predictions += weight * pred
        
        return ensemble_predictions
    
    def class_specific_correction(self, predictions: np.ndarray, class_corrections: Dict[int, float]) -> np.ndarray:
        """
        Aplica correções específicas para classes problemáticas.
        
        Args:
            predictions: Predições originais
            class_corrections: Dict com fatores de correção por classe
            
        Returns:
            Predições com correções específicas
        """
        corrected_predictions = predictions.copy()
        
        for class_id, correction_factor in class_corrections.items():
            if class_id < self.num_classes:
                corrected_predictions[:, class_id] *= correction_factor
        
        # Renormalizar
        corrected_predictions = corrected_predictions / np.sum(corrected_predictions, axis=1, keepdims=True)
        
        return corrected_predictions
    
    def adaptive_bias_correction(self, predictions: np.ndarray, signal_quality: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Aplica correção adaptativa baseada na qualidade do sinal.
        
        Args:
            predictions: Predições originais
            signal_quality: Scores de qualidade do sinal (0-1)
            
        Returns:
            Predições com correção adaptativa
        """
        if signal_quality is None:
            # Usar confiança como proxy para qualidade
            signal_quality = np.max(predictions, axis=1)
        
        corrected_predictions = np.zeros_like(predictions)
        
        for i, (pred, quality) in enumerate(zip(predictions, signal_quality)):
            if quality > 0.8:  # Alta qualidade - correção mínima
                alpha = 0.1
            elif quality > 0.5:  # Qualidade média - correção moderada
                alpha = 0.3
            else:  # Baixa qualidade - correção agressiva
                alpha = 0.5
            
            # Interpolar entre predição original e distribuição uniforme
            uniform_pred = np.ones(self.num_classes) / self.num_classes
            corrected_predictions[i] = (1 - alpha) * pred + alpha * uniform_pred
        
        return corrected_predictions
    
    def create_comprehensive_correction(self, predictions: np.ndarray, 
                                      signal_quality: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Aplica múltiplas técnicas de correção e retorna todas as variantes.
        
        Args:
            predictions: Predições originais
            signal_quality: Scores de qualidade do sinal
            
        Returns:
            Dicionário com diferentes versões corrigidas
        """
        corrections = {}
        
        # 1. Temperature scaling
        corrections['temperature_scaled'] = self.temperature_scaling(predictions, temperature=1.5)
        
        # 2. Label smoothing
        corrections['label_smoothed'] = self.label_smoothing(predictions, smoothing=0.1)
        
        # 3. Frequency rebalancing
        corrections['frequency_rebalanced'] = self.frequency_rebalancing(predictions)
        
        # 4. Confidence thresholding
        corrections['confidence_thresholded'] = self.confidence_thresholding(predictions, confidence_threshold=0.7)
        
        # 5. Correção específica para classe 46 (RAO/RAE)
        class_corrections = {46: 0.3}  # Reduzir significativamente classe 46
        corrections['class_specific'] = self.class_specific_correction(predictions, class_corrections)
        
        # 6. Correção adaptativa
        corrections['adaptive'] = self.adaptive_bias_correction(predictions, signal_quality)
        
        # 7. Ensemble de múltiplas correções
        ensemble_methods = [
            corrections['temperature_scaled'],
            corrections['frequency_rebalanced'],
            corrections['class_specific']
        ]
        corrections['ensemble'] = self.ensemble_correction(ensemble_methods, weights=[0.4, 0.3, 0.3])
        
        # 8. Correção conservadora (para pesquisa clínica)
        conservative_methods = [
            corrections['label_smoothed'],
            corrections['confidence_thresholded']
        ]
        corrections['conservative'] = self.ensemble_correction(conservative_methods, weights=[0.6, 0.4])
        
        return corrections
    
    def evaluate_correction_quality(self, original_predictions: np.ndarray, 
                                   corrected_predictions: np.ndarray,
                                   true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Avalia a qualidade da correção aplicada.
        
        Args:
            original_predictions: Predições originais
            corrected_predictions: Predições corrigidas
            true_labels: Labels verdadeiros (opcional)
            
        Returns:
            Métricas de qualidade da correção
        """
        metrics = {}
        
        # Entropia média (diversidade)
        original_entropy = np.mean([-np.sum(p * np.log(p + 1e-10)) for p in original_predictions])
        corrected_entropy = np.mean([-np.sum(p * np.log(p + 1e-10)) for p in corrected_predictions])
        
        metrics['entropy_improvement'] = corrected_entropy - original_entropy
        
        # Distribuição de classes
        original_dist = np.mean(original_predictions, axis=0)
        corrected_dist = np.mean(corrected_predictions, axis=0)
        
        # Uniformidade da distribuição (menor = mais uniforme)
        metrics['original_uniformity'] = np.std(original_dist)
        metrics['corrected_uniformity'] = np.std(corrected_dist)
        metrics['uniformity_improvement'] = metrics['original_uniformity'] - metrics['corrected_uniformity']
        
        # Bias na classe 46
        metrics['original_class_46_bias'] = original_dist[46] if len(original_dist) > 46 else 0
        metrics['corrected_class_46_bias'] = corrected_dist[46] if len(corrected_dist) > 46 else 0
        metrics['class_46_bias_reduction'] = metrics['original_class_46_bias'] - metrics['corrected_class_46_bias']
        
        # Se temos labels verdadeiros, calcular métricas de performance
        if true_labels is not None:
            original_accuracy = np.mean(np.argmax(original_predictions, axis=1) == true_labels)
            corrected_accuracy = np.mean(np.argmax(corrected_predictions, axis=1) == true_labels)
            
            metrics['original_accuracy'] = original_accuracy
            metrics['corrected_accuracy'] = corrected_accuracy
            metrics['accuracy_change'] = corrected_accuracy - original_accuracy
        
        return metrics
    
    def generate_correction_report(self, predictions: np.ndarray, 
                                 corrected_predictions_dict: Dict[str, np.ndarray],
                                 true_labels: Optional[np.ndarray] = None) -> str:
        """
        Gera relatório detalhado das correções aplicadas.
        
        Args:
            predictions: Predições originais
            corrected_predictions_dict: Dicionário com predições corrigidas
            true_labels: Labels verdadeiros (opcional)
            
        Returns:
            Relatório em formato string
        """
        report = ["="*80]
        report.append("RELATÓRIO DE CORREÇÃO DE BIAS - CARDIOAI PRO")
        report.append("="*80)
        report.append("")
        
        # Análise de bias original
        bias_analysis = self.analyze_bias_patterns(predictions, true_labels)
        
        report.append("📊 ANÁLISE DE BIAS ORIGINAL:")
        report.append(f"   Classes problemáticas: {bias_analysis['problematic_classes']}")
        report.append(f"   Entropia média: {np.mean(bias_analysis['prediction_entropy']):.3f}")
        
        if 46 in bias_analysis['class_frequency']:
            report.append(f"   Frequência classe 46 (RAO/RAE): {bias_analysis['class_frequency'][46]:.1%}")
        report.append("")
        
        # Avaliação de cada método de correção
        report.append("🔧 AVALIAÇÃO DOS MÉTODOS DE CORREÇÃO:")
        report.append("")
        
        for method_name, corrected_preds in corrected_predictions_dict.items():
            metrics = self.evaluate_correction_quality(predictions, corrected_preds, true_labels)
            
            report.append(f"📋 {method_name.upper()}:")
            report.append(f"   Melhoria de entropia: {metrics['entropy_improvement']:+.3f}")
            report.append(f"   Melhoria de uniformidade: {metrics['uniformity_improvement']:+.3f}")
            report.append(f"   Redução bias classe 46: {metrics['class_46_bias_reduction']:+.3f}")
            
            if 'accuracy_change' in metrics:
                report.append(f"   Mudança na acurácia: {metrics['accuracy_change']:+.3f}")
            
            report.append("")
        
        # Recomendações
        report.append("💡 RECOMENDAÇÕES PARA PESQUISA:")
        report.append("   1. Use 'ensemble' para melhor balanceamento geral")
        report.append("   2. Use 'conservative' para pesquisa clínica cautelosa")
        report.append("   3. Use 'class_specific' para focar no bias da classe 46")
        report.append("   4. Sempre documente limitações em publicações")
        report.append("   5. Valide resultados com especialistas médicos")
        report.append("")
        
        report.append("⚠️  LIMITAÇÕES:")
        report.append("   - Correções são heurísticas, não substituem retreinamento")
        report.append("   - Bias populacional ainda presente")
        report.append("   - Validação clínica necessária para uso médico")
        report.append("   - Resultados podem variar entre populações")
        report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)

def create_bias_corrector_for_ptbxl() -> BiasCorrector:
    """Cria corretor de bias específico para modelo PTB-XL."""
    return BiasCorrector(num_classes=71)

# Exemplo de uso
def example_bias_correction():
    """Exemplo de como usar o corretor de bias."""
    
    # Simular predições com bias (classe 46 dominante)
    np.random.seed(42)
    n_samples = 1000
    n_classes = 71
    
    # Criar predições enviesadas
    predictions = np.random.dirichlet([1] * n_classes, n_samples)
    # Adicionar bias extremo na classe 46
    predictions[:, 46] *= 5
    predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
    
    # Criar corretor
    corrector = create_bias_corrector_for_ptbxl()
    
    # Aplicar correções
    corrections = corrector.create_comprehensive_correction(predictions)
    
    # Gerar relatório
    report = corrector.generate_correction_report(predictions, corrections)
    print(report)
    
    return corrections

if __name__ == "__main__":
    example_bias_correction()