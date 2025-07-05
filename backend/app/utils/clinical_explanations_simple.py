"""
Serviço de Explicações Clínicas Simplificado
Fornece explicações para diagnósticos de ECG
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ClinicalExplanationService:
    """
    Serviço para fornecer explicações clínicas dos diagnósticos de ECG
    """
    
    def __init__(self):
        self.explanations = {
            "1AVB": "Bloqueio atrioventricular de primeiro grau (BAV 1º grau) é um atraso na condução elétrica entre os átrios e os ventrículos, geralmente benigno.",
            "AFIB": "Fibrilação Atrial (FA) é um ritmo cardíaco irregular e frequentemente rápido que pode aumentar o risco de AVC, insuficiência cardíaca e outras complicações.",
            "AFLT": "Flutter Atrial é um tipo de arritmia em que os átrios do coração batem muito rápido, mas de forma regular. Pode levar a complicações se não tratado.",
            "CRBBB": "Bloqueio de Ramo Direito Completo (BRD) é um bloqueio na via elétrica do ventrículo direito. Pode ser um achado normal ou indicar doença cardíaca subjacente.",
            "IRBBB": "Bloqueio de Ramo Direito Incompleto (BRD Incompleto) é uma variante do BRD, geralmente considerado benigno e comum em indivíduos saudáveis.",
            "LAFB": "Bloqueio Fascicular Anterior Esquerdo (BFAE) é um bloqueio em uma das duas principais vias do ramo esquerdo. Frequentemente associado a outras condições cardíacas.",
            "LAD": "Desvio do Eixo para a Esquerda pode indicar hipertrofia ventricular esquerda ou BFAE.",
            "LPR": "Intervalo PR Curto pode estar associado a síndromes de pré-excitação, como Wolff-Parkinson-White.",
            "LQT": "Síndrome do QT Longo é um distúrbio do ritmo cardíaco que pode causar batimentos cardíacos rápidos e caóticos (arritmias).",
            "NORM": "Ritmo cardíaco e condução dentro dos limites da normalidade para a idade e sexo do paciente. Nenhuma anormalidade significativa detectada.",
            "PAC": "Contração Atrial Prematura (CAP) são batimentos cardíacos extras que se originam nos átrios. Geralmente são inofensivos.",
            "PVC": "Contração Ventricular Prematura (CVP) são batimentos cardíacos extras que se originam nos ventrículos. Podem ser benignos ou indicar um problema cardíaco.",
            "RAD": "Desvio do Eixo para a Direita pode ser normal em crianças e adultos altos e magros, mas também pode indicar doença pulmonar ou hipertrofia ventricular direita.",
            "RVE": "Sobrecarga Ventricular Direita pode ser causada por hipertensão pulmonar ou outras condições que aumentam a pressão no ventrículo direito.",
            "SA": "Arritmia Sinusal é uma variação normal do ritmo sinusal, comum em jovens, onde a frequência cardíaca varia com a respiração.",
            "SB": "Bradicardia Sinusal é uma frequência cardíaca lenta (abaixo de 60 bpm) que se origina do nó sinusal. Pode ser normal em atletas ou durante o sono.",
            "STACH": "Taquicardia Sinusal é uma frequência cardíaca rápida (acima de 100 bpm) que se origina do nó sinusal, geralmente uma resposta a estresse, febre ou exercício.",
            "SVE": "Sobrecarga Ventricular Esquerda, também conhecida como Hipertrofia Ventricular Esquerda (HVE), é o espessamento do músculo do ventrículo esquerdo.",
            "TAb": "Anormalidade da Onda T pode indicar uma variedade de condições, incluindo isquemia miocárdica ou desequilíbrios eletrolíticos.",
            "TInv": "Inversão da Onda T pode ser normal em algumas derivações, mas também pode ser um sinal de isquemia coronariana, entre outras condições.",
            "ritmo_sinusal": "Ritmo cardíaco e condução dentro dos limites da normalidade. O impulso elétrico se origina normalmente no nó sinusal e se propaga sem intercorrências."
        }
    
    def get_explanation(self, diagnosis_code: str) -> str:
        """
        Retorna a explicação clínica para um código de diagnóstico
        
        Args:
            diagnosis_code: Código do diagnóstico (ex: "AFIB", "NORM", "ritmo_sinusal")
            
        Returns:
            Explicação clínica em texto
        """
        explanation = self.explanations.get(diagnosis_code)
        
        if explanation:
            logger.info(f"Explicação encontrada para {diagnosis_code}")
            return explanation
        else:
            logger.warning(f"Explicação não encontrada para {diagnosis_code}")
            return f"Diagnóstico {diagnosis_code} requer avaliação clínica individualizada. Consulte um cardiologista para interpretação adequada."
    
    def get_all_explanations(self) -> Dict[str, str]:
        """
        Retorna todas as explicações disponíveis
        
        Returns:
            Dicionário com todos os códigos e suas explicações
        """
        return self.explanations.copy()
    
    def add_explanation(self, diagnosis_code: str, explanation: str) -> None:
        """
        Adiciona uma nova explicação ao serviço
        
        Args:
            diagnosis_code: Código do diagnóstico
            explanation: Explicação clínica
        """
        self.explanations[diagnosis_code] = explanation
        logger.info(f"Explicação adicionada para {diagnosis_code}")

