# backend/app/services/explainability_service.py
import shap
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Any, List, Dict, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ExplainabilityService:
    """
    Serviço para gerar explicações para as predições do modelo de ECG.
    Utiliza o SHAP (SHapley Additive exPlanations) para maior interpretabilidade.
    """

    def __init__(self, model: Any, training_data_summary: np.ndarray, feature_names: List[str]):
        """
        Inicializa o serviço de explicabilidade.

        Args:
            model: O modelo de ML treinado (deve ter um método `predict` ou `predict_proba`).
            training_data_summary: Um resumo dos dados de treinamento (e.g., SHAP KernelExplainer summary).
                                   Pode ser criado com `shap.kmeans(training_data, 10)`.
            feature_names: Nomes das derivações do ECG (e.g., ['DI', 'DII', ..., 'V6']).
        """
        try:
            logger.info("Inicializando ExplainabilityService...")
            
            self.model = model
            self.feature_names = feature_names
            
            # Verificar se o modelo tem o método necessário
            if hasattr(model, 'predict_proba'):
                self.predict_function = model.predict_proba
            elif hasattr(model, 'predict'):
                self.predict_function = model.predict
            else:
                raise ValueError("Modelo deve ter método 'predict' ou 'predict_proba'")
            
            # O ideal é usar um explainer adequado ao tipo de modelo.
            # KernelExplainer é agnóstico ao modelo.
            # TreeExplainer é muito mais rápido para modelos baseados em árvores (XGBoost, RandomForest).
            
            # Tentar usar TreeExplainer primeiro (mais rápido)
            try:
                self.explainer = shap.TreeExplainer(model)
                self.explainer_type = "tree"
                logger.info("✅ TreeExplainer inicializado com sucesso")
            except Exception as tree_error:
                logger.warning(f"TreeExplainer falhou: {tree_error}. Usando KernelExplainer...")
                # Fallback para KernelExplainer
                self.explainer = shap.KernelExplainer(self.predict_function, training_data_summary)
                self.explainer_type = "kernel"
                logger.info("✅ KernelExplainer inicializado com sucesso")
            
            logger.info("✅ ExplainabilityService inicializado com sucesso.")
            
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar ExplainabilityService: {e}")
            raise

    def explain_instance(self, ecg_instance: np.ndarray, class_names: List[str], 
                        max_display: int = 10) -> Dict[str, Any]:
        """
        Gera uma explicação SHAP para uma única instância de ECG.

        Args:
            ecg_instance: A amostra de ECG a ser explicada (formato esperado pelo modelo).
            class_names: Lista de nomes das classes/diagnósticos.
            max_display: Número máximo de features a exibir na explicação.

        Returns:
            Um dicionário contendo os valores SHAP e uma visualização em base64.
        """
        try:
            if ecg_instance.ndim == 1:
                ecg_instance = ecg_instance.reshape(1, -1)
                
            logger.info("Gerando valores SHAP para a instância de ECG...")
            
            # Gerar valores SHAP
            if self.explainer_type == "tree":
                shap_values = self.explainer.shap_values(ecg_instance)
            else:
                shap_values = self.explainer.shap_values(ecg_instance)
            
            logger.info("✅ Valores SHAP gerados.")

            # Processar valores SHAP baseado no tipo de saída
            if isinstance(shap_values, list):
                # Multi-classe: shap_values é uma lista
                processed_shap_values = shap_values
                expected_values = self.explainer.expected_value
            else:
                # Binário ou regressão: shap_values é um array
                processed_shap_values = [shap_values]
                expected_values = [self.explainer.expected_value]

            # Gerar visualizações
            explanations = self._generate_explanations(
                processed_shap_values, 
                expected_values,
                ecg_instance, 
                class_names, 
                max_display
            )

            return {
                "success": True,
                "shap_values": [val.tolist() for val in processed_shap_values],
                "expected_values": expected_values if isinstance(expected_values, list) else [expected_values],
                "explanations": explanations,
                "feature_names": self.feature_names[:ecg_instance.shape[1]],
                "explainer_type": self.explainer_type
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar explicação: {e}")
            return {
                "success": False,
                "error": str(e),
                "shap_values": None,
                "explanations": None
            }

    def _generate_explanations(self, shap_values: List[np.ndarray], expected_values: List[float],
                             ecg_instance: np.ndarray, class_names: List[str], 
                             max_display: int) -> Dict[str, Any]:
        """
        Gera diferentes tipos de visualizações SHAP.
        """
        explanations = {
            "force_plots": {},
            "waterfall_plots": {},
            "summary_data": {},
            "feature_importance": {}
        }
        
        try:
            # Limitar número de classes para performance
            num_classes = min(len(class_names), len(shap_values))
            
            for i in range(num_classes):
                class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
                
                try:
                    # 1. Force Plot
                    force_plot = self._create_force_plot(
                        expected_values[i] if i < len(expected_values) else expected_values[0],
                        shap_values[i],
                        ecg_instance,
                        class_name,
                        max_display
                    )
                    explanations["force_plots"][class_name] = force_plot
                    
                    # 2. Waterfall Plot
                    waterfall_plot = self._create_waterfall_plot(
                        expected_values[i] if i < len(expected_values) else expected_values[0],
                        shap_values[i],
                        ecg_instance,
                        class_name,
                        max_display
                    )
                    explanations["waterfall_plots"][class_name] = waterfall_plot
                    
                    # 3. Feature Importance
                    feature_importance = self._calculate_feature_importance(
                        shap_values[i],
                        max_display
                    )
                    explanations["feature_importance"][class_name] = feature_importance
                    
                except Exception as class_error:
                    logger.warning(f"Erro ao gerar explicação para classe {class_name}: {class_error}")
                    explanations["force_plots"][class_name] = None
                    explanations["waterfall_plots"][class_name] = None
                    explanations["feature_importance"][class_name] = None
            
            return explanations
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar visualizações: {e}")
            return explanations

    def _create_force_plot(self, expected_value: float, shap_values: np.ndarray, 
                          instance: np.ndarray, class_name: str, max_display: int) -> Optional[str]:
        """
        Cria um force plot SHAP e retorna como base64.
        """
        try:
            # Configurar matplotlib para não usar GUI
            plt.switch_backend('Agg')
            
            # Criar figura
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Obter valores para o plot
            shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
            instance_vals = instance[0] if instance.ndim > 1 else instance
            
            # Limitar número de features
            if len(shap_vals) > max_display:
                # Pegar as features mais importantes
                importance_indices = np.argsort(np.abs(shap_vals))[-max_display:]
                shap_vals = shap_vals[importance_indices]
                instance_vals = instance_vals[importance_indices]
                feature_names = [self.feature_names[i] if i < len(self.feature_names) 
                               else f"Feature_{i}" for i in importance_indices]
            else:
                feature_names = self.feature_names[:len(shap_vals)]
            
            # Criar force plot manual (versão simplificada)
            y_pos = np.arange(len(shap_vals))
            colors = ['red' if val < 0 else 'blue' for val in shap_vals]
            
            bars = ax.barh(y_pos, shap_vals, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('SHAP Value (impact on model output)')
            ax.set_title(f'SHAP Force Plot - {class_name}\nBase Value: {expected_value:.3f}')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Adicionar valores nas barras
            for i, (bar, val) in enumerate(zip(bars, shap_vals)):
                ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', ha='left' if val >= 0 else 'right', va='center', fontsize=8)
            
            plt.tight_layout()
            
            # Converter para base64
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
            plt.close(fig)
            
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            return f"data:image/png;base64,{data}"
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar force plot: {e}")
            return None

    def _create_waterfall_plot(self, expected_value: float, shap_values: np.ndarray, 
                              instance: np.ndarray, class_name: str, max_display: int) -> Optional[str]:
        """
        Cria um waterfall plot SHAP e retorna como base64.
        """
        try:
            plt.switch_backend('Agg')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Obter valores
            shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
            instance_vals = instance[0] if instance.ndim > 1 else instance
            
            # Limitar e ordenar por importância
            if len(shap_vals) > max_display:
                importance_indices = np.argsort(np.abs(shap_vals))[-max_display:]
                shap_vals = shap_vals[importance_indices]
                instance_vals = instance_vals[importance_indices]
                feature_names = [self.feature_names[i] if i < len(self.feature_names) 
                               else f"Feature_{i}" for i in importance_indices]
            else:
                feature_names = self.feature_names[:len(shap_vals)]
            
            # Ordenar por valor SHAP
            sorted_indices = np.argsort(shap_vals)
            shap_vals = shap_vals[sorted_indices]
            feature_names = [feature_names[i] for i in sorted_indices]
            
            # Criar waterfall plot
            cumulative = expected_value
            x_pos = 0
            
            # Base value
            ax.bar(x_pos, expected_value, color='gray', alpha=0.7, label='Base Value')
            ax.text(x_pos, expected_value/2, f'Base\n{expected_value:.3f}', 
                   ha='center', va='center', fontsize=8)
            x_pos += 1
            
            # Features
            for i, (name, val) in enumerate(zip(feature_names, shap_vals)):
                color = 'red' if val < 0 else 'blue'
                ax.bar(x_pos, val, bottom=cumulative, color=color, alpha=0.7)
                ax.text(x_pos, cumulative + val/2, f'{name}\n{val:.3f}', 
                       ha='center', va='center', fontsize=7, rotation=45)
                cumulative += val
                x_pos += 1
            
            # Final prediction
            ax.bar(x_pos, 0, bottom=cumulative, color='green', alpha=0.7, label='Prediction')
            ax.text(x_pos, cumulative, f'Final\n{cumulative:.3f}', 
                   ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f'SHAP Waterfall Plot - {class_name}')
            ax.set_ylabel('Model Output')
            ax.set_xticks([])
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Converter para base64
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
            plt.close(fig)
            
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            return f"data:image/png;base64,{data}"
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar waterfall plot: {e}")
            return None

    def _calculate_feature_importance(self, shap_values: np.ndarray, max_display: int) -> Dict[str, float]:
        """
        Calcula importância das features baseada nos valores SHAP.
        """
        try:
            shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
            
            # Calcular importância absoluta
            importance = np.abs(shap_vals)
            
            # Criar dicionário com nomes das features
            feature_importance = {}
            for i, imp in enumerate(importance):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}"
                feature_importance[feature_name] = float(imp)
            
            # Ordenar por importância e limitar
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True)[:max_display])
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"❌ Erro ao calcular importância das features: {e}")
            return {}

    def get_global_explanation(self, X_sample: np.ndarray, class_names: List[str], 
                              max_display: int = 10) -> Dict[str, Any]:
        """
        Gera explicação global baseada em uma amostra de dados.
        
        Args:
            X_sample: Amostra de dados para análise global
            class_names: Nomes das classes
            max_display: Número máximo de features a exibir
            
        Returns:
            Dicionário com explicação global
        """
        try:
            logger.info("Gerando explicação global...")
            
            # Limitar tamanho da amostra para performance
            if len(X_sample) > 100:
                indices = np.random.choice(len(X_sample), 100, replace=False)
                X_sample = X_sample[indices]
            
            # Gerar valores SHAP para a amostra
            if self.explainer_type == "tree":
                shap_values = self.explainer.shap_values(X_sample)
            else:
                shap_values = self.explainer.shap_values(X_sample)
            
            # Processar valores SHAP
            if not isinstance(shap_values, list):
                shap_values = [shap_values]
            
            global_explanation = {
                "feature_importance_global": {},
                "summary_plots": {}
            }
            
            for i, class_name in enumerate(class_names[:len(shap_values)]):
                # Importância global das features
                mean_abs_shap = np.mean(np.abs(shap_values[i]), axis=0)
                
                feature_importance = {}
                for j, imp in enumerate(mean_abs_shap):
                    feature_name = self.feature_names[j] if j < len(self.feature_names) else f"Feature_{j}"
                    feature_importance[feature_name] = float(imp)
                
                # Ordenar e limitar
                sorted_importance = dict(sorted(feature_importance.items(), 
                                              key=lambda x: x[1], reverse=True)[:max_display])
                global_explanation["feature_importance_global"][class_name] = sorted_importance
            
            logger.info("✅ Explicação global gerada com sucesso")
            return {
                "success": True,
                "global_explanation": global_explanation
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar explicação global: {e}")
            return {
                "success": False,
                "error": str(e)
            }

