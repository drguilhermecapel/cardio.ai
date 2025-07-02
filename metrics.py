
"""
Funções para cálculo de métricas de avaliação
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from typing import Dict, List, Union
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    y_proba: Union[np.ndarray, List] = None,
    average: str = "weighted",
    labels: List[str] = None,
) -> Dict[str, float]:
    """Calcula diversas métricas de avaliação para classificação."""
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, average=average, zero_division=0)

    if y_proba is not None:
        try:
            if len(np.unique(y_true)) > 2:  # Multiclass
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
            else:  # Binary
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except ValueError as e:
            logger.warning(f"Não foi possível calcular ROC AUC: {e}")
            metrics["roc_auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    logger.info(f"Matriz de Confusão:\n{cm}")

    logger.info(f"Métricas calculadas: {metrics}")
    return metrics


