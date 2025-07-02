
"""
Funções para visualização de resultados de avaliação
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    normalize: bool = False,
    title: str = "Matriz de Confusão",
    cmap=plt.cm.Blues,
    save_path: Optional[str] = None,
):
    """Plota a matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title = title + " Normalizada"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Matriz de confusão salva em {save_path}")
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    title: str = "Curva ROC",
    save_path: Optional[str] = None,
):
    """Plota a curva ROC para cada classe."""
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC curve of class {class_name} (area = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Curva ROC salva em {save_path}")
    plt.show()


