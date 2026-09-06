"""
validate.py
===========
Validación rigurosa del modelo Random Forest y calibración regional de umbrales
para el Sistema Arrecifal Mesoamericano (SAM).

Funciones principales:
  - Generar curvas ROC y Precision-Recall (PR).
  - Comparar desempeño del umbral global NOAA (4°C·sem) vs umbral regional (2.97°C·sem).
  - Comparar modelo original (RF_Model_Optimized.pkl) vs modelo corregido v2.
  - Exportar reporte de métricas científicas para sustento/defensa.
"""

import logging
from pathlib import Path
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

ORIGINAL_MODEL_PATH = Path(CFG["model"]["original_path"])
CORRECTED_MODEL_PATH = Path(CFG["model"]["corrected_path"])
THRESH_NOAA = CFG["thresholds"]["noaa_global"]
THRESH_REGIONAL = CFG["thresholds"]["regional_optimized"]


def compute_threshold_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Calcula métricas de clasificación para un umbral de decisión específico."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    return {
        "threshold": threshold,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_weighted": f1,
    }


def compare_models(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Compara el desempeño del modelo original de tesis vs el modelo corregido v2
    sobre el conjunto de datos de prueba independiente (2023-2024).
    """
    comparison = []
    
    # 1. Evaluar modelo original si existe
    if ORIGINAL_MODEL_PATH.exists():
        try:
            orig_artifact = joblib.load(ORIGINAL_MODEL_PATH)
            # Soporta formato objeto simple o diccionario
            orig_model = orig_artifact["model"] if isinstance(orig_artifact, dict) and "model" in orig_artifact else orig_artifact
            
            # Ajustar columnas si es necesario
            if hasattr(orig_model, "predict_proba"):
                y_prob_orig = orig_model.predict_proba(X_test)[:, 1]
                pr_orig = average_precision_score(y_test, y_prob_orig)
                roc_orig = roc_auc_score(y_test, y_prob_orig)
                comparison.append({
                    "Modelo": "Original (Tesis)",
                    "PR-AUC": round(pr_orig, 4),
                    "ROC-AUC": round(roc_orig, 4),
                    "Validación": "Aleatoria (Original)"
                })
        except Exception as e:
            logger.warning(f"No se pudo evaluar modelo original: {e}")

    # 2. Evaluar modelo corregido
    if CORRECTED_MODEL_PATH.exists():
        corr_artifact = joblib.load(CORRECTED_MODEL_PATH)
        corr_model = corr_artifact["model"]
        y_prob_corr = corr_model.predict_proba(X_test)[:, 1]
        
        pr_corr = average_precision_score(y_test, y_prob_corr)
        roc_corr = roc_auc_score(y_test, y_prob_corr)
        comparison.append({
            "Modelo": "Corregido v2 (Calibrado)",
            "PR-AUC": round(pr_corr, 4),
            "ROC-AUC": round(roc_corr, 4),
            "Validación": "Temporal + Espacial (GroupKFold)"
        })

    df_comp = pd.DataFrame(comparison)
    logger.info(f"\n=== COMPARATIVA DE MODELOS ===\n{df_comp.to_string(index=False)}")
    return df_comp
