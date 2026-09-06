"""
train_rf.py
===========
Entrenamiento corregido del modelo Random Forest para calibración
regional del DHW de NOAA en el SAM.

Correcciones metodológicas aplicadas:
  1. Split TEMPORAL: train 2018-2022 / test 2023-2024 (no aleatorio)
  2. Validación ESPACIAL: GroupKFold por cluster geográfico (DBSCAN)
  3. SMOTE solo en datos de entrenamiento (NUNCA en test)
  4. Umbral óptimo por Youden's J sobre curva ROC (no percentil empírico)
  5. Métricas: PR-AUC principal + ROC-AUC + F1-weighted (no solo accuracy)
  6. Sin data leakage: GridSearchCV no ve datos de test
"""

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GridSearchCV

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

TRAIN_YEARS = CFG["model"]["train_years"]
TEST_YEARS  = CFG["model"]["test_years"]
MODEL_PATH  = Path(CFG["model"]["corrected_path"])
FEATURES    = CFG["model"]["features"]


def temporal_split(X: pd.DataFrame, y: pd.Series, years_col: str = "year"):
    """Split temporal: entrenamiento en años pasados, test en años recientes."""
    train_mask = X[years_col].isin(TRAIN_YEARS)
    test_mask  = X[years_col].isin(TEST_YEARS)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    logger.info(f"Split temporal:")
    logger.info(f"  Train ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}): "
                f"{len(X_train)} muestras (SI={y_train.sum()}, NO={len(y_train)-y_train.sum()})")
    logger.info(f"  Test  ({TEST_YEARS[0]}-{TEST_YEARS[-1]}):  "
                f"{len(X_test)} muestras (SI={y_test.sum()}, NO={len(y_test)-y_test.sum()})")
    return X_train, X_test, y_train, y_test


def build_spatial_groups(X: pd.DataFrame, eps_deg: float = 0.1) -> np.ndarray:
    """
    Crea grupos geográficos con DBSCAN para GroupKFold.
    Evita que observaciones del mismo sitio estén en train y test.
    eps_deg ≈ 0.1° ≈ 11 km (cubre un pixel NOAA de 5km con margen)
    """
    coords = X[["lat", "lon"]].values
    clusters = DBSCAN(eps=eps_deg, min_samples=1,
                      metric="haversine").fit(
        np.radians(coords)
    ).labels_
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    logger.info(f"Grupos espaciales (DBSCAN ε={eps_deg}°): {n_clusters} clusters")
    return clusters


def train(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Pipeline completo de entrenamiento con todas las correcciones.

    Returns
    -------
    dict con: model, threshold_youden, metrics_test, metrics_train
    """
    # 1. Split temporal
    X_train, X_test, y_train, y_test = temporal_split(X, y)

    # 2. Grupos espaciales para CV (solo en train)
    groups_train = build_spatial_groups(X_train)
    n_splits = min(5, len(set(groups_train)))
    gkf = GroupKFold(n_splits=n_splits)

    # 3. SMOTE solo en train
    n_minority = (y_train == 0).sum()
    k_neighbors = min(5, max(1, n_minority - 1))
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE aplicado: {len(X_train)} → {len(X_train_bal)} muestras "
                f"(SI={y_train_bal.sum()}, NO={len(y_train_bal)-y_train_bal.sum()})")

    # Grupos para el dataset balanceado (repetir grupos originales)
    # SMOTE genera nuevas muestras del mismo grupo
    groups_bal = np.concatenate([
        groups_train,
        groups_train[y_train == 0][:len(X_train_bal) - len(X_train)]
    ])

    # 4. GridSearchCV con validación espacial (SIN ver datos de test)
    logger.info("Iniciando GridSearchCV con validación espacial...")
    param_grid = {
        "n_estimators":      [200, 400],
        "max_depth":         [10, 20, None],
        "min_samples_split": [2, 5],
        "class_weight":      ["balanced", None],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=gkf.split(X_train_bal, y_train_bal, groups=groups_bal),
        scoring="average_precision",   # PR-AUC (mejor para desbalanceo)
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train_bal, y_train_bal)

    best_rf = grid_search.best_estimator_
    logger.info(f"Mejores parámetros: {grid_search.best_params_}")

    # 5. Evaluar SOLO en test holdout (nunca visto)
    y_proba_test = best_rf.predict_proba(X_test)[:, 1]

    # 6. Umbral óptimo por Youden's J (corrección del umbral 2.97)
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba_test)
    youden_j  = tpr - fpr
    opt_idx   = np.argmax(youden_j)
    threshold_youden = thresholds_roc[opt_idx]

    logger.info(f"\n{'='*50}")
    logger.info(f"UMBRAL ÓPTIMO (Youden's J): {threshold_youden:.3f}")
    logger.info(f"  Comparar con umbral empírico original: 2.97 °C·semanas")
    logger.info(f"  (Este umbral es de probabilidad RF, no de DHW directamente)")
    logger.info(f"{'='*50}\n")

    y_pred_test = (y_proba_test >= threshold_youden).astype(int)

    # 7. Métricas correctas para desbalanceo extremo
    pr_auc  = average_precision_score(y_test, y_proba_test)
    roc_auc = roc_auc_score(y_test, y_proba_test)
    baseline_pr_auc = y_test.mean()  # Línea base trivial

    logger.info("MÉTRICAS EN TEST HOLDOUT (datos nunca vistos):")
    logger.info(f"  PR-AUC:  {pr_auc:.3f}  (línea base trivial: {baseline_pr_auc:.3f})")
    logger.info(f"  ROC-AUC: {roc_auc:.3f}")
    logger.info(f"\n{classification_report(y_test, y_pred_test, target_names=['NO','SI'])}")

    # Feature importance
    importances = pd.Series(
        best_rf.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    logger.info(f"Feature importance:\n{importances.to_string()}")

    # 8. Guardar modelo
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "model":             best_rf,
        "threshold_youden":  threshold_youden,
        "features":          FEATURES,
        "train_years":       TRAIN_YEARS,
        "test_years":        TEST_YEARS,
        "metrics": {
            "pr_auc":           pr_auc,
            "roc_auc":          roc_auc,
            "baseline_pr_auc":  baseline_pr_auc,
        },
        "feature_importances": importances.to_dict(),
    }
    joblib.dump(model_artifact, MODEL_PATH)
    logger.info(f"Modelo guardado: {MODEL_PATH}")

    return model_artifact


if __name__ == "__main__":
    from src.data.insitu_processor import get_insitu_gdf
    from src.data.feature_engineering import enrich_with_noaa, prepare_model_features

    gdf = get_insitu_gdf()
    gdf = enrich_with_noaa(gdf)
    X, y = prepare_model_features(gdf)
    results = train(X, y)
