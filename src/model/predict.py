"""
predict.py
==========
Módulo de inferencia y predicción espacial para el SAM Coral Bleaching Monitor.

Aplica el modelo Random Forest regionalmente calibrado sobre datos diarios de
NOAA Coral Reef Watch para estimar probabilidades de blanqueamiento y generar
alertas regionalizadas.
"""

import logging
from pathlib import Path
from datetime import datetime
import yaml
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

CORRECTED_MODEL_PATH = Path(CFG["model"]["corrected_path"])
ORIGINAL_MODEL_PATH = Path(CFG["model"]["original_path"])
THRESH_REGIONAL = CFG["thresholds"]["regional_optimized"]
THRESH_NOAA = CFG["thresholds"]["noaa_global"]
FEATURES = CFG["model"]["features"]


def load_active_model() -> tuple:
    """Carga el modelo activo (corregido v2 preferentemente, con fallback a original)."""
    if CORRECTED_MODEL_PATH.exists():
        logger.info(f"Cargando modelo corregido: {CORRECTED_MODEL_PATH}")
        artifact = joblib.load(CORRECTED_MODEL_PATH)
        model = artifact["model"]
        threshold = artifact.get("threshold_youden", 0.5)
        return model, threshold
    elif ORIGINAL_MODEL_PATH.exists():
        logger.warning(f"Usando modelo original de tesis: {ORIGINAL_MODEL_PATH}")
        artifact = joblib.load(ORIGINAL_MODEL_PATH)
        model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
        return model, 0.5
    else:
        raise FileNotFoundError("No se encontró ningún modelo entrenado (.pkl). Ejecuta train_rf.py primero.")


def predict_from_dataset(ds: xr.Dataset, date: datetime = None) -> gpd.GeoDataFrame:
    """
    Toma un Dataset de NOAA CRW del SAM para un día y genera predicciones
    geoespaciales con probabilidad RF y niveles de alerta.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset que contiene CRW_SST, CRW_SSTANOMALY, CRW_HOTSPOT, CRW_DHW, CRW_BAA.
    date : datetime, optional
        Fecha correspondiente a los datos.

    Returns
    -------\\
    gpd.GeoDataFrame
        Puntos geoespaciales con valores térmicos, probabilidad RF y nivel de alerta.
    """
    model, proba_thresh = load_active_model()
    
    if date is None:
        date = datetime.utcnow()

    # Convertir xarray a DataFrame tabular plano
    df = ds.to_dataframe().reset_index().dropna(subset=["CRW_SST", "CRW_DHW"])
    
    if df.empty:
        logger.warning("Dataset vacío tras eliminar valores nulos.")
        return gpd.GeoDataFrame()

    # Mapear nombres de coordenadas a lat/lon si difieren
    if "latitude" in df.columns:
        df["lat"] = df["latitude"]
    if "longitude" in df.columns:
        df["lon"] = df["longitude"]

    df["month"] = date.month
    df["year"] = date.year

    # Asegurar que todas las features requeridas existen
    X = df[FEATURES].copy()

    # Inferencia de probabilidades
    df["rf_bleaching_proba"] = model.predict_proba(X)[:, 1]
    df["rf_bleaching_pred"] = (df["rf_bleaching_proba"] >= proba_thresh).astype(int)

    from src.analysis.alerts import categorize_alerts
    df = categorize_alerts(df)
    df["alert_level"] = df["alert_category"]

    # Convertir a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )
    
    return gdf
