"""
alerts.py
=========
Módulo para categorización y análisis de alertas de blanqueamiento coralino
en el Sistema Arrecifal Mesoamericano (SAM).

Compara y sintetiza:
  - Umbral global de NOAA (DHW >= 4.0 °C·sem)
  - Umbral regional optimizado (DHW >= 2.97 °C·sem)
  - Alerta Severa / Mortalidad (DHW >= 8.0 °C·sem)
  - Desglose por países (México, Belice, Guatemala, Honduras)
"""

import logging
from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

THRESH_NOAA = CFG["thresholds"]["noaa_global"]
THRESH_REGIONAL = CFG["thresholds"]["regional_optimized"]


def categorize_alerts(df_or_gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna categorías de alerta a cada punto o registro basándose en DHW y HotSpot.
    
    Categorías:
      - 0: Sin Estrés (DHW < 1 y HotSpot <= 0)
      - 1: Advertencia Térmica / Watch (HotSpot > 0 y DHW < 2.97)
      - 2: Alerta Regional SAM (2.97 <= DHW < 4.0) -> Detección temprana regional
      - 3: Alerta Nivel 1 NOAA (4.0 <= DHW < 8.0)
      - 4: Alerta Nivel 2 Severa (DHW >= 8.0)
    """
    df = df_or_gdf.copy()
    
    conditions = [
        (df["CRW_DHW"] >= 8.0),
        (df["CRW_DHW"] >= THRESH_NOAA),
        (df["CRW_DHW"] >= THRESH_REGIONAL),
        (df["CRW_HOTSPOT"] > 0),
    ]
    
    labels = [
        "Nivel 2 (Severo - Mortalidad Probable)",
        "Nivel 1 (Alerta Global NOAA)",
        "Alerta Regional SAM (Optimizada)",
        "Advertencia Térmica (Watch)",
    ]
    
    df["alert_category"] = np.select(conditions, labels, default="Sin Estrés")
    df["alert_code"] = np.select(
        [
            df["CRW_DHW"] >= 8.0,
            df["CRW_DHW"] >= THRESH_NOAA,
            df["CRW_DHW"] >= THRESH_REGIONAL,
            df["CRW_HOTSPOT"] > 0
        ],
        [4, 3, 2, 1],
        default=0
    )
    
    return df


def generate_alert_summary(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Genera un resumen estadístico de las alertas activas en el SAM.
    """
    if gdf.empty or "CRW_DHW" not in gdf.columns:
        return {"total_points": 0, "active_alerts": 0}
    
    df = categorize_alerts(gdf)
    
    total = len(df)
    reg_alerts = (df["CRW_DHW"] >= THRESH_REGIONAL).sum()
    noaa_alerts = (df["CRW_DHW"] >= THRESH_NOAA).sum()
    severe_alerts = (df["CRW_DHW"] >= 8.0).sum()
    
    # Diferencia de detección temprana aportada por el umbral regional
    early_warnings = ((df["CRW_DHW"] >= THRESH_REGIONAL) & (df["CRW_DHW"] < THRESH_NOAA)).sum()
    
    summary = {
        "total_points": total,
        "max_dhw": float(df["CRW_DHW"].max()),
        "mean_dhw": float(df["CRW_DHW"].mean()),
        "max_sst": float(df["CRW_SST"].max()) if "CRW_SST" in df.columns else None,
        "alerts_regional_sam": int(reg_alerts),
        "alerts_noaa_global": int(noaa_alerts),
        "alerts_severe_level2": int(severe_alerts),
        "early_detection_gain_points": int(early_warnings),
        "percent_area_under_regional_alert": round(float((reg_alerts / total) * 100), 2) if total > 0 else 0.0,
        "percent_area_under_noaa_alert": round(float((noaa_alerts / total) * 100), 2) if total > 0 else 0.0,
        "category_counts": df["alert_category"].value_counts().to_dict(),
    }
    
    return summary


def get_alert_color_map() -> Dict[str, str]:
    """Paleta estándar de colores para visualización de alertas."""
    return {
        "Sin Estrés": "#2b83ba",                             # Azul
        "Advertencia Térmica (Watch)": "#abdda4",             # Verde claro
        "Alerta Regional SAM (Optimizada)": "#fdae61",        # Naranja suave (alerta temprana)
        "Nivel 1 (Alerta Global NOAA)": "#f46d43",            # Naranja fuerte
        "Nivel 2 (Severo - Mortalidad Probable)": "#d53e4f",  # Rojo oscuro
    }
