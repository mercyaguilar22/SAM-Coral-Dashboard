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
    Asigna categorías de alerta a cada punto o registro basándose en DHW y HotSpot,
    según la escala oficial de alerta e impacto ecológico (NOAA CRW v3.1 + Calibración Regional SAM).
    
    Escala:
      - Sin Estrés (No Stress): HotSpot <= 0 (Celeste claro)
      - Vigilancia (Watch): 0 < HotSpot < 1 (Amarillo)
      - Advertencia (Warning): HotSpot >= 1 y 0 < DHW < 2.97 (Naranja)
      - Alerta Regional SAM: 2.97 <= DHW < 4.0 (Naranja Rojizo - Detección Temprana)
      - Nivel 1 (Alert Level 1): 4.0 <= DHW < 8.0 (Rojo)
      - Nivel 2 (Alert Level 2): 8.0 <= DHW < 12.0 (Rojo Vino / Maroon)
      - Nivel 3 (Alert Level 3): 12.0 <= DHW < 16.0 (Marrón)
      - Nivel 4 (Alert Level 4): 16.0 <= DHW < 20.0 (Magenta)
      - Nivel 5 (Alert Level 5): DHW >= 20.0 (Púrpura Oscuro)
    """
    df = df_or_gdf.copy()
    
    hs_series = df["CRW_HOTSPOT"] if "CRW_HOTSPOT" in df.columns else (df["mean_hotspot"] if "mean_hotspot" in df.columns else 0.0)
    conditions = [
        (df["CRW_DHW"] >= 20.0),
        (df["CRW_DHW"] >= 16.0),
        (df["CRW_DHW"] >= 12.0),
        (df["CRW_DHW"] >= 8.0),
        (df["CRW_DHW"] >= 4.0),
        (df["CRW_DHW"] >= THRESH_REGIONAL),
        ((df["CRW_DHW"] > 0) | (hs_series >= 1.0)),
        (hs_series > 0),
    ]
    
    labels = [
        "Nivel 5 (AL5)",
        "Nivel 4 (AL4)",
        "Nivel 3 (AL3)",
        "Nivel 2 (AL2)",
        "Nivel 1 (Alerta NOAA)",
        "Alerta Regional SAM (2.97)",
        "Advertencia (Warning)",
        "Vigilancia (Watch)",
    ]
    
    df["alert_category"] = np.select(conditions, labels, default="Sin Estrés")
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
    
    early_warnings = ((df["CRW_DHW"] >= THRESH_REGIONAL) & (df["CRW_DHW"] < THRESH_NOAA)).sum()
    
    max_dhw_val = float(df["CRW_DHW_MAX"].max()) if "CRW_DHW_MAX" in df.columns else (
        float(df["max_dhw"].max()) if "max_dhw" in df.columns else float(df["CRW_DHW"].max())
    )
    
    summary = {
        "total_points": total,
        "max_dhw": max_dhw_val,
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
    """Paleta oficial de colores según la escala de impacto NOAA CRW 2024."""
    return {
        # Escala oficial estándar
        "Sin Estrés": "#bbf2f6",
        "Sin Estrés (No Stress)": "#bbf2f6",
        "No Stress": "#bbf2f6",
        "Vigilancia (Watch)": "#ffff00",
        "Vigilancia (Bleach Watch)": "#ffff00",
        "Watch": "#ffff00",
        "Advertencia (Warning)": "#f99f1b",
        "Advertencia (Bleaching Warning)": "#f99f1b",
        "Advertencia Térmica (Watch)": "#f99f1b",
        "Warning": "#f99f1b",
        "Alerta Regional SAM (2.97)": "#ff5500",
        "Alerta Regional SAM (Optimizada 2.97)": "#ff5500",
        "Alerta Regional SAM (Optimizada)": "#ff5500",
        "Alerta Regional SAM": "#ff5500",
        "Nivel 1 (Alerta NOAA)": "#ff0000",
        "Nivel 1 (Alert Level 1)": "#ff0000",
        "Nivel 1 (Riesgo Blanqueamiento Arrecifal)": "#ff0000",
        "Nivel 1 (Alerta Global NOAA)": "#ff0000",
        "Alert Level 1": "#ff0000",
        "Nivel 2 (AL2)": "#800000",
        "Nivel 2 (Alert Level 2)": "#800000",
        "Nivel 2 (Mortalidad Corales Sensibles)": "#800000",
        "Nivel 2 (Severo - Mortalidad Probable)": "#800000",
        "AL2": "#800000",
        "Nivel 3 (AL3)": "#8c4a1e",
        "Nivel 3 (Alert Level 3)": "#8c4a1e",
        "Nivel 3 (Mortalidad Multi-Especie)": "#8c4a1e",
        "AL3": "#8c4a1e",
        "Nivel 4 (AL4)": "#ff00ff",
        "Nivel 4 (Alert Level 4)": "#ff00ff",
        "Nivel 4 (Mortalidad Severa >50%)": "#ff00ff",
        "AL4": "#ff00ff",
        "Nivel 5 (AL5)": "#4b0082",
        "Nivel 5 (Alert Level 5)": "#4b0082",
        "Nivel 5 (Mortalidad Casi Completa >80%)": "#4b0082",
        "AL5": "#4b0082",
    }

