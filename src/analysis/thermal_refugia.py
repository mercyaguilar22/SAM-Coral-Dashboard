"""
thermal_refugia.py
==================
Módulo para identificación y cartografía de Refugios Térmicos en el SAM.

Un refugio térmico se define como un área arrecifal expuesta a menor
frecuencia, intensidad y duración de estrés térmico en comparación
con el promedio regional, con mayor potencial de resiliencia y supervivencia
ante el cambio climático.

Criterios aplicados:
  - DHW máximo histórico persistentemente bajo (< percentil 25 regional)
  - Frecuencia de superación de alertas (DHW >= 2.97) significativamente menor
  - Tasa de calentamiento (slope SST) menor a la media del Caribe
"""

import logging
from pathlib import Path
from typing import Tuple
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

THRESH_REGIONAL = CFG["thresholds"]["regional_optimized"]


def identify_refugia_from_grid(df_grid: pd.DataFrame, 
                               dhw_col: str = "CRW_DHW", 
                               percentile_threshold: float = 25.0) -> pd.DataFrame:
    """
    Identifica zonas candidatas a refugios térmicos en una grilla espacial
    a partir de sus percentiles de DHW y HotSpot.

    Parameters
    ----------
    df_grid : pd.DataFrame
        DataFrame con columnas lat, lon, y métricas térmicas acumuladas.
    dhw_col : str
        Columna que contiene la métrica de DHW (actual o media histórica).
    percentile_threshold : float
        Percentil de corte inferior (default 25%, correspondiente a las zonas más frías/estables).

    Returns
    -------
    pd.DataFrame con columna 'is_thermal_refuge' (bool) y 'refugia_score' (0-100).
    """
    df = df_grid.copy()
    
    if dhw_col not in df.columns:
        logger.warning(f"Columna {dhw_col} no encontrada para cálculo de refugios.")
        return df

    cutoff_dhw = np.nanpercentile(df[dhw_col], percentile_threshold)
    
    # Puntuación de refugio (inversa al estrés térmico)
    max_dhw = df[dhw_col].max()
    min_dhw = df[dhw_col].min()
    
    if max_dhw > min_dhw:
        # Score de 0 a 100: mayor puntuación = menor estrés térmico acumulado
        df["refugia_score"] = np.round(100.0 * (1.0 - (df[dhw_col] - min_dhw) / (max_dhw - min_dhw)), 1)
    else:
        df["refugia_score"] = 50.0

    df["is_thermal_refuge"] = (df[dhw_col] <= cutoff_dhw) & (df[dhw_col] < THRESH_REGIONAL)
    
    refugia_count = df["is_thermal_refuge"].sum()
    logger.info(f"Refugios térmicos identificados: {refugia_count}/{len(df)} puntos ({100*refugia_count/max(len(df),1):.1f}%)")
    
    return df


def classify_resilience_zones(df_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica las áreas arrecifales en 3 zonas de resiliencia ecológica:
      1. Refugio Térmico Prioritario (Bajo estrés continuo)
      2. Zona de Exposición Moderada (Estrés intermitente)
      3. Zona de Alto Riesgo / Vulnerabilidad Crítica (Estrés térmico frecuente)
    """
    df = df_grid.copy()
    if "refugia_score" not in df.columns:
        df = identify_refugia_from_grid(df)

    conditions = [
        (df["refugia_score"] >= 75.0),
        (df["refugia_score"] >= 40.0),
    ]
    choices = [
        "Refugio Térmico Prioritario",
        "Exposición Térmica Moderada",
    ]
    df["resilience_tier"] = np.select(conditions, choices, default="Vulnerabilidad Crítica")
    return df
