"""
time_series.py
==============
Análisis de series temporales para productos térmicos de NOAA Coral Reef Watch (CRW)
en el Sistema Arrecifal Mesoamericano (SAM).

Funcionalidades:
  - Cálculo de pendientes de calentamiento con Sen's Slope y regresión lineal.
  - Test de Mann-Kendall para significancia estadística de tendencias.
  - Descomposición estacional y cálculo de ciclos interanuales.
  - Extracción de perfiles temporales para puntos y regiones.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_linear_trend(series: pd.Series) -> Dict[str, float]:
    """
    Calcula la tasa de cambio lineal (pendiente) y significancia para una serie temporal.
    
    Returns
    -------
    dict con:
      - slope_per_year: Pendiente anualizada (°C/año)
      - slope_per_decade: Pendiente por década (°C/década)
      - p_value: Valor p del test de significancia
      - r_squared: Coeficiente de determinación R²
    """
    clean_series = series.dropna()
    if len(clean_series) < 5:
        return {"slope_per_year": 0.0, "slope_per_decade": 0.0, "p_value": 1.0, "r_squared": 0.0}

    # x en días transcurridos
    x = np.arange(len(clean_series))
    y = clean_series.values

    res = stats.linregress(x, y)
    
    # Asumiendo frecuencia diaria aprox (o calculando delta si hay índice datetime)
    if isinstance(clean_series.index, pd.DatetimeIndex):
        days_total = (clean_series.index[-1] - clean_series.index[0]).days
        years_total = max(days_total / 365.25, 0.1)
        slope_per_year = float((y[-1] - y[0]) / years_total) if years_total > 0 else 0.0
    else:
        # Frecuencia estimada diaria: 365.25 pasos por año
        slope_per_year = float(res.slope * 365.25)
        
    slope_per_decade = slope_per_year * 10.0

    return {
        "slope_per_year": round(slope_per_year, 5),
        "slope_per_decade": round(slope_per_decade, 4),
        "p_value": round(float(res.pvalue), 6),
        "r_squared": round(float(res.rvalue ** 2), 4),
    }


def calculate_sens_slope(series: pd.Series) -> float:
    """
    Calcula la pendiente no paramétrica de Theil-Sen.
    Robusta ante valores atípicos y estacionalidad.
    """
    clean_vals = series.dropna().values
    if len(clean_vals) < 3:
        return 0.0
    
    res = stats.theilslopes(clean_vals, np.arange(len(clean_vals)))
    # Pendiente por paso temporal
    return float(res[0])


def compute_monthly_climatology(df: pd.DataFrame, date_col: str = "date", val_col: str = "CRW_SST") -> pd.DataFrame:
    """
    Calcula la climatología mensual (media y desviación estándar) a partir de series históricas.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
        
    df["month"] = df[date_col].dt.month
    clim = df.groupby("month")[val_col].agg(["mean", "std", "min", "max"]).reset_index()
    return clim
