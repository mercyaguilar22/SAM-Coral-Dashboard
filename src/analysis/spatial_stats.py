"""
spatial_stats.py
================
Módulo de agregación espacial y estadísticas por país / subregión
para el Sistema Arrecifal Mesoamericano (México, Belice, Guatemala, Honduras).
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

COUNTRIES = CFG["region"]["countries"]
THRESH_REGIONAL = CFG["thresholds"]["regional_optimized"]
THRESH_NOAA = CFG["thresholds"]["noaa_global"]


def assign_country_by_lat_bounds(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Aproximación geográfica por rangos latitudinales del SAM si no se cuenta con shapefile de límites:
      - México (Quintana Roo / Yucatán): lat >= 18.2
      - Belice: 15.8 <= lat < 18.2 y lon >= -88.6
      - Guatemala: 15.6 <= lat < 16.0 y lon < -88.6
      - Honduras (Islas de la Bahía y costa norte): lat < 16.6 y lon > -87.2
    """
    gdf = gdf.copy()
    if "Country" in gdf.columns and gdf["Country"].notna().any():
        return gdf

    conditions = [
        (gdf["lat"] >= 18.2),
        ((gdf["lat"] >= 15.8) & (gdf["lat"] < 18.2) & (gdf["lon"] >= -88.5)),
        ((gdf["lat"] >= 15.6) & (gdf["lat"] < 16.0) & (gdf["lon"] < -88.5)),
        ((gdf["lat"] < 16.8) & (gdf["lon"] > -87.2)),
    ]
    choices = ["México", "Belice", "Guatemala", "Honduras"]
    gdf["Country"] = np.select(conditions, choices, default="Mar Abierto / SAM")
    return gdf


def aggregate_metrics_by_country(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas de estrés térmico y alertas agregadas por país.
    """
    if gdf.empty:
        return pd.DataFrame()
        
    gdf = assign_country_by_lat_bounds(gdf)
    
    records = []
    for country, group in gdf.groupby("Country"):
        total = len(group)
        alerts_reg = (group["CRW_DHW"] >= THRESH_REGIONAL).sum()
        alerts_noaa = (group["CRW_DHW"] >= THRESH_NOAA).sum()
        
        records.append({
            "País": country,
            "Puntos Arrecifales": total,
            "SST Media (°C)": round(group["CRW_SST"].mean(), 2) if "CRW_SST" in group.columns else np.nan,
            "SST Máx (°C)": round(group["CRW_SST"].max(), 2) if "CRW_SST" in group.columns else np.nan,
            "DHW Medio (°C·sem)": round(group["CRW_DHW"].mean(), 2),
            "DHW Máx (°C·sem)": round(group["CRW_DHW"].max(), 2),
            "Puntos en Alerta Regional (2.97)": int(alerts_reg),
            "% Área en Alerta Regional": round((alerts_reg / total) * 100.0, 1),
            "Puntos en Alerta NOAA (4.0)": int(alerts_noaa),
            "% Área en Alerta NOAA": round((alerts_noaa / total) * 100.0, 1),
        })
        
    df_agg = pd.DataFrame(records)
    return df_agg
