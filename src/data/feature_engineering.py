"""
feature_engineering.py
=======================
Extrae features dinámicas de NOAA CRW para cada punto in situ.

CORRECCIÓN CRÍTICA vs versión original:
  - Original: usaba rasters promedio 2018-2024 (mean_dhw_2018_2024.tif)
    → Cada punto recibía el mismo valor sin importar el año/mes
  - Corregido: extrae el valor EXACTO del día de cada observación in situ
    → El modelo aprende calibración real, no patrones espaciales estáticos
"""

import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from src.data.noaa_api_client import fetch_crw_daily, get_point_value

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

FEATURES = CFG["model"]["features"]


def enrich_with_noaa(gdf: gpd.GeoDataFrame,
                     cache_dir: Path = None) -> gpd.GeoDataFrame:
    """
    Para cada punto in situ, descarga (o usa caché) el valor NOAA CRW
    del día EXACTO de la observación y lo agrega como feature.

    Parameters
    ----------
    gdf : GeoDataFrame
        Colonias in situ con columna 'date', 'latitude', 'longitude'.
    cache_dir : Path, optional
        Directorio para cachear NetCDF por fecha y evitar re-descargas.

    Returns
    -------
    GeoDataFrame enriquecido con columnas CRW_SST, CRW_DHW, etc.
    """
    noaa_cols = {v: [] for v in
                 ["CRW_SST", "CRW_SSTANOMALY", "CRW_HOTSPOT",
                  "CRW_DHW", "CRW_BAA"]}

    # Agrupar por fecha para descargar una vez por día (no por punto)
    unique_dates = gdf["date"].dt.normalize().unique()
    logger.info(f"Enriqueciendo {len(gdf)} colonias en {len(unique_dates)} fechas únicas")

    date_cache = {}  # {date_str: xr.Dataset}

    for obs_date in sorted(unique_dates):
        date_str = pd.Timestamp(obs_date).strftime("%Y-%m-%d")

        if date_str not in date_cache:
            try:
                if cache_dir:
                    cache_file = cache_dir / f"crw_{date_str.replace('-','')}.nc"
                    if cache_file.exists():
                        import xarray as xr
                        date_cache[date_str] = xr.open_dataset(cache_file)
                        logger.info(f"  Caché: {date_str}")
                    else:
                        ds = fetch_crw_daily(pd.Timestamp(obs_date).to_pydatetime())
                        date_cache[date_str] = ds
                        if cache_dir:
                            cache_dir.mkdir(parents=True, exist_ok=True)
                            ds.to_netcdf(cache_file)
                else:
                    ds = fetch_crw_daily(pd.Timestamp(obs_date).to_pydatetime())
                    date_cache[date_str] = ds
            except Exception as ex:
                logger.warning(f"  No se pudo descargar {date_str}: {ex}")
                date_cache[date_str] = None

    # Extraer valor por punto
    for _, row in gdf.iterrows():
        date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        ds = date_cache.get(date_str)

        for var in noaa_cols:
            if ds is not None:
                val = get_point_value(ds, row["latitude"], row["longitude"], var)
            else:
                val = np.nan
            noaa_cols[var].append(val)

    # Agregar columnas al GeoDataFrame
    for var, vals in noaa_cols.items():
        gdf = gdf.copy()
        gdf[var] = vals

    # Agregar features temporales
    gdf["year"]  = gdf["date"].dt.year
    gdf["month"] = gdf["date"].dt.month

    # Reporte de cobertura
    dhw_valid = gdf["CRW_DHW"].notna().sum()
    logger.info(f"Cobertura NOAA: {dhw_valid}/{len(gdf)} puntos ({100*dhw_valid/len(gdf):.1f}%)")

    return gdf


def prepare_model_features(gdf: gpd.GeoDataFrame) -> tuple:
    """
    Prepara X (features) e y (target) para el modelo RF.

    Returns
    -------
    X : pd.DataFrame, y : pd.Series
    """
    required_cols = FEATURES + ["potential_bleaching"]
    missing = [c for c in required_cols if c not in gdf.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    # Filtrar filas con NaN en features críticas
    gdf_clean = gdf.dropna(subset=FEATURES).copy()
    n_dropped = len(gdf) - len(gdf_clean)
    if n_dropped > 0:
        logger.warning(f"  Eliminadas {n_dropped} filas con NaN en features")

    X = gdf_clean[FEATURES].copy()
    y = (gdf_clean["potential_bleaching"] == "SI").astype(int)

    logger.info(f"Dataset final: {len(X)} muestras | "
                f"SI={y.sum()} ({100*y.mean():.1f}%) | "
                f"NO={len(y)-y.sum()} ({100*(1-y.mean()):.1f}%)")
    return X, y


if __name__ == "__main__":
    from src.data.insitu_processor import get_insitu_gdf
    gdf = get_insitu_gdf()
    gdf_enriched = enrich_with_noaa(gdf)
    X, y = prepare_model_features(gdf_enriched)
    print(X.describe())
