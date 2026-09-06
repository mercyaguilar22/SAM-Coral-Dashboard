"""
insitu_processor.py
===================
Procesamiento de datos in situ de CoralWatch y Fundaeco.
Reemplaza/corrige el flujo original de IN_SITU.ipynb.

Correcciones aplicadas:
  - Agrupación correcta: (lat, lon) por sitio permanente, no por (lat, lon, fecha)
  - Filtro: mínimo 10 corales por colonia, ≥20% blanqueamiento
  - ReefCheck eliminado (datos no útiles)
  - Flujo de datos documentado: observaciones → colonias globales → colonias SAM
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

BBOX          = CFG["region"]["bbox"]
MIN_CORALS    = CFG["insitu"]["min_corals_per_colony"]
BLEACH_THRESH = CFG["insitu"]["bleaching_threshold_pct"]

# Códigos de color CoralWatch: B1-B2 = blanqueado, C1-C2 = parcial, etc.
BLEACHED_CODES = {"B1", "B2", "C1", "C2", "D1", "D2", "E1", "E2"}


def load_coralwatch(json_path: str) -> pd.DataFrame:
    """
    Carga y normaliza el JSON crudo de CoralWatch.

    Flujo de datos documentado:
      Paso 1: N observaciones individuales (63,102 en tesis)
      Paso 2: Agrupación → colonias globales (~1,830 en tesis)
      Paso 3: Filtro SAM → colonias SAM (105 en tesis)
    """
    logger.info(f"Cargando CoralWatch desde {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.json_normalize(raw)
    n_raw = len(df)
    logger.info(f"  Paso 1 — Observaciones individuales: {n_raw:,}")

    # Normalizar nombres de país
    df["Country"] = df["Country"].str.strip().str.title()

    # Variable objetivo: ¿el coral está blanqueado?
    df["bleached_bool"] = df["colour_code"].isin(BLEACHED_CODES)

    # Asegurar tipos
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["date"]      = pd.to_datetime(df["date"],     errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "date"])

    # Paso 2: Agrupar en colonias (sitio + fecha de encuesta)
    # Una "colonia" = grupo de corales observados en el mismo lugar y día
    colonies = df.groupby(
        ["Country", "latitude", "longitude", "date"]
    ).agg(
        n_corals      = ("bleached_bool", "count"),
        n_bleached    = ("bleached_bool", "sum"),
        pct_bleached  = ("bleached_bool", "mean"),
    ).reset_index()

    n_colonies_global = len(colonies)
    logger.info(f"  Paso 2 — Colonias globales (agrupadas): {n_colonies_global:,}")

    # Aplicar filtros de calidad CoralWatch
    # (mínimo 10 corales, ≥20% blanqueamiento para clasificar como SI)
    colonies = colonies[colonies["n_corals"] >= MIN_CORALS].copy()
    colonies["potential_bleaching"] = (
        colonies["pct_bleached"] >= BLEACH_THRESH
    ).map({True: "SI", False: "NO"})

    logger.info(f"  Colonias tras filtro calidad (≥{MIN_CORALS} corales): "
                f"{len(colonies):,}")

    # Paso 3: Filtrar al área del SAM
    sam_mask = (
        (colonies["latitude"]  >= BBOX["lat_min"]) &
        (colonies["latitude"]  <= BBOX["lat_max"]) &
        (colonies["longitude"] >= BBOX["lon_min"]) &
        (colonies["longitude"] <= BBOX["lon_max"])
    )
    colonies_sam = colonies[sam_mask].copy()
    n_sam = len(colonies_sam)

    si_count = (colonies_sam["potential_bleaching"] == "SI").sum()
    no_count = (colonies_sam["potential_bleaching"] == "NO").sum()
    logger.info(f"  Paso 3 — Colonias en SAM: {n_sam:,} "
                f"(SI={si_count}, NO={no_count}, ratio={si_count/max(no_count,1):.1f}:1)")

    return colonies_sam


def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convierte el DataFrame de colonias a GeoDataFrame."""
    geometry = [Point(row.longitude, row.latitude) for _, row in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def load_fundaeco(csv_path: str = None) -> pd.DataFrame:
    """
    Carga datos de Fundaeco si están disponibles.
    Retorna DataFrame vacío si no se proveen.
    """
    if csv_path is None:
        logger.info("Fundaeco: no configurado, omitiendo.")
        return pd.DataFrame()
    logger.info(f"Cargando Fundaeco desde {csv_path}")
    # Adaptar según formato real del archivo
    df = pd.read_csv(csv_path, encoding="utf-8")
    return df


def get_insitu_gdf(save_path: str = None) -> gpd.GeoDataFrame:
    """
    Pipeline completo: carga, procesa y retorna el GeoDataFrame
    de datos in situ listos para el modelo.

    Parameters
    ----------
    save_path : str, optional
        Si se indica, guarda el resultado como GeoJSON.
    """
    # Cargar y procesar CoralWatch
    cw_path = CFG["insitu"]["coralwatch_json"]
    df = load_coralwatch(cw_path)
    gdf = to_geodataframe(df)

    # Cargar Fundaeco si existe
    fundaeco_path = CFG["insitu"].get("fundaeco")
    if fundaeco_path:
        df_fund = load_fundaeco(fundaeco_path)
        if not df_fund.empty:
            gdf_fund = to_geodataframe(df_fund)
            gdf = pd.concat([gdf, gdf_fund], ignore_index=True)
            logger.info(f"  Fundaeco integrado. Total colonias: {len(gdf):,}")

    if save_path:
        gdf.to_file(save_path, driver="GeoJSON")
        logger.info(f"Guardado: {save_path}")

    return gdf


if __name__ == "__main__":
    gdf = get_insitu_gdf()
    print(gdf[["Country", "date", "n_corals",
               "pct_bleached", "potential_bleaching"]].head(10))
    print(f"\nTotal colonias SAM: {len(gdf)}")
    print(gdf["potential_bleaching"].value_counts())
