"""
publish.py
==========
Módulo para serializar y publicar los productos diarios del SAM Coral Monitor
hacia el almacenamiento del dashboard (GeoJSON, GeoParquet y resúmenes JSON).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import geopandas as gpd
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DAILY_DIR = Path(__file__).parents[2] / CFG["dashboard"]["data_daily_path"]


def publish_daily_results(gdf_predictions: gpd.GeoDataFrame,
                          summary_metrics: dict,
                          date: datetime = None) -> Path:
    """
    Exporta el producto diario a data/daily/ con nombre de fecha
    y actualiza 'latest.geojson' y 'latest_summary.json' para el dashboard.
    """
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    
    if date is None:
        date = datetime.utcnow()
        
    date_str = date.strftime("%Y-%m-%d")
    out_geojson = DAILY_DIR / f"{date_str}.geojson"
    latest_geojson = DAILY_DIR / "latest.geojson"
    latest_summary = DAILY_DIR / "latest_summary.json"

    # Exportar GeoJSON del día
    gdf_predictions.to_file(out_geojson, driver="GeoJSON")
    gdf_predictions.to_file(latest_geojson, driver="GeoJSON")
    logger.info(f"Guardado producto diario: {out_geojson}")

    # Exportar resumen
    summary_metrics["last_updated_date"] = date_str
    summary_metrics["last_updated_timestamp"] = datetime.utcnow().isoformat()
    
    with open(latest_summary, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Guardado resumen métrico: {latest_summary}")

    return out_geojson
