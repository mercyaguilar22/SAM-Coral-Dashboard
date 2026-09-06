"""
daily_pipeline.py
=================
Orquestador diario para el SAM Coral Bleaching Monitor.
Diseñado para ser invocado localmente o vía GitHub Actions en cron diario (06:00 UTC).

Flujo de ejecución:
  1. Descarga datos de ayer de NOAA CRW vía ERDDAP API (recortados al SAM).
  2. Ejecuta inferencia con el modelo Random Forest regionalmente calibrado.
  3. Aplica los umbrales de alerta (Global 4.0 vs Regional SAM 2.97).
  4. Genera estadísticas espaciales agregadas por país.
  5. Publica 'latest.geojson' y 'latest_summary.json' para el dashboard de Streamlit.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml

from src.data.noaa_api_client import fetch_crw_daily
from src.model.predict import predict_from_dataset
from src.analysis.alerts import categorize_alerts, generate_alert_summary
from src.analysis.spatial_stats import aggregate_metrics_by_country
from src.dashboard.publish import publish_daily_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def run_daily_update(target_date: datetime = None) -> dict:
    """
    Ejecuta el pipeline de actualización diario completo.
    """
    if target_date is None:
        # Por defecto ayer (datos CRW consolidados)
        target_date = datetime.utcnow() - timedelta(days=1)
        
    date_str = target_date.strftime("%Y-%m-%d")
    logger.info(f"=== INICIANDO PIPELINE DIARIO SAM PARA: {date_str} ===")

    # 1. Descarga de NOAA CRW
    try:
        ds_crw = fetch_crw_daily(target_date)
    except Exception as e:
        logger.error(f"Fallo en descarga ERDDAP: {e}")
        raise e

    # 2. Inferencia y predicción espacial
    gdf_preds = predict_from_dataset(ds_crw, date=target_date)
    if gdf_preds.empty:
        logger.error("No se generaron predicciones a partir del dataset.")
        return {}

    # 3. Categorización de alertas
    gdf_alerts = categorize_alerts(gdf_preds)

    # 4. Resúmenes estadísticos
    summary = generate_alert_summary(gdf_alerts)
    df_country = aggregate_metrics_by_country(gdf_alerts)
    summary["country_breakdown"] = df_country.to_dict(orient="records")

    # 5. Publicación
    publish_daily_results(gdf_alerts, summary, date=target_date)
    
    logger.info(f"=== PIPELINE FINALIZADO CON ÉXITO PARA: {date_str} ===")
    logger.info(f"Puntos procesados: {summary['total_points']}")
    logger.info(f"DHW Máximo en SAM: {summary['max_dhw']:.2f} °C·sem")
    logger.info(f"Puntos en Alerta Regional (2.97): {summary['alerts_regional_sam']}")
    logger.info(f"Puntos en Alerta NOAA Global (4.0): {summary['alerts_noaa_global']}")
    logger.info(f"Ganancia de Detección Temprana: +{summary['early_detection_gain_points']} sitios")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Diario SAM Coral Bleaching Monitor")
    parser.add_argument("--date", type=str, default=None, help="Fecha en formato YYYY-MM-DD")
    args = parser.parse_args()

    date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else None
    run_daily_update(date)
