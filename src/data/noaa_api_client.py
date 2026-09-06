"""
noaa_api_client.py
==================
Cliente para descargar productos NOAA Coral Reef Watch (CRW) v3.1
vía ERDDAP API — reemplaza las descargas manuales de 12,000+ NetCDF.

Productos descargados:
  - CRW_SST      : Sea Surface Temperature (CoralTemp)
  - CRW_SSTANOMALY: SST Anomaly
  - CRW_HOTSPOT  : Thermal HotSpot
  - CRW_DHW      : Degree Heating Weeks
  - CRW_BAA      : Bleaching Alert Area

Uso:
    from src.data.noaa_api_client import fetch_crw_daily, fetch_crw_range
    ds = fetch_crw_daily(datetime(2024, 8, 1))
    ds_range = fetch_crw_range(datetime(2024,1,1), datetime(2024,8,31))
"""

import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from erddapy import ERDDAP

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cargar configuración
CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

ERDDAP_SERVER = CFG["noaa"]["erddap_server"]
DATASET_ID    = CFG["noaa"]["dataset_id"]
VARIABLES     = CFG["noaa"]["variables"]
BBOX          = CFG["region"]["bbox"]


def _build_erddap_connection() -> ERDDAP:
    """Crea y retorna una conexión ERDDAP configurada."""
    e = ERDDAP(server=ERDDAP_SERVER, protocol="griddap")
    e.dataset_id = DATASET_ID
    return e


def fetch_crw_daily(date: datetime) -> xr.Dataset:
    """
    Descarga todos los productos CRW para una fecha específica,
    recortados al bounding box del SAM.

    Parameters
    ----------
    date : datetime
        Fecha a descargar.

    Returns
    -------
    xr.Dataset
        Dataset con variables: CRW_SST, CRW_SSTANOMALY,
        CRW_HOTSPOT, CRW_DHW, CRW_BAA
    """
    logger.info(f"Descargando datos NOAA CRW para {date.strftime('%Y-%m-%d')}")

    e = _build_erddap_connection()
    e.constraints = {
        "time>=": date.strftime("%Y-%m-%dT00:00:00Z"),
        "time<=": date.strftime("%Y-%m-%dT23:59:59Z"),
        "latitude>=":  BBOX["lat_min"],
        "latitude<=":  BBOX["lat_max"],
        "longitude>=": BBOX["lon_min"],
        "longitude<=": BBOX["lon_max"],
    }
    e.variables = VARIABLES

    try:
        ds = e.to_xarray()
        logger.info(f"  ✓ Descargado: {dict(ds.dims)}")
        return ds
    except Exception as ex:
        logger.error(f"  ✗ Error descargando {date.date()}: {ex}")
        raise


def fetch_crw_range(start: datetime, end: datetime,
                    save_dir: Path = None) -> xr.Dataset:
    """
    Descarga un rango de fechas, con opción de guardar localmente.

    Parameters
    ----------
    start, end : datetime
        Rango de fechas (inclusive).
    save_dir : Path, optional
        Si se indica, guarda cada día como NetCDF en ese directorio.

    Returns
    -------
    xr.Dataset
        Dataset combinado con todas las fechas del rango.
    """
    datasets = []
    current = start

    while current <= end:
        try:
            ds = fetch_crw_daily(current)
            datasets.append(ds)

            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                fname = save_dir / f"crw_{current.strftime('%Y%m%d')}.nc"
                ds.to_netcdf(fname)
                logger.info(f"  Guardado: {fname.name}")

        except Exception as ex:
            logger.warning(f"  Saltando {current.date()}: {ex}")

        current += timedelta(days=1)

    if not datasets:
        raise ValueError("No se pudo descargar ningún dato en el rango.")

    combined = xr.concat(datasets, dim="time")
    logger.info(f"Rango completado: {start.date()} → {end.date()} "
                f"({len(datasets)} días)")
    return combined


def get_point_value(ds: xr.Dataset, lat: float, lon: float,
                    variable: str) -> float:
    """
    Extrae el valor de una variable en un punto lat/lon exacto.
    Usado para asociar el valor del día de observación in situ.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset CRW del día de la observación.
    lat, lon : float
        Coordenadas del punto in situ.
    variable : str
        Nombre de la variable (ej. "CRW_DHW").

    Returns
    -------
    float
        Valor interpolado en el punto.
    """
    try:
        val = ds[variable].sel(
            latitude=lat,
            longitude=lon,
            method="nearest"
        ).values
        return float(np.squeeze(val))
    except Exception as ex:
        logger.warning(f"No se pudo extraer {variable} en ({lat}, {lon}): {ex}")
        return np.nan


if __name__ == "__main__":
    # Test rápido: descargar datos de ayer
    yesterday = datetime.utcnow() - timedelta(days=1)
    ds = fetch_crw_daily(yesterday)
    print(ds)
    print(f"\nDHW máximo en el SAM: {float(ds['CRW_DHW'].max().values):.2f} °C·semanas")
