"""
map_layers.py
=============
Módulo para generación de capas cartográficas interactivas en Streamlit
usando Folium con capas continuas de SST y DHW calibradas para el SAM.
"""

import base64
from pathlib import Path
from typing import Optional
import folium
from folium.plugins import Fullscreen, MeasureControl
import geopandas as gpd
import pandas as pd

from src.analysis.alerts import get_alert_color_map

STATIC_DIR = Path(__file__).parents[2] / "data" / "static"


def get_image_data_uri(png_path: Path) -> Optional[str]:
    """Lee una imagen PNG y la convierte a Data URI base64."""
    if png_path.exists():
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    return None


def create_base_map(center_lat: float = 18.5, 
                    center_lon: float = -87.5, 
                    zoom: int = 8,
                    year: Optional[int] = None) -> folium.Map:
    """
    Crea el mapa base con imagen satelital Esri World Imagery como mapa base único
    y añade las capas continuas satelitales de SST y DHW recortadas al SAM correspondientes
    al año seleccionado (2018-2024) o el consolidado multianual.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
    )

    # 1. Mapa Base Satelital Esri
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satélite (Esri World Imagery)",
        overlay=False,
        control=True,
    ).add_to(m)

    # Determinar si existen rasters anuales específicos
    rasters_dir = STATIC_DIR / "rasters"
    sst_png = None
    dhw_png = None
    label_suffix = ""

    if year and rasters_dir.exists():
        annual_sst = rasters_dir / f"sst_{year}.png"
        annual_dhw = rasters_dir / f"dhw_{year}.png"
        if annual_sst.exists():
            sst_png = annual_sst
            label_suffix = f" ({year})"
        if annual_dhw.exists():
            dhw_png = annual_dhw

    if sst_png is None:
        sst_png = STATIC_DIR / "mean_sst_sam.png"
        label_suffix = " (Consolidado 2018-2024)"

    if dhw_png is None:
        dhw_png = STATIC_DIR / "mean_dhw_sam.png"

    # 2. Capa continua de SST recortada al SAM
    sst_uri = get_image_data_uri(sst_png)
    if sst_uri:
        folium.raster_layers.ImageOverlay(
            name=f"SST - NOAA/CRW{label_suffix}",
            image=sst_uri,
            bounds=[[15.70, -88.95], [22.00, -83.05]],
            opacity=0.80,
            overlay=True,
            control=True,
            show=True,
        ).add_to(m)

    # 3. Capa continua de DHW recortada al SAM
    dhw_uri = get_image_data_uri(dhw_png)
    if dhw_uri:
        folium.raster_layers.ImageOverlay(
            name=f"DHW - NOAA CRW{label_suffix}",
            image=dhw_uri,
            bounds=[[15.70, -88.95], [22.00, -83.05]],
            opacity=0.80,
            overlay=True,
            control=True,
            show=False,
        ).add_to(m)

    Fullscreen(position="topright").add_to(m)
    MeasureControl(position="bottomleft").add_to(m)
    return m


def add_sam_boundary(m: folium.Map, geojson_path: Path) -> folium.Map:
    """Añade el límite poligonal del SAM al mapa."""
    if geojson_path.exists():
        gdf_sam = gpd.read_file(geojson_path)
        folium.GeoJson(
            gdf_sam,
            name="Límite SAM",
            style_function=lambda x: {
                "fillColor": "none",
                "color": "#00ffff",
                "weight": 2.5,
                "dashArray": "5, 5",
            },
            tooltip="Área de Estudio: Sistema Arrecifal Mesoamericano",
        ).add_to(m)
    return m


def add_alert_points(m: folium.Map, gdf: gpd.GeoDataFrame, sample_max: int = 1500) -> folium.Map:
    """
    Añade los puntos de estrés térmico y alertas de blanqueamiento al mapa
    con código de colores representativo y control de capas colapsable en la esquina.
    """
    if gdf.empty:
        folium.LayerControl(position="topright", collapsed=True).add_to(m)
        return m

    color_map = get_alert_color_map()
    
    # Submuestreo si hay demasiados puntos para mantener alta velocidad en navegador
    points_to_plot = gdf.sample(n=min(len(gdf), sample_max), random_state=42) if len(gdf) > sample_max else gdf

    feature_group = folium.FeatureGroup(name="Alertas de Blanqueamiento", show=True)

    for _, row in points_to_plot.iterrows():
        cat = row.get("alert_category", "Sin Estrés")
        color = color_map.get(cat, "#2b83ba")
        dhw = row.get("CRW_DHW", 0.0)
        sst = row.get("CRW_SST", 0.0)
        rf_prob = row.get("rf_bleaching_proba", None)

        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 180px;">
            <b style="color: {color}; font-size: 13px;">{cat}</b><br>
            <hr style="margin: 4px 0;">
            <b>DHW:</b> {dhw:.2f} °C·sem<br>
            <b>SST:</b> {sst:.2f} °C<br>
            {"<b>Probabilidad RF:</b> " + f"{rf_prob*100:.1f}%<br>" if rf_prob is not None else ""}
            <small>Lat: {row.geometry.y:.3f}, Lon: {row.geometry.x:.3f}</small>
        </div>
        """

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4.5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{cat} (DHW: {dhw:.2f} °C·sem)",
        ).add_to(feature_group)

    feature_group.add_to(m)
    
    # Control de capas desplegable / colapsable tipo Google Earth Engine (collapsed=True)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    return m
