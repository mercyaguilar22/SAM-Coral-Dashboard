"""
map_layers.py
=============
Módulo para generación de capas cartográficas interactivas en Streamlit
usando Folium y Leafmap.

Capas soportadas:
  - Mapa térmico de SST (°C)
  - Grilla de DHW (°C·semanas)
  - Alertas regionales SAM (umbral 2.97) vs Alertas NOAA (umbral 4.0)
  - Probabilidad de blanqueamiento calibrada por Random Forest
  - Polígono del área de estudio SAM
"""

from pathlib import Path
from typing import Optional
import folium
from folium.plugins import Fullscreen, MeasureControl, HeatMap
import geopandas as gpd
import pandas as pd
from branca.colormap import LinearColormap

from src.analysis.alerts import get_alert_color_map


def create_base_map(center_lat: float = 18.5, center_lon: float = -87.5, zoom: int = 8) -> folium.Map:
    """Crea un mapa base con imagen satelital Esri como mapa único."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satélite (Esri World Imagery)",
        control_scale=True,
    )

    # 🛰️ Capa satelital de Temperatura Superficial del Mar (NOAA Coral Reef Watch)
    folium.WmsTileLayer(
        url="https://coralreefwatch.noaa.gov/geoserver/wms",
        layers="crw:coraltemp",
        name="🛰️ Temperatura Superficial del Mar (SST) - NOAA CRW",
        fmt="image/png",
        transparent=True,
        overlay=True,
        control=True,
        show=False,
        opacity=0.65,
        attr="NOAA Coral Reef Watch",
    ).add_to(m)

    # 🔥 Capa satelital de Grados de Calentamiento Semanal (NOAA CRW)
    folium.WmsTileLayer(
        url="https://coralreefwatch.noaa.gov/geoserver/wms",
        layers="crw:dhw_5km",
        name="🔥 Grados de Calentamiento Semanal (DHW) - NOAA CRW",
        fmt="image/png",
        transparent=True,
        overlay=True,
        control=True,
        show=False,
        opacity=0.65,
        attr="NOAA Coral Reef Watch",
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
            name="Límite del SAM",
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
    con código de colores representativo.
    """
    if gdf.empty:
        return m

    color_map = get_alert_color_map()
    
    # 🌡️ Superficie Continua de Temperatura Superficial del Mar interpolada
    heat_data = [
        [row.geometry.y, row.geometry.x, float(row["CRW_SST"])]
        for _, row in gdf.iterrows()
        if "CRW_SST" in row and pd.notna(row["CRW_SST"])
    ]
    if heat_data:
        sst_vals = [d[2] for d in heat_data]
        min_s, max_s = min(sst_vals), max(sst_vals)
        norm_heat = [[lat, lon, (val - min_s) / max(max_s - min_s, 0.1)] for lat, lon, val in heat_data]
        
        heat_group = folium.FeatureGroup(name="🌡️ Superficie Continua de Temperatura (°C)", show=True)
        HeatMap(
            norm_heat,
            min_opacity=0.35,
            radius=24,
            blur=16,
            gradient={0.2: '#2b83ba', 0.4: '#abdda4', 0.6: '#ffffbf', 0.8: '#fdae61', 1.0: '#d7191c'}
        ).add_to(heat_group)
        heat_group.add_to(m)

    # Submuestreo si hay demasiados puntos para mantener alta velocidad en navegador
    points_to_plot = gdf.sample(n=min(len(gdf), sample_max), random_state=42) if len(gdf) > sample_max else gdf

    feature_group = folium.FeatureGroup(name="Alertas de Blanqueamiento de Coral", show=True)

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
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m
