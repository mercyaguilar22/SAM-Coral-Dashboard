"""
charts.py
=========
Generación de gráficos interactivos con Plotly para el dashboard de Streamlit.

Visualizaciones:
  - Serie temporal de SST y anomalía (SSTA).
  - Serie temporal de DHW con líneas de umbral crítico (2.97 vs 4.0 °C·sem).
  - Comparativa de áreas en alerta (NOAA vs Regional SAM).
  - Distribución de probabilidades predichas por Random Forest.
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_dhw_time_series(df_ts: pd.DataFrame, 
                         thresh_reg: float = 2.97, 
                         thresh_noaa: float = 4.0) -> go.Figure:
    """
    Gráfico interactivo de evolución temporal de Degree Heating Weeks (DHW)
    mostrando los umbrales de alerta comparativos.
    """
    fig = go.Figure()

    # Serie principal de DHW
    fig.add_trace(go.Scatter(
        x=df_ts["date"],
        y=df_ts["CRW_DHW"],
        mode="lines",
        name="DHW Medio (°C·semanas)",
        line=dict(color="#d95f02", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(217, 95, 2, 0.15)",
    ))

    # Umbral regional optimizado (2.97)
    fig.add_hline(
        y=thresh_reg,
        line_dash="dash",
        line_color="#fdae61",
        line_width=2,
        annotation_text=f"Umbral Regional SAM ({thresh_reg} °C·sem)",
        annotation_position="top left",
    )

    # Umbral global NOAA (4.0)
    fig.add_hline(
        y=thresh_noaa,
        line_dash="dot",
        line_color="#e41a1c",
        line_width=2,
        annotation_text=f"Alerta Nivel 1 NOAA ({thresh_noaa} °C·sem)",
        annotation_position="top left",
    )

    # Nivel Severo 2 (8.0)
    fig.add_hline(
        y=8.0,
        line_dash="dashdot",
        line_color="#7f0000",
        line_width=1.5,
        annotation_text="Nivel 2 Mortalidad (8.0 °C·sem)",
        annotation_position="bottom left",
    )

    fig.update_layout(
        title="Evolución del Estrés Térmico Acumulado (DHW) en el SAM",
        xaxis_title="Fecha",
        yaxis_title="Degree Heating Weeks (°C·semanas)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    return fig


def plot_sst_anomaly_series(df_ts: pd.DataFrame) -> go.Figure:
    """
    Gráfico de Temperatura Superficial del Mar (SST) y Anomalías Térmicas (SSTA).
    """
    fig = go.Figure()

    if "CRW_SST" in df_ts.columns:
        fig.add_trace(go.Scatter(
            x=df_ts["date"],
            y=df_ts["CRW_SST"],
            mode="lines",
            name="SST Media (°C)",
            line=dict(color="#2b83ba", width=2),
        ))

    if "CRW_SSTANOMALY" in df_ts.columns:
        fig.add_trace(go.Scatter(
            x=df_ts["date"],
            y=df_ts["CRW_SSTANOMALY"],
            mode="lines",
            name="Anomalía SST (°C)",
            line=dict(color="#fdae61", width=1.5, dash="dot"),
            yaxis="y2",
        ))

    fig.update_layout(
        title="Temperatura Superficial del Mar y Anomalías Térmicas",
        xaxis_title="Fecha",
        yaxis=dict(title="SST (°C)"),
        yaxis2=dict(
            title="Anomalía (°C)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
    )
    return fig


def plot_country_alert_comparison(df_country: pd.DataFrame) -> go.Figure:
    """
    Gráfico de barras comparando el porcentaje de arrecifes bajo alerta
    entre el Umbral Regional Optimizado (2.97) y el Umbral Global NOAA (4.0).
    """
    if df_country.empty:
        return go.Figure()

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_country["País"],
        y=df_country["% Área en Alerta Regional"],
        name="Alerta Regional SAM (2.97)",
        marker_color="#fdae61",
        text=df_country["% Área en Alerta Regional"].apply(lambda v: f"{v}%"),
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        x=df_country["País"],
        y=df_country["% Área en Alerta NOAA"],
        name="Alerta NOAA Global (4.0)",
        marker_color="#e41a1c",
        text=df_country["% Área en Alerta NOAA"].apply(lambda v: f"{v}%"),
        textposition="auto",
    ))

    fig.update_layout(
        title="Sensibilidad Comparativa de Alertas por País",
        xaxis_title="País / Región",
        yaxis_title="% Arrecifes bajo Alerta",
        barmode="group",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_rf_probability_histogram(df_pts: pd.DataFrame) -> go.Figure:
    """Histograma de probabilidades estimadas por el modelo Random Forest."""
    if "rf_bleaching_proba" not in df_pts.columns or df_pts.empty:
        return go.Figure()

    fig = px.histogram(
        df_pts,
        x="rf_bleaching_proba",
        nbins=25,
        title="Distribución de Probabilidad de Blanqueamiento (Random Forest)",
        labels={"rf_bleaching_proba": "Probabilidad Estimada"},
        color_discrete_sequence=["#2b83ba"],
        template="plotly_white",
    )
    fig.update_layout(
        xaxis=dict(tickformat=".0%"),
        height=300,
        margin=dict(l=40, r=40, t=40, b=30),
    )
    return fig
