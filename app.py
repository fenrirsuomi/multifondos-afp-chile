"""
Análisis Multifondos AFP Chile (A, B, C, D, E)
Migración desde notebook Jupyter a app web interactiva con Streamlit.
"""
import glob

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Multifondos AFP Chile",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

AFPS_ACTUALES = ['HABITAT', 'PROVIDA', 'CUPRUM', 'PLANVITAL', 'UNO', 'CAPITAL', 'MODELO']
CRISIS_SUBPRIME = ('2008-01-01', '2009-05-01')
CRISIS_COVID = ('2020-02-01', '2020-10-01')

FONDOS_INFO = {
    'A': "Más riesgoso — mayor exposición a renta variable internacional. Mayor potencial de retorno, mayor volatilidad.",
    'B': "Riesgoso — alta exposición a renta variable, algo menor que el Fondo A.",
    'C': "Intermedio (fondo por defecto para quienes no eligen) — balance entre renta variable y renta fija.",
    'D': "Conservador — mayor proporción de renta fija, menor volatilidad.",
    'E': "Más conservador — predominantemente renta fija. Pensado para quienes están cerca de jubilar.",
}


# ============================================================
# DETECCIÓN DE ARCHIVOS DE FONDOS DISPONIBLES
# ============================================================
def detectar_fondos():
    """Busca archivos *Fondo_X_Valor_Cuota_Final.csv en la carpeta actual."""
    fondos = {}
    for letra in ['A', 'B', 'C', 'D', 'E']:
        matches = glob.glob(f"*Fondo_{letra}_Valor_Cuota_Final.csv")
        if matches:
            fondos[letra] = matches[0]
    return fondos


FONDO_FILES = detectar_fondos()

if not FONDO_FILES:
    st.error(
        "No se encontró ningún archivo `Fondo_X_Valor_Cuota_Final.csv` en la carpeta "
        "de la app. Verifica que los CSV estén junto a `app.py`."
    )
    st.stop()


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.set_index('Fecha').sort_index()
    return df


@st.cache_data
def get_afps_existentes(df, afps):
    return [c for c in afps if c in df.columns]


@st.cache_data
def rentabilidad_anual_historica(df, cols):
    """Rentabilidad anual histórica completa, con fix manual AFP Capital (año de entrada)."""
    df_anual = df[cols].resample('YE').last()
    df_rent = (df_anual / df_anual.shift(1) - 1) * 100

    if 'CAPITAL' in cols:
        primer_dato = df['CAPITAL'].dropna()
        if not primer_dato.empty:
            fecha_inicio = primer_dato.index[0]
            if fecha_inicio.year == 2009:
                precio_ini = primer_dato.iloc[0]
                idx_2009 = df_anual.index[df_anual.index.year == 2009]
                if not idx_2009.empty:
                    precio_fin = df_anual.loc[idx_2009[0], 'CAPITAL']
                    df_rent.loc[idx_2009[0], 'CAPITAL'] = ((precio_fin / precio_ini) - 1) * 100
    return df_anual, df_rent


@st.cache_data
def rentabilidad_periodo(df, cols, start_date):
    """Rentabilidad anual para un sub-periodo (ej. 2020-2025), con fix del primer año parcial."""
    df_periodo = df.loc[start_date:, cols].copy()
    df_anual = df_periodo.resample('YE').last()
    df_rent = df_anual.pct_change(fill_method=None) * 100

    start_year = pd.Timestamp(start_date).year
    for afp in df_rent.columns:
        first_valid = df_periodo[afp].first_valid_index()
        if first_valid is not None and first_valid.year == start_year:
            p_ini = df_periodo.loc[first_valid, afp]
            idx_year = df_anual.index[df_anual.index.year == start_year]
            if len(idx_year):
                p_fin = df_anual.loc[idx_year[0], afp]
                if p_ini and p_ini != 0:
                    df_rent.loc[idx_year[0], afp] = ((p_fin / p_ini) - 1) * 100
    return df_periodo, df_anual, df_rent


# ============================================================
# UF: VALOR CUOTA REAL (AJUSTADO POR INFLACIÓN) + CAGR
# ============================================================
@st.cache_data
def load_uf(path='UF.xlsx'):
    """UF de cierre de cada año (1980-2025). Resolución anual: es lo máximo
    disponible porque no se cuenta con el valor UF diario."""
    df_uf = pd.read_excel(path, skiprows=2)
    df_uf.columns = ['Año', 'IPC_Anual', 'Valor_UF_Dic']
    df_uf['Año'] = df_uf['Año'].astype(int)
    return df_uf[['Año', 'Valor_UF_Dic']]


@st.cache_data
def construir_valor_cuota_real(df, cols, df_uf):
    """Valor cuota de fin de año, dividido por el valor de la UF de cierre del
    mismo año. El resultado queda "denominado en UF": una serie comparable en
    el tiempo sin el efecto de la inflación."""
    df_anual = df[cols].resample('YE').last().copy()
    df_anual['Año'] = df_anual.index.year
    df_cruce = df_anual.merge(df_uf, on='Año', how='left').set_index('Año')
    df_real = df_cruce[cols].div(df_cruce['Valor_UF_Dic'], axis=0)
    df_real.index = pd.to_datetime(df_real.index.astype(str) + '-12-31')
    return df_real


@st.cache_data
def ranking_cagr(df_niveles, cols, start_date=None):
    """Ranking por CAGR (tasa de crecimiento anual compuesta), no por promedio
    aritmético de retornos anuales -- el promedio simple sobreestima el
    crecimiento real cuando hay volatilidad (más aún en fondos como el A).
    Usa el primer/último dato VÁLIDO de cada AFP por separado, para que cada
    una se compare dentro de su propio período real de existencia."""
    base = df_niveles.loc[start_date:] if start_date else df_niveles
    filas = []
    for c in cols:
        if c not in base.columns:
            continue
        s = base[c].dropna()
        if len(s) < 2:
            continue
        dias = (s.index.max() - s.index.min()).days
        anios = dias / 365.25
        if anios <= 0:
            continue
        rend_total = (s.iloc[-1] / s.iloc[0]) - 1
        cagr = ((1 + rend_total) ** (1 / anios) - 1) * 100
        filas.append({
            'AFP': c,
            'Rentabilidad Promedio': round(cagr, 2),
            'Info Periodo': f"{s.index.min().year}-{s.index.max().year} ({anios:.0f} años)",
        })
    if not filas:
        return pd.DataFrame(columns=['AFP', 'Rentabilidad Promedio', 'Info Periodo'])
    return pd.DataFrame(filas).sort_values('Rentabilidad Promedio', ascending=False)


@st.cache_data
def retornos_anuales_reales(df_real, cols):
    """Retorno año a año de la serie YA denominada en UF (para el heatmap real).
    El primer año de cada AFP queda en NaN: solo tenemos el cierre de año en UF,
    no un precio de entrada intra-año en UF (no hay UF diaria disponible)."""
    return df_real[cols].pct_change(fill_method=None) * 100


def formato_pesos_clp(valor):
    return f"$ {int(valor):,}".replace(',', '.')


def render_chart(fig):
    """Renderiza un gráfico Plotly con estilo fijo (independiente del tema
    claro/oscuro/sistema del navegador del visitante)."""
    fig.update_layout(
        legend=dict(
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='#d0d0d0',
            borderwidth=1,
            font=dict(color='#1a1a1a'),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ============================================================
# SIDEBAR: SELECTOR DE FONDO + NAVEGACIÓN
# ============================================================
st.sidebar.title("📊 Multifondos AFP Chile")

fondos_disponibles = sorted(FONDO_FILES.keys())
fondo_actual = st.sidebar.selectbox(
    "Selecciona el Fondo:",
    fondos_disponibles,
    format_func=lambda l: f"Fondo {l}",
)
st.sidebar.caption(FONDOS_INFO.get(fondo_actual, ""))
st.sidebar.divider()

# --- Carga de datos del fondo seleccionado ---
df = load_data(FONDO_FILES[fondo_actual])
afps_existentes = get_afps_existentes(df, AFPS_ACTUALES)
todas_las_afps = [c for c in df.columns]

df_anual_hist, df_rent_hist = rentabilidad_anual_historica(df, afps_existentes)
df_moderna, df_anual_moderna, df_rent_moderna = rentabilidad_periodo(df, afps_existentes, '2020-01-01')

# --- UF y valor cuota real (ajustado por inflación) ---
DF_UF = load_uf('UF.xlsx')
df_real = construir_valor_cuota_real(df, afps_existentes, DF_UF)

modo_valor = st.sidebar.radio(
    "Ver rentabilidad en:",
    ["Nominal (CAGR)", "Real (ajustado UF)"],
    help=(
        "Nominal: en pesos corrientes. Real: el valor cuota se divide por la UF "
        "de cierre de cada año, para descontar el efecto de la inflación. Ambos "
        "usan CAGR (tasa de crecimiento anual compuesta), no promedio simple."
    ),
)
es_real = modo_valor.startswith("Real")
etiqueta_modo = "Real (UF)" if es_real else "Nominal"

st.sidebar.divider()

secciones = [
    "🏠 Introducción",
    "📈 Evolución Histórica",
    "🏆 Ranking Histórico Completo",
    "🔍 Radiografía por AFP",
    "🥊 Competencia 2020–2025",
    "🌡️ Heatmap Anual",
    "📉 Drawdown",
    "📐 Escala Logarítmica",
    "〰️ Volatilidad",
    "💰 Interés Compuesto (Fijo)",
    "🎛️ Simulador Interactivo",
    "✅ Conclusiones",
]
pagina = st.sidebar.radio("Ir a sección:", secciones, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption(
    "Fuente: Superintendencia de Pensiones de Chile. "
    f"Datos de valor cuota diario, Fondo {fondo_actual}, "
    f"{df.index.min().year}–{df.index.max().year}."
)

# ============================================================
# 1. INTRODUCCIÓN
# ============================================================
if pagina == "🏠 Introducción":
    st.title(f"📊 Fondo {fondo_actual}: Radiografía del sistema de multifondos en Chile")
    st.markdown(
        f"""
El sistema de pensiones en Chile opera bajo un modelo de **capitalización individual**,
donde cada trabajador aporta obligatoriamente el 10% de su sueldo bruto a una cuenta
personal administrada por una AFP (Administradora de Fondos de Pensiones).

El sistema cuenta con un esquema de **multifondos (A, B, C, D y E)** diferenciados por
su nivel de exposición al riesgo. Estás viendo el **Fondo {fondo_actual}**:

> {FONDOS_INFO.get(fondo_actual, "")}

### Objetivo
Analizar de forma cuantitativa y objetiva el desempeño histórico de las AFP:
rentabilidad, capacidad de recuperación ante crisis (*drawdown*) y el efecto del
interés compuesto en el largo plazo.

### Fuente de datos
Valor cuota diario histórico de la Superintendencia de Pensiones de Chile.
        """
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("AFPs en el dataset", len(todas_las_afps))
    c2.metric("AFPs vigentes hoy", len(afps_existentes))
    c3.metric("Rango de datos", f"{df.index.min().year}–{df.index.max().year}")

    st.info(
        "Usa el selector de arriba en el menú lateral para cambiar de Fondo (A-E), "
        "y el menú de navegación para moverte entre secciones: evolución histórica, "
        "rankings, heatmap, drawdown, volatilidad y el simulador de interés compuesto."
    )

# ============================================================
# 2. EVOLUCIÓN HISTÓRICA (todas las AFP, incluidas las fusionadas)
# ============================================================
elif pagina == "📈 Evolución Histórica":
    st.title(f"📈 Evolución Histórica del Valor Cuota — Fondo {fondo_actual}")
    st.caption("Incluye todas las administradoras que han pasado por el sistema (activas y fusionadas).")

    seleccion = st.multiselect(
        "Selecciona AFPs a mostrar:", todas_las_afps, default=todas_las_afps
    )

    if seleccion:
        if es_real:
            cols_real = [c for c in seleccion if c in df_real.columns]
            if not cols_real:
                st.warning("Ninguna de las AFP seleccionadas tiene serie real (UF) disponible.")
                st.stop()
            df_plot = df_real[cols_real].reset_index().rename(columns={'index': 'Fecha'}).melt(
                id_vars='Fecha', var_name='AFP', value_name='Valor Cuota'
            ).dropna()
            titulo = f'Evolución Histórica Valor Cuota REAL (en UF) — Fondo {fondo_actual}'
            eje_y = "Valor Cuota (en UF)"
            fmt_y = ",.4f"
            st.caption(
                "📏 Resolución anual: el ajuste por UF solo puede calcularse con el cierre de "
                "cada año, así que en modo Real se pierde el detalle diario."
            )
        else:
            df_plot = df[seleccion].reset_index().melt(
                id_vars='Fecha', var_name='AFP', value_name='Valor Cuota'
            ).dropna()
            titulo = f'Evolución Histórica Valor Cuota — Fondo {fondo_actual}'
            eje_y = "Valor Cuota ($ CLP)"
            fmt_y = ",.0f"

        fig = px.line(
            df_plot, x='Fecha', y='Valor Cuota', color='AFP',
            title=titulo, template='plotly_white',
        )
        fig.update_layout(
            yaxis_title=eje_y, xaxis_title="Año",
            legend_title="AFP", hovermode="x unified", height=600,
        )
        fig.update_yaxes(tickformat=fmt_y)
        if es_real:
            fig.update_traces(mode='lines+markers')
        render_chart(fig)
    else:
        st.warning("Selecciona al menos una AFP.")

# ============================================================
# 3. RANKING HISTÓRICO COMPLETO (AFP vigentes)
# ============================================================
elif pagina == "🏆 Ranking Histórico Completo":
    st.title(f"🏆 Rentabilidad {etiqueta_modo} Histórica (CAGR) — Fondo {fondo_actual}")
    st.caption("Solo las 7 AFP vigentes actualmente, cada una con su periodo real de operación.")
    st.caption(
        "Se usa CAGR (tasa de crecimiento anual compuesta), no el promedio simple de "
        "retornos anuales — el promedio simple sobreestima el crecimiento real cuando hay "
        "volatilidad."
    )

    if es_real:
        ranking = ranking_cagr(df_real, afps_existentes, start_date='2003-01-01')
        titulo = f'Rentabilidad Real (ajustada UF) Histórica — Fondo {fondo_actual}'
    else:
        # 2003-01-01: el sistema de multifondos partió en agosto 2002, así que ese
        # año quedaría incompleto y distorsionaría el CAGR.
        ranking = ranking_cagr(df, afps_existentes, start_date='2003-01-01')
        titulo = f'Rentabilidad Nominal Histórica — Fondo {fondo_actual}'

    fig = px.bar(
        ranking, x='Rentabilidad Promedio', y='AFP', orientation='h',
        text=ranking['Rentabilidad Promedio'].map(lambda v: f"{v:.2f}%"),
        color='Rentabilidad Promedio', color_continuous_scale='viridis',
        hover_data={'Info Periodo': True, 'Rentabilidad Promedio': ':.2f'},
        title=titulo,
        template='plotly_white',
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Rentabilidad Anualizada Compuesta - CAGR (%)",
        coloraxis_showscale=False, height=500,
    )
    fig.update_traces(textposition='outside')
    render_chart(fig)
    st.dataframe(ranking.set_index('AFP'), use_container_width=True)

# ============================================================
# 4. RADIOGRAFÍA INDIVIDUAL (Small multiples)
# ============================================================
elif pagina == "🔍 Radiografía por AFP":
    st.title(f"🔍 Radiografía Individual: Evolución Comparada — Fondo {fondo_actual}")
    st.caption("Small multiples — resample semanal para aligerar el gráfico.")

    df_plotly = df[afps_existentes].resample('W').last().reset_index()
    df_melted = df_plotly.melt(id_vars='Fecha', var_name='AFP', value_name='Valor Cuota')

    tema = st.radio("Tema del gráfico:", ["Oscuro", "Claro"], horizontal=True)
    template = 'plotly_dark' if tema == "Oscuro" else 'plotly_white'

    fig = px.area(
        df_melted, x='Fecha', y='Valor Cuota', color='AFP', facet_col='AFP',
        facet_col_wrap=4,
        title=f'Radiografía Individual: Evolución de cada AFP (Fondo {fondo_actual} Nominal)',
        template=template,
    )
    fig.update_yaxes(matches=None)
    fig.update_layout(showlegend=False, height=650, title_font_size=20)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    render_chart(fig)

# ============================================================
# 5. COMPETENCIA 2020-2025 (mismas 7 AFP, mismo periodo)
# ============================================================
elif pagina == "🥊 Competencia 2020–2025":
    st.title(f"🥊 Competencia entre las 7 AFP actuales (2020–2025) — Fondo {fondo_actual}")
    st.markdown(
        "Comparación en igualdad de condiciones: mismo periodo para las 7 AFP que "
        "conviven en el sistema desde la creación de AFP UNO en 2019. "
        "Se usa CAGR, no promedio simple de retornos anuales."
    )

    if es_real:
        ranking = ranking_cagr(df_real, afps_existentes, start_date='2020-01-01')
        titulo = f'Rentabilidad Real (UF) Fondo {fondo_actual} — Era Moderna (2020–2025)'
    else:
        ranking = ranking_cagr(df, afps_existentes, start_date='2020-01-01')
        titulo = f'Rentabilidad Nominal Fondo {fondo_actual} — Era Moderna (2020–2025)'

    fig = px.bar(
        ranking, x='Rentabilidad Promedio', y='AFP', orientation='h',
        text=ranking['Rentabilidad Promedio'].map(lambda v: f"{v:.2f}%"),
        color='Rentabilidad Promedio', color_continuous_scale='plasma',
        title=titulo,
        template='plotly_white',
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Rentabilidad Anualizada Compuesta - CAGR (%)",
        coloraxis_showscale=False, height=450,
    )
    fig.update_traces(textposition='outside')
    render_chart(fig)
    st.dataframe(ranking.set_index('AFP'), use_container_width=True)

    csv = ranking.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Descargar ranking (CSV)", csv, f"Ranking_Fondo{fondo_actual}_2020_2025_{etiqueta_modo}.csv", "text/csv"
    )

# ============================================================
# 6. HEATMAP ANUAL
# ============================================================
elif pagina == "🌡️ Heatmap Anual":
    st.title(f"🌡️ Mapa de Calor — Rentabilidad {etiqueta_modo} Anual — Fondo {fondo_actual}")
    st.caption("Verde = ganancia, rojo = pérdida. Ordenado por antigüedad en el sistema.")

    if es_real:
        df_heatmap = retornos_anuales_reales(df_real, afps_existentes).copy()
        st.caption(
            "📏 En modo Real, el primer año de cada AFP queda sin dato: solo tenemos el "
            "cierre de año en UF, no un precio de entrada intra-año en UF."
        )
    else:
        df_heatmap = df_rent_hist[afps_existentes].copy()
    df_heatmap.index = df_heatmap.index.year
    years_of_service = df_heatmap.count()
    sorted_afps = years_of_service.sort_values(ascending=False).index.tolist()
    df_heatmap = df_heatmap[sorted_afps]

    fig = px.imshow(
        df_heatmap.T, color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
        text_auto='.1f', aspect='auto',
        labels=dict(x="Año", y="AFP", color="Rentabilidad (%)"),
        title=f'Mapa de Calor — Rentabilidad {etiqueta_modo} Fondo {fondo_actual} (Histórico Completo)',
    )
    fig.update_layout(height=500, template='plotly_white')
    fig.update_xaxes(type='category')
    render_chart(fig)

# ============================================================
# 7. DRAWDOWN
# ============================================================
elif pagina == "📉 Drawdown":
    st.title(f"📉 Drawdown Histórico — Caída desde Máximos — Fondo {fondo_actual}")
    st.markdown(
        "Este gráfico muestra el porcentaje de pérdida respecto al máximo "
        "histórico anterior, para medir el impacto de las crisis de mercado."
    )

    df_dd = df[afps_existentes].copy()
    drawdown_pct = ((df_dd - df_dd.cummax()) / df_dd.cummax()) * 100

    fig = go.Figure()
    for afp in afps_existentes:
        fig.add_trace(go.Scatter(x=drawdown_pct.index, y=drawdown_pct[afp], name=afp, mode='lines'))

    fig.add_vrect(x0=CRISIS_SUBPRIME[0], x1=CRISIS_SUBPRIME[1], fillcolor="red", opacity=0.12, line_width=0,
                  annotation_text="Crisis Subprime", annotation_position="top left")
    fig.add_vrect(x0=CRISIS_COVID[0], x1=CRISIS_COVID[1], fillcolor="red", opacity=0.12, line_width=0,
                  annotation_text="COVID-19", annotation_position="top left")
    fig.add_hline(y=0, line_color="black", line_width=0.8)

    fig.update_layout(
        title=f'Drawdown Histórico Fondo {fondo_actual} (Caída desde Máximos)',
        yaxis_title="Caída (%)", xaxis_title="Año",
        template='plotly_white', height=600, hovermode="x unified",
    )
    render_chart(fig)

# ============================================================
# 8. ESCALA LOGARÍTMICA
# ============================================================
elif pagina == "📐 Escala Logarítmica":
    st.title(f"📐 Visión de Largo Plazo: Escala Logarítmica — Fondo {fondo_actual}")
    st.markdown(
        "En periodos largos la escala lineal puede ser engañosa — da la sensación "
        "de que las crisis son una baja menor, o no se aprecia la inflación. La "
        "escala logarítmica muestra la tendencia real de crecimiento compuesto."
    )

    if es_real:
        base_plot = df_real
        eje_y = "Valor Cuota (en UF)"
        titulo = f'Evolución Valor Cuota Fondo {fondo_actual} REAL en UF (Escala Logarítmica)'
        st.caption("📏 Resolución anual en modo Real (ver nota en Evolución Histórica).")
    else:
        base_plot = df
        eje_y = "Valor Cuota ($) CLP"
        titulo = f'Evolución Valor Cuota Fondo {fondo_actual} Nominal (Escala Logarítmica)'

    fig = go.Figure()
    for afp in afps_existentes:
        if afp in base_plot.columns:
            modo_linea = 'lines+markers' if es_real else 'lines'
            fig.add_trace(go.Scatter(x=base_plot.index, y=base_plot[afp], name=afp, mode=modo_linea))

    fig.add_vrect(x0=CRISIS_SUBPRIME[0], x1=CRISIS_SUBPRIME[1], fillcolor="red", opacity=0.12, line_width=0,
                  annotation_text="Crisis Subprime", annotation_position="top left")
    fig.add_vrect(x0=CRISIS_COVID[0], x1=CRISIS_COVID[1], fillcolor="red", opacity=0.12, line_width=0,
                  annotation_text="COVID-19", annotation_position="top left")

    fig.update_yaxes(type="log", title=eje_y)
    fig.update_layout(
        title=titulo,
        xaxis_title="Fecha", template='plotly_white', height=600, hovermode="x unified",
    )
    render_chart(fig)

# ============================================================
# 9. VOLATILIDAD
# ============================================================
elif pagina == "〰️ Volatilidad":
    st.title(f"〰️ Análisis de Volatilidad Histórica (Rolling Volatility) — Fondo {fondo_actual}")
    st.markdown(
        "Mide el *\"nerviosismo\"* del fondo: la intensidad de las variaciones diarias, "
        "anualizada, en una ventana móvil."
    )

    window = st.slider("Ventana móvil (días hábiles):", min_value=30, max_value=250, value=90, step=10)

    df_returns = df[afps_existentes].pct_change(fill_method=None)
    df_volatility = df_returns.rolling(window=window).std() * np.sqrt(252) * 100

    fig = go.Figure()
    for afp in afps_existentes:
        fig.add_trace(go.Scatter(x=df_volatility.index, y=df_volatility[afp], name=afp, mode='lines',
                                  line=dict(width=1)))

    fig.add_vrect(x0=CRISIS_SUBPRIME[0], x1=CRISIS_SUBPRIME[1], fillcolor="red", opacity=0.12, line_width=0,
                  annotation_text="Crisis Subprime", annotation_position="top left")
    fig.add_vrect(x0=CRISIS_COVID[0], x1=CRISIS_COVID[1], fillcolor="red", opacity=0.12, line_width=0,
                  annotation_text="COVID-19", annotation_position="top left")

    fig.update_layout(
        title=f'Volatilidad Histórica Anualizada Fondo {fondo_actual} (Ventana Móvil {window} días)',
        yaxis_title="Volatilidad (%)", xaxis_title="Fecha",
        template='plotly_white', height=600, hovermode="x unified",
    )
    render_chart(fig)

# ============================================================
# 10. INTERÉS COMPUESTO (escenario fijo, comparativo entre AFP)
# ============================================================
elif pagina == "💰 Interés Compuesto (Fijo)":
    st.title(f"💰 El Poder del Interés Compuesto: Aporte vs Rentabilidad — Fondo {fondo_actual}")
    st.markdown(
        """
Para aterrizar los porcentajes a la realidad de un afiliado, se proyecta un escenario
fijo: un cotizante que aporta **$100.000 mensuales** durante **30 años**, usando el
**CAGR** (tasa de crecimiento anual compuesta) de cada AFP en la *era moderna* (2020–2025).
        """
    )
    if es_real:
        st.info(
            "💡 **Modo Real activo:** la tasa usada es el CAGR real (ajustado por UF). "
            "El monto proyectado queda expresado en **poder adquisitivo de hoy** (UF), "
            "asumiendo que el aporte mensual también se reajusta con la inflación "
            "(igual que ocurre con los sueldos reales en la práctica)."
        )

    aporte_mensual = 100_000
    anos = 30
    meses = anos * 12
    inversion_total = aporte_mensual * meses

    if es_real:
        promedios_modernos = ranking_cagr(df_real, afps_existentes, start_date='2020-01-01')
    else:
        promedios_modernos = ranking_cagr(df, afps_existentes, start_date='2020-01-01')
    promedios_modernos = promedios_modernos.rename(columns={'Rentabilidad Promedio': 'Rentabilidad_Anual'})
    df_sim = promedios_modernos[promedios_modernos['AFP'].isin(afps_existentes)].copy()

    df_sim['Tasa_Mensual'] = (1 + df_sim['Rentabilidad_Anual'] / 100) ** (1 / 12) - 1
    df_sim['Monto_Final'] = aporte_mensual * (((1 + df_sim['Tasa_Mensual']) ** meses - 1) / df_sim['Tasa_Mensual'])
    df_sim['Aporte'] = inversion_total
    df_sim['Ganancia'] = df_sim['Monto_Final'] - inversion_total
    df_sim['Pct_Aporte'] = (df_sim['Aporte'] / df_sim['Monto_Final']) * 100
    df_sim['Pct_Ganancia'] = (df_sim['Ganancia'] / df_sim['Monto_Final']) * 100
    df_sim = df_sim.sort_values('Monto_Final', ascending=False)

    sufijo_titulo = " (pesos de hoy)" if es_real else ""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sim['AFP'], y=df_sim['Aporte'], name=f'Capital Aportado (${int(inversion_total/1e6)}M)',
        marker_color='#4a6fa5',
        text=[f"{p:.0f}%<br>Aporte" for p in df_sim['Pct_Aporte']], textposition='inside',
    ))
    fig.add_trace(go.Bar(
        x=df_sim['AFP'], y=df_sim['Ganancia'], name=f'Rentabilidad (Era Moderna, {etiqueta_modo})', marker_color='#55a630',
        text=[f"{p:.0f}%<br>Ganancia" for p in df_sim['Pct_Ganancia']], textposition='inside',
    ))
    fig.update_layout(
        barmode='stack',
        title=f'Proyección a {anos} Años usando CAGR {etiqueta_modo} (2020–2025) — Fondo {fondo_actual}<br><sup>Base: Aporte de ${aporte_mensual:,}/mes{sufijo_titulo}</sup>',
        yaxis_title=f"Monto Acumulado (CLP{sufijo_titulo})", template='plotly_white', height=600,
    )
    fig.update_yaxes(tickformat=",.0f", ticksuffix=" $")
    for _, row in df_sim.iterrows():
        fig.add_annotation(x=row['AFP'], y=row['Monto_Final'], text=f"${row['Monto_Final']/1e6:.1f}M",
                            showarrow=False, yshift=15, font=dict(size=12, color='black'))
    render_chart(fig)

    st.markdown(f"**CAGR 2020–2025 usado para la proyección ({etiqueta_modo}):**")
    st.dataframe(df_sim[['AFP', 'Rentabilidad_Anual']].round(2).set_index('AFP'), use_container_width=True)

# ============================================================
# 11. SIMULADOR INTERACTIVO
# ============================================================
elif pagina == "🎛️ Simulador Interactivo":
    st.title(f"🎛️ Simulador Interactivo: Proyección de Jubilación — Fondo {fondo_actual}")
    st.markdown(
        "Calculadora de interés compuesto con escenarios *optimista / esperado / "
        "pesimista*, basada en el CAGR 2020–2025 de la AFP elegida."
    )
    if es_real:
        st.info(
            "💡 **Modo Real activo:** se usa el CAGR real (ajustado por UF). Los montos "
            "de abajo quedan expresados en **pesos de hoy** (poder adquisitivo constante), "
            "asumiendo que el aporte mensual también se reajusta con la inflación — no en "
            "pesos nominales futuros."
        )

    if es_real:
        promedios_modernos = ranking_cagr(df_real, afps_existentes, start_date='2020-01-01')
    else:
        promedios_modernos = ranking_cagr(df, afps_existentes, start_date='2020-01-01')
    promedios_modernos = promedios_modernos.rename(columns={'Rentabilidad Promedio': 'Rentabilidad_Anual'})
    promedios_modernos = promedios_modernos[promedios_modernos['AFP'].isin(afps_existentes)]

    col1, col2 = st.columns([1, 2])

    with col1:
        cap_ini = st.number_input("Capital Inicial ($):", min_value=0, value=0, step=100_000)
        aporte = st.number_input("Contribución Mensual ($):", min_value=0, value=100_000, step=10_000)
        anos = st.number_input("Años de Inversión:", min_value=1, max_value=60, value=30, step=1)
        var = st.number_input("Varianza (+/- %):", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        afp_sel = st.selectbox(
            "Seleccionar AFP:", afps_existentes,
            index=afps_existentes.index('HABITAT') if 'HABITAT' in afps_existentes else 0,
        )

    tasa_normal = promedios_modernos[promedios_modernos['AFP'] == afp_sel]['Rentabilidad_Anual'].values[0]
    tasa_opt = tasa_normal + var
    tasa_pes = tasa_normal - var

    def tasa_mensual(t_anual):
        return (1 + t_anual / 100) ** (1 / 12) - 1 if t_anual != 0 else 0

    tm_norm, tm_opt, tm_pes = tasa_mensual(tasa_normal), tasa_mensual(tasa_opt), tasa_mensual(tasa_pes)

    eje_x_anos = np.arange(0, anos + 1)
    val_capital = [cap_ini]
    val_normal, val_opt, val_pes = [cap_ini], [cap_ini], [cap_ini]

    def calc_vf(tm, meses):
        if tm == 0:
            return cap_ini + (aporte * meses)
        vf_base = cap_ini * ((1 + tm) ** meses)
        vf_aportes = aporte * (((1 + tm) ** meses - 1) / tm)
        return vf_base + vf_aportes

    for y in range(1, anos + 1):
        meses = y * 12
        val_capital.append(cap_ini + (aporte * meses))
        val_normal.append(calc_vf(tm_norm, meses))
        val_opt.append(calc_vf(tm_opt, meses))
        val_pes.append(calc_vf(tm_pes, meses))

    df_plot = pd.DataFrame({
        'Año': eje_x_anos,
        'Capital Aportado': val_capital,
        f'Pesimista ({tasa_pes:.2f}%)': val_pes,
        f'Esperado - {afp_sel} ({tasa_normal:.2f}%)': val_normal,
        f'Optimista ({tasa_opt:.2f}%)': val_opt,
    })
    df_melt = df_plot.melt(id_vars='Año', var_name='Escenario', value_name='Monto_Real')
    df_melt['Monto_Millones'] = df_melt['Monto_Real'] / 1_000_000

    colores_escenarios = {
        'Capital Aportado': '#7f7f7f',
        f'Pesimista ({tasa_pes:.2f}%)': '#d62728',
        f'Esperado - {afp_sel} ({tasa_normal:.2f}%)': '#1f77b4',
        f'Optimista ({tasa_opt:.2f}%)': '#2ca02c',
    }

    with col2:
        sufijo_titulo_sim = " · pesos de hoy" if es_real else ""
        fig = px.line(
            df_melt, x='Año', y='Monto_Millones', color='Escenario', markers=True,
            custom_data=['Monto_Real'], color_discrete_map=colores_escenarios,
            title=f'Calculadora de Interés Compuesto — Proyección {afp_sel} (Fondo {fondo_actual})<br><sup>(CAGR {etiqueta_modo} Era Moderna: 2020–2025{sufijo_titulo_sim})</sup>',
        )
        fig.update_layout(
            hovermode="x unified", yaxis_title=f"Monto Acumulado (Millones $ CLP{sufijo_titulo_sim})",
            xaxis_title="Años de Inversión", legend_title="Escenarios", template='plotly_white',
            height=550,
        )
        fig.update_traces(
            hovertemplate="%{data.name}<br><b>$ %{customdata[0]:,.0f}</b><extra></extra>",
            line=dict(width=3), marker=dict(size=6),
        )
        fig.update_yaxes(tickformat=",.0f", ticksuffix=" M")
        render_chart(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Escenario Pesimista", formato_pesos_clp(val_pes[-1]))
    c2.metric(f"Escenario Esperado ({afp_sel})", formato_pesos_clp(val_normal[-1]))
    c3.metric("Escenario Optimista", formato_pesos_clp(val_opt[-1]))

# ============================================================
# 12. CONCLUSIONES
# ============================================================
elif pagina == "✅ Conclusiones":
    st.title(f"✅ Conclusiones — Fondo {fondo_actual}")
    st.markdown(
        f"""
Tras el análisis de los datos históricos del Fondo {fondo_actual}, se concluye que:

1. **Resiliencia:** a pesar de enfrentar crisis financieras mayores (2008 y 2020),
   el fondo ha demostrado capacidad de recuperación en el mediano plazo.
2. **Volatilidad vs. Rentabilidad:** el nivel de riesgo del fondo se refleja en la
   magnitud de las caídas transitorias observadas en el gráfico de Drawdown.
3. **Consistencia:** las diferencias de rentabilidad entre AFPs, aunque parecen
   marginales año a año (décimas de porcentaje), generan un impacto significativo
   en el patrimonio final debido al interés compuesto.
4. **Interés compuesto:** buena parte del monto final del afiliado es pura
   rentabilidad, mientras que el aporte propio representa una fracción menor —
   revisa la sección "Interés Compuesto (Fijo)" para el desglose exacto de este fondo.
5. **Elección de fondo según edad:** mientras más joven, mayor capacidad de asumir
   riesgo (fondos A/B); al acercarse a la jubilación conviene moverse a fondos más
   conservadores (D/E), dado el menor tiempo para recuperarse de pérdidas.
        """
    )
