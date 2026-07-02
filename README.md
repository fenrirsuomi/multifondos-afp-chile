# Fondo A — Análisis AFP Chile (app Streamlit)

App web interactiva migrada desde el notebook `fondosA.ipynb`. Incluye evolución
histórica, ranking, heatmap, drawdown, volatilidad y un simulador interactivo de
interés compuesto (reemplaza los `ipywidgets` originales por controles nativos de
Streamlit, que sí funcionan fuera de Jupyter).

## Estructura
```
fondos_app/
├── app.py                             # app principal
├── Fondo_A_Valor_Cuota_Final.csv      # datos (debe estar en la misma carpeta)
├── requirements.txt
└── README.md
```

## 1. Correrlo en tu computador

```bash
# (opcional pero recomendado) crear un entorno virtual
python -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate

# instalar dependencias
pip install -r requirements.txt

# ejecutar
streamlit run app.py
```

Se abrirá automáticamente en `http://localhost:8501`.

## 2. Publicarla gratis en internet (Streamlit Community Cloud)

1. Sube esta carpeta a un repositorio de GitHub (público o privado), asegurándote
   de incluir `app.py`, `requirements.txt` y el CSV.
2. Entra a https://share.streamlit.io con tu cuenta de GitHub.
3. Click en "New app" → selecciona el repo, la rama y `app.py` como archivo
   principal.
4. Deploy. En 1-2 minutos tendrás una URL pública tipo
   `https://tu-usuario-fondosa.streamlit.app`.

No necesitas servidor propio ni tarjeta de crédito para este plan gratuito.

## 3. Notas sobre la migración desde el notebook

- Los gráficos de `matplotlib`/`seaborn` se convirtieron a **Plotly** para que
  sean interactivos en el navegador (zoom, hover, exportar imagen).
- El simulador de `ipywidgets` (botón + panel) se reemplazó por controles nativos
  de Streamlit (`number_input`, `selectbox`, `slider`); Streamlit re-ejecuta el
  script automáticamente al cambiar cualquier valor, así que no hace falta botón
  "Calcular".
- La limpieza de datos (unión de CSV originales, fix AFP Capital 2009, fix del
  primer año 2020) se mantuvo igual, encapsulada en funciones cacheadas
  (`@st.cache_data`) para que la app cargue rápido.
- Si más adelante quieres actualizar los datos, basta con reemplazar
  `Fondo_A_Valor_Cuota_Final.csv` por una versión más reciente con las mismas
  columnas (`Fecha` + una columna por AFP).
