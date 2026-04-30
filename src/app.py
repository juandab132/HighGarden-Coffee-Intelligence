import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import warnings
import gc # Para limpiar memoria

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="High Garden Coffee Intelligence",
    page_icon="☕",
    layout="wide"
)

# ─── CARGA DE DATOS ───────────────────────────────────────────
@st.cache_data
def cargar_datos(path):
    df = pd.read_parquet(path)
    year_cols = [c for c in df.columns if '/' in str(c)]
    df_long = df.melt(
        id_vars=['Country', 'Coffee type'],
        value_vars=year_cols,
        var_name='period',
        value_name='consumption'
    )
    df_long['year'] = df_long['period'].apply(lambda x: int(x.split('/')[0]))
    return df, df_long

# FIX 1: Movimos la importación fuera de la función caché para evitar bloqueos
from transformers import pipeline

@st.cache_resource
def cargar_modelo_nlp():
    # Usamos la versión "distilled" que es más ligera para el servidor
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=-1 # Forzamos CPU para evitar errores de CUDA en la nube
    )

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x70/1B5E20/FFFFFF?text=High+Garden+Coffee")
    st.markdown("### ⚙️ Configuración")
    archivo = st.file_uploader("📂 Cargar dataset .parquet", type=['parquet'])
    st.markdown("---")
    n_future = st.slider("Años a predecir", 1, 10, 5)

if archivo:
    df, df_long = cargar_datos(archivo)
else:
    st.sidebar.info("ℹ️ Carga tu archivo coffee_db.parquet")
    st.stop()

paises_disponibles = df['Country'].unique().tolist()

with st.sidebar:
    paises_sel = st.multiselect(
        "🌍 Filtrar países",
        paises_disponibles,
        default=paises_disponibles[:5]
    )

# ─── PREDICCIÓN ───────────────────────────────────────────────
@st.cache_data
def generar_predicciones(n_future):
    ts = df_long.groupby('year')['consumption'].sum().reset_index().sort_values('year')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(ts[['consumption']].values)
    WINDOW = 5

    weights = np.exp(np.linspace(0, 1, WINDOW))
    weights /= weights.sum()

    preds, seq = [], list(scaled.flatten())
    for _ in range(n_future):
        val = float(np.dot(weights, seq[-WINDOW:]))
        preds.append(val)
        seq.append(val)

    future = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    hist = ts['consumption'].values
    years = ts['year'].values
    return years, hist, future

years_hist, hist_vals, future_vals = generar_predicciones(n_future)
future_years = list(range(int(years_hist[-1])+1, int(years_hist[-1])+n_future+1))
r_min, r_max = future_vals * 0.92, future_vals * 1.08

# ─── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🔮 Predicción", "📈 Tendencias", "🤖 Chatbot IA"])

# TAB 1, 2 y 3 se mantienen igual que tu código original (Omitidos aquí por brevedad)
# [Inserta aquí tus bloques de Tab 1, 2 y 3 tal cual los tienes]

# ══ TAB 4: CHATBOT (FIX 2: Estabilidad de sesión) ══════════════
with tab4:
    st.markdown('### 🤖 Asistente IA — High Garden Coffee')
    
    contexto = f"Dataset 1990-2020. Highest consumption: Brazil. Prediction for {future_years[0]}: {future_vals[0]/1e9:.2f}B units."

    if "pregunta_activa" not in st.session_state:
        st.session_state.pregunta_activa = ""

    pregunta_input = st.text_input("Pregunta en inglés:", value=st.session_state.pregunta_activa)

    if st.button("🔍 Consultar", type="primary"):
        if pregunta_input:
            with st.spinner("Analizando con IA..."):
                try:
                    qa = cargar_modelo_nlp()
                    resultado = qa(question=pregunta_input, context=contexto)
                    st.success(resultado['answer'])
                    # FIX 3: Liberar memoria explícitamente
                    del qa
                    gc.collect()
                except Exception as e:
                    st.error(f"Error: {e}")

st.markdown("---")
st.caption("High Garden Coffee Platform | Juan David Sánchez Meza (Slash) - 2026")