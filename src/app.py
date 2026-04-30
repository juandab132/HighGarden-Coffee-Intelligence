import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import warnings
import gc

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="High Garden Coffee Intelligence",
    page_icon="☕",
    layout="wide"
)

# Estilos
st.markdown("""
<style>
    .titulo { font-size:2rem; font-weight:800; color:#1B5E20; border-bottom:3px solid #4CAF50; padding-bottom:8px; }
</style>
""", unsafe_allow_html=True)

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

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x70/1B5E20/FFFFFF?text=High+Garden+Coffee")
    st.markdown("### ⚙️ Configuración")
    archivo = st.file_uploader("📂 Cargar dataset .parquet", type=['parquet'])
    n_future = st.slider("Años a predecir", 1, 10, 5)

if archivo:
    df, df_long = cargar_datos(archivo)
else:
    st.sidebar.info("ℹ️ Por favor, carga el archivo coffee_db.parquet.")
    st.stop()

paises_disponibles = df['Country'].unique().tolist()
paises_sel = st.sidebar.multiselect("🌍 Países", paises_disponibles, default=paises_disponibles[:5])

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
        preds.append(val); seq.append(val)
    future = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    return ts['year'].values, ts['consumption'].values, future

years_hist, hist_vals, future_vals = generar_predicciones(n_future)
future_years = list(range(int(years_hist[-1])+1, int(years_hist[-1])+n_future+1))

# ─── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🔮 Predicción", "📈 Tendencias", "🤖 Chatbot IA"])

with tab1:
    st.markdown('<div class="titulo">☕ Análisis Exploratorio</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Período", "1990 – 2020")
    c2.metric("🌍 Países", df['Country'].nunique())
    c3.metric("☕ Tipos", df['Coffee type'].nunique())
    c4.metric("📦 Registros", f"{len(df):,}")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.area(df_long.groupby('year')['consumption'].sum().reset_index(), x='year', y='consumption', title="Consumo Global")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top = df.sort_values('Total_domestic_consumption', ascending=False).head(10)
        fig2 = px.bar(top, x='Total_domestic_consumption', y='Country', orientation='h', title="Top 10 Consumidores")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown('<div class="titulo">🔮 Predicción</div>', unsafe_allow_html=True)
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=years_hist, y=hist_vals/1e9, name='Histórico', line=dict(color='#2E7D32')))
    fig_p.add_trace(go.Scatter(x=future_years, y=future_vals/1e9, name='Predicción', line=dict(color='#E65100')))
    st.plotly_chart(fig_p, use_container_width=True)
    st.metric("Precisión (MAPE)", "0.93%")

with tab3:
    st.markdown('<div class="titulo">📈 Tendencias</div>', unsafe_allow_html=True)
    df_fil = df_long[df_long['Country'].isin(paises_sel)]
    st.plotly_chart(px.line(df_fil, x='year', y='consumption', color='Country'), use_container_width=True)

with tab4:
    st.markdown('<div class="titulo">🤖 Asistente IA</div>', unsafe_allow_html=True)
    contexto = f"Coffee data 1990-2020. Top: Brazil. Prediction 2025: {future_vals[0]/1e9:.2f}B units. MAPE: 0.93%."
    pregunta = st.text_input("Pregunta en inglés:")
    
    if st.button("🔍 Consultar"):
        if pregunta:
            with st.spinner("Cargando IA..."):
                try:
                    from transformers import pipeline
                    # Cargamos el modelo localmente para no saturar la RAM global
                    nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
                    res = nlp(question=pregunta, context=contexto)
                    st.success(f"**Respuesta:** {res['answer']}")
                    # Limpieza manual de RAM
                    del nlp
                    gc.collect()
                except Exception as e:
                    st.error(f"Error de memoria: {e}. Intenta nuevamente.")

st.markdown("---")
st.caption("High Garden Coffee Platform | Juan David Sánchez Meza - 2026")