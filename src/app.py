import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="High Garden Coffee Intelligence",
    page_icon="☕",
    layout="wide"
)

st.markdown("""
<style>
    .titulo { font-size:2rem; font-weight:800; color:#1B5E20; border-bottom:3px solid #4CAF50; padding-bottom:8px; }
    .kpi { background:#E8F5E9; border-left:4px solid #2E7D32; border-radius:8px; padding:12px 16px; }
</style>
""", unsafe_allow_html=True)

# ─── CARGA DE DATOS ─────────────────────────────
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

# ─── MODELO NLP ────────────────────────────────
@st.cache_resource
def cargar_modelo_nlp():
    from transformers import pipeline
    return pipeline(
        task="question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

# ─── SIDEBAR ───────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    archivo = st.file_uploader("📂 Cargar dataset .parquet", type=['parquet'])
    n_future = st.slider("Años a predecir", 1, 10, 5)

if not archivo:
    st.info("Carga tu archivo .parquet para continuar")
    st.stop()

df, df_long = cargar_datos(archivo)

paises_sel = st.sidebar.multiselect(
    "🌍 Filtrar países",
    df['Country'].unique(),
    default=df['Country'].unique()[:5]
)

# ─── PREDICCIÓN ────────────────────────────────
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
    return ts, future

ts, future_vals = generar_predicciones(n_future)

# ─── TABS ──────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🔮 Predicción", "📈 Tendencias", "🤖 Chatbot IA"])

# ══ TAB CHATBOT ════════════════════════════════
with tab4:
    st.markdown("## 🤖 Chatbot IA")

    contexto = f"""
    Coffee consumption dataset from 1990 to 2020.
    Countries: {df['Country'].nunique()}.
    Coffee types: {', '.join(df['Coffee type'].unique().tolist())}.
    Top country: {df.nlargest(1,'Total_domestic_consumption')['Country'].values[0]}.
    Trend: growing.
    """

    pregunta = st.text_input("Haz una pregunta en inglés:")

    if st.button("Consultar") and pregunta:
        with st.spinner("Pensando..."):
            try:
                qa = cargar_modelo_nlp()
                res = qa(question=pregunta, context=contexto)

                st.success(res["answer"])
                st.caption(f"Confianza: {res['score']:.2%}")

            except Exception as e:
                st.error(f"Error en el modelo: {str(e)}")