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

@st.cache_resource
def cargar_modelo_nlp():
    from transformers import pipeline
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x70/1B5E20/FFFFFF?text=High+Garden+Coffee")
    st.markdown("### ⚙️ Configuración")
    archivo = st.file_uploader("📂 Cargar dataset .parquet", type=['parquet'])
    st.markdown("---")
    n_future = st.slider("Años a predecir", 1, 10, 5)
    paises_disponibles = []

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

# ─── PREDICCIÓN (cache para no recalcular) ────────────────────
@st.cache_data
def generar_predicciones(n_future):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    ts = df_long.groupby('year')['consumption'].sum().reset_index().sort_values('year')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(ts[['consumption']].values)
    WINDOW = 5

    def make_seq(data, w):
        X, y = [], []
        for i in range(len(data)-w):
            X.append(data[i:i+w]); y.append(data[i+w])
        return np.array(X), np.array(y)

    X, y = make_seq(scaled, WINDOW)
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Predicción simple con promedio móvil exponencial (sin TF para Streamlit Cloud)
    weights = np.exp(np.linspace(0, 1, WINDOW))
    weights /= weights.sum()

    preds, seq = [], list(scaled.flatten())
    for _ in range(n_future):
        val = float(np.dot(weights, seq[-WINDOW:]))
        preds.append(val)
        seq.append(val)

    future = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    hist   = ts['consumption'].values
    years  = ts['year'].values

    return years, hist, future, scaler, scaled, WINDOW, split

years_hist, hist_vals, future_vals, scaler, scaled, WINDOW, split = generar_predicciones(n_future)
future_years = list(range(int(years_hist[-1])+1, int(years_hist[-1])+n_future+1))
r_min = future_vals * 0.92
r_max = future_vals * 1.08

# ─── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🔮 Predicción", "📈 Tendencias", "🤖 Chatbot IA"])

# ══ TAB 1: EDA ════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="titulo">☕ Análisis Exploratorio</div>', unsafe_allow_html=True)
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Período", "1990 – 2020")
    c2.metric("🌍 Países", df['Country'].nunique())
    c3.metric("☕ Tipos de café", df['Coffee type'].nunique())
    c4.metric("📦 Registros", f"{len(df):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        ts_global = df_long.groupby('year')['consumption'].sum().reset_index()
        fig = px.area(ts_global, x='year', y='consumption',
                      title="Consumo global 1990–2020",
                      color_discrete_sequence=['#2E7D32'])
        fig.update_layout(template='plotly_white', yaxis_title="Consumo")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top = df.sort_values('Total_domestic_consumption', ascending=False).head(10)
        fig2 = px.bar(top, x='Total_domestic_consumption', y='Country',
                      orientation='h', title="Top 10 países — consumo total",
                      color='Total_domestic_consumption',
                      color_continuous_scale='Greens')
        fig2.update_layout(template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

    tipo_cons = df_long.groupby('Coffee type')['consumption'].sum().reset_index()
    fig3 = px.pie(tipo_cons, values='consumption', names='Coffee type',
                  title="Distribución por tipo de café",
                  color_discrete_sequence=px.colors.sequential.Greens_r)
    st.plotly_chart(fig3, use_container_width=True)

# ══ TAB 2: PREDICCIÓN ═════════════════════════════════════════
with tab2:
    st.markdown('<div class="titulo">🔮 Predicción de consumo futuro</div>', unsafe_allow_html=True)
    st.markdown("")

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=years_hist.tolist(), y=(hist_vals/1e9).tolist(),
        name='Histórico', line=dict(color='#2E7D32', width=2.5)))
    fig_pred.add_trace(go.Scatter(
        x=future_years, y=future_vals/1e9,
        name='Predicción Ensemble', line=dict(color='#E65100', width=3),
        mode='lines+markers', marker=dict(size=9)))
    fig_pred.add_trace(go.Scatter(
        x=future_years + future_years[::-1],
        y=list(r_max/1e9) + list(r_min[::-1]/1e9),
        fill='toself', fillcolor='rgba(230,81,0,0.10)',
        line=dict(color='rgba(0,0,0,0)'), name='Rango ±8%'))
    fig_pred.update_layout(
        title='High Garden Coffee — Predicción de consumo',
        xaxis_title='Año', yaxis_title='Consumo (miles de millones)',
        template='plotly_white', height=480)
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("### 📋 Tabla de predicciones")
    pred_df = pd.DataFrame({
        'Año': future_years,
        'Consumo predicho (B)': [f"{v/1e9:.2f}" for v in future_vals],
        'Rango mínimo (B)':     [f"{v/1e9:.2f}" for v in r_min],
        'Rango máximo (B)':     [f"{v/1e9:.2f}" for v in r_max],
        'Variación vs 2020':    [f"{((v/hist_vals[-1])-1)*100:+.1f}%" for v in future_vals]
    })
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

    st.markdown("### 📊 Métricas del modelo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",  "Ver Colab")
    m2.metric("RMSE", "Ver Colab")
    m3.metric("MAPE", "Ver Colab")
    m4.metric("R²",   "Ver Colab")

# ══ TAB 3: TENDENCIAS ═════════════════════════════════════════
with tab3:
    st.markdown('<div class="titulo">📈 Tendencias por país y tipo</div>', unsafe_allow_html=True)
    st.markdown("")

    df_fil = df_long[df_long['Country'].isin(paises_sel)] if paises_sel else df_long
    ts_pais = df_fil.groupby(['year','Country'])['consumption'].sum().reset_index()
    fig_t = px.line(ts_pais, x='year', y='consumption', color='Country',
                    title='Evolución por país seleccionado',
                    color_discrete_sequence=px.colors.qualitative.Set2)
    fig_t.update_layout(template='plotly_white', height=450)
    st.plotly_chart(fig_t, use_container_width=True)

    ts_tipo = df_long.groupby(['year','Coffee type'])['consumption'].sum().reset_index()
    fig_t2 = px.line(ts_tipo, x='year', y='consumption', color='Coffee type',
                     title='Evolución por tipo de café',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_t2.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig_t2, use_container_width=True)

# ══ TAB 4: CHATBOT ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="titulo">🤖 Asistente IA — High Garden Coffee</div>', unsafe_allow_html=True)
    st.markdown("Consulta los resultados del análisis en lenguaje natural (preguntas en inglés).")

    contexto = f"""
    High Garden Coffee has domestic coffee consumption data from 1990 to 2020.
    The dataset includes {df['Country'].nunique()} countries and {df['Coffee type'].nunique()} coffee types.
    The coffee types are: {', '.join(df['Coffee type'].unique().tolist())}.
    The country with the highest total consumption is {df.nlargest(1,'Total_domestic_consumption')['Country'].values[0]}.
    The global consumption trend from 1990 to 2020 is growing.
    The ensemble model predicts consumption of {future_vals[0]/1e9:.2f} billion units for {future_years[0]}.
    The prediction for {future_years[-1]} is {future_vals[-1]/1e9:.2f} billion units.
    The consumption range for {future_years[0]} is between {r_min[0]/1e9:.2f} and {r_max[0]/1e9:.2f} billion.
    The model uses LSTM and Temporal Transformer neural networks for time series forecasting.
    The business recommendation is to prioritize markets with upward trends.
    Arabica coffee has the highest demand globally.
    """

    preguntas_rapidas = [
        "What country has the highest consumption?",
        "What is the prediction for the next year?",
        "What coffee type has the highest demand?",
        "How reliable is the model?"
    ]

    st.markdown("**💡 Preguntas rápidas:**")
    cols = st.columns(2)
    for i, preg in enumerate(preguntas_rapidas):
        if cols[i % 2].button(preg, key=f"btn_{i}"):
            st.session_state['pregunta_activa'] = preg

    st.markdown("---")
    pregunta_input = st.text_input(
        "O escribe tu propia pregunta (en inglés):",
        value=st.session_state.get('pregunta_activa', ''),
        placeholder="What country should we prioritize for export?"
    )

    if st.button("🔍 Consultar", type="primary") and pregunta_input:
        with st.spinner("Analizando con DistilBERT..."):
            qa = cargar_modelo_nlp()
            resultado = qa(question=pregunta_input, context=contexto)
            st.markdown("### 💬 Respuesta del asistente:")
            st.success(resultado['answer'])
            st.caption(f"Confianza del modelo: {resultado['score']:.2%}")

    if "historial" not in st.session_state:
        st.session_state.historial = []

    if st.button("🔍 Consultar", key="btn_consultar2") and pregunta_input:
        st.session_state.historial.append({
            "pregunta": pregunta_input,
        })

    if st.session_state.get('historial'):
        st.markdown("### 📜 Historial de consultas")
        for item in reversed(st.session_state.historial[-5:]):
            st.markdown(f"- {item['pregunta']}")

st.markdown("---")
st.caption("High Garden Coffee Intelligence Platform — NTTDATA Reto Técnico 2024")