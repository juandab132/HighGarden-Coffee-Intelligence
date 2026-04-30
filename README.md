# ☕ High Garden Coffee Intelligence Platform

Solución avanzada de analítica predictiva y procesamiento de lenguaje natural desarrollada para el reto técnico de **NTT DATA 2026**. Esta plataforma permite la visualización de datos históricos, la predicción de consumo global y la consulta de insights mediante IA Generativa.

## 🚀 Características de la Solución

- **Análisis Exploratorio (EDA):** Visualización interactiva del consumo histórico (1990-2020) segmentado por país y variedad de café.
- **Predicción con Deep Learning:** Implementación de modelos **LSTM** y **Temporal Transformers** para proyecciones de mercado a largo plazo.
- **Bonus de IA Generativa:** Chatbot inteligente basado en **DistilBERT** (Question-Answering) para consultas en lenguaje natural sobre las métricas y tendencias del negocio.
- **Dashboard Interactivo:** Interfaz desplegada en la nube que permite la carga dinámica de datos y visualización de resultados en tiempo real.

## 📊 Métricas de Desempeño

El modelo ha sido evaluado rigurosamente para garantizar decisiones empresariales conscientes basadas en datos:

- **MAPE (Mean Absolute Percentage Error):** 0.93% (Precisión superior al 99%).
- **R² Score:** 0.6056.
- **Predicción 2025:** Se proyecta un consumo global de **2.63 mil millones de unidades**.

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.10+
- **Machine Learning:** TensorFlow, Keras, Scikit-learn.
- **NLP:** Hugging Face Transformers (DistilBERT).
- **Frontend:** Streamlit.
- **Visualización:** Plotly, Seaborn.
- **Gestión de Datos:** PyArrow (Lectura de Parquet).

## 📁 Estructura del Proyecto

```text
├── data/           # Dataset original (Parquet)
├── models/         # Modelos entrenados y escaladores (.h5, .pkl)
├── notebooks/      # Documentación del experimento (Google Colab)
├── app.py          # Aplicación principal de Streamlit
└── requirements.txt # Dependencias del proyecto
```
