import streamlit as st
import joblib
import spacy
import pandas as pd
import unicodedata

# Configuración de la página
st.set_page_config(page_title="Asistente Marista NLP - Comparador", page_icon="🏫", layout="wide")

# Diseño Premium
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #1e3a8a;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        modelos = {
            "Logistic Regression": joblib.load('modelo_logistic_regression.pkl'),
            "Naive Bayes": joblib.load('modelo_naive_bayes.pkl'),
            "Random Forest": joblib.load('modelo_random_forest.pkl'),
            "SVM": joblib.load('modelo_svm.pkl')
        }
        vectorizer = joblib.load('vectorizador_tfidf.pkl')
        nlp = spacy.load("es_core_news_sm")
        return modelos, vectorizer, nlp
    except Exception as e:
        st.error(f"Error al cargar los activos: {e}. Asegúrate de ejecutar fase4_multi_modelo.py")
        st.stop()

modelos, vectorizer, nlp = load_assets()

# --- INTERFAZ ---
st.title("🏫 Sistema de Clasificación de Consultas - Colegio Marista")
st.markdown("### Prototipo Multimodelo para Análisis de Admisión")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_Gemini_logo.svg/2560px-Google_Gemini_logo.svg.png", width=200)
st.sidebar.header("Configuración")
if st.sidebar.button("🧹 Recargar Modelos"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.info("Este sistema compara 4 arquitecturas de IA para clasificar mensajes de padres de familia de forma automática.")

# Entrada de Datos
mensaje_input = st.text_area("📄 Ingrese el mensaje del padre de familia:", height=120, 
                             placeholder="Ej: Estimados, deseo conocer los requisitos y costos para inicial de 4 años...")

if st.button("🚀 Procesar con IA (Multimodelo)"):
    if mensaje_input:
        # Preprocesamiento
        texto_base = str(mensaje_input).lower()
        texto_base = unicodedata.normalize('NFKD', texto_base).encode('ascii', 'ignore').decode('ascii')
        doc = nlp(texto_base)
        tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_digit]
        texto_final = " ".join(tokens)
        vector = vectorizer.transform([texto_final])

        # Creación de Pestañas
        tabs = st.tabs(list(modelos.keys()))

        respuestas = {
            "ADMIS_COSTOS": "Enviar tarifario de pensiones y cronograma de cuota de ingreso.",
            "ADMIS_VACANTES": "Consultar disponibilidad en el SIAGIE y confirmar vacante por grado.",
            "ADMIS_REQUIS_DOC": "Adjuntar lista de documentos (DNI, Partida, Ficha de matrícula).",
            "ADMIS_CITAS": "Programar charla informativa o entrevista en calendario."
        }

        for i, (nombre, modelo) in enumerate(modelos.items()):
            with tabs[i]:
                col_res, col_tech = st.columns([2, 1])
                
                # Predicción
                prediccion = modelo.predict(vector)[0]
                probabilidades = modelo.predict_proba(vector)
                confianza = probabilidades.max()
                
                with col_res:
                    st.success(f"### Resultado: {prediccion}")
                    st.write(f"**Confianza:** {confianza:.2%}")
                    st.progress(confianza)
                    
                    st.subheader("📋 Acción Sugerida")
                    sugerencia = respuestas.get(prediccion, "Derivar a la secretaría.")
                    st.info(sugerencia)
                
                with col_tech:
                    st.subheader("Análisis Técnico")
                    st.write("**Probabilidades por Clase:**")
                    prob_df = pd.DataFrame(probabilidades, columns=modelo.classes_).T
                    prob_df.columns = ['Probabilidad']
                    st.bar_chart(prob_df)
                    
                    with st.expander("Ver Tokens"):
                        st.write(tokens)
    else:
        st.error("Por favor, ingrese un mensaje para analizar.")

# Sección Informativa Inferior
with st.expander("ℹ️ Información sobre los modelos"):
    st.write("""
    - **Logistic Regression**: Robusto y rápido para clasificación lineal.
    - **Naive Bayes**: Basado en probabilidad, extremadamente ligero.
    - **Random Forest**: Ensamble de árboles de decisión, maneja bien datos no lineales.
    - **SVM**: Busca el hiperplano óptimo de separación entre clases.
    """)
