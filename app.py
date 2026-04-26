import streamlit as st
import joblib
import spacy
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Asistente Marista NLP", page_icon="🏫", layout="wide")

# Botón para forzar recarga en caso de errores de caché
if st.sidebar.button("🧹 Limpiar Caché y Recargar"):
    st.cache_resource.clear()
    st.rerun()

@st.cache_resource
def load_assets():
    try:
        # Cargamos con un timestamp simple para depuración
        modelo = joblib.load('modelo_admision_marista.pkl')
        vectorizer = joblib.load('vectorizador_tfidf.pkl')
        nlp = spacy.load("es_core_news_sm")
        return modelo, vectorizer, nlp
    except Exception as e:
        st.error("Error al cargar los modelos. Por favor ejecuta las Fases 1 a 4 primero.")
        st.stop()

modelo, vectorizer, nlp = load_assets()

# --- INTERFAZ ---
st.title("🏫 Gestión Inteligente de Admisión - Colegio Marista")
st.sidebar.header("Configuración del Sistema")
st.sidebar.info("Prototipo de Clasificación PLN")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Entrada de Consulta")
    mensaje_input = st.text_area("Pegue aquí el correo o mensaje de WhatsApp:", height=150, 
                                 placeholder="Ej: Hola, quisiera saber los costos para 1ero de primaria para mi hijo.")

    if st.button("🚀 Procesar con IA"):
        if mensaje_input:
            # A. Preprocesamiento
            doc = nlp(str(mensaje_input).lower())
            tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_digit]
            texto_final = " ".join(tokens)
            
            # B. Predicción
            vector = vectorizer.transform([texto_final])
            prediccion = modelo.predict(vector)[0]
            probabilidades = modelo.predict_proba(vector)
            confianza = probabilidades.max()

            # C. Mostrar Resultados
            st.success(f"### Categoría Detectada: {prediccion}")
            st.metric("Nivel de Confianza", f"{confianza:.2%}")

            # D. Sugerencia de Respuesta Dinámica
            st.subheader("📋 Acción Sugerida")
            respuestas = {
                "ADMIS_COSTOS": "Enviar tarifario de pensiones y cronograma de cuota de ingreso.",
                "ADMIS_VACANTES": "Consultar disponibilidad en el SIAGIE y confirmar vacante por grado.",
                "ADMIS_REQUIS_DOC": "Adjuntar lista de documentos (DNI, Partida, Ficha de matrícula).",
                "ADMIS_CITAS": "Programar charla informativa o entrevista en calendario."
            }
            sugerencia = respuestas.get(prediccion, "Derivar a la secretaría para evaluación manual.")
            st.info(sugerencia)
            
        else:
            st.error("Por favor, ingrese un texto para analizar.")

with col2:
    st.subheader("Análisis Técnico")
    if mensaje_input and 'prediccion' in locals():
        st.write("**Tokens extraídos:**")
        st.write(tokens)
        
        st.write("**Probabilidades:**")
        prob_df = pd.DataFrame(probabilidades, columns=modelo.classes_).T
        prob_df.columns = ['Probabilidad']
        st.bar_chart(prob_df)
