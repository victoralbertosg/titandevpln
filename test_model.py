import joblib
import spacy

def test_prediction():
    modelo = joblib.load('modelo_admision_marista.pkl')
    vectorizer = joblib.load('vectorizador_tfidf.pkl')
    nlp = spacy.load("es_core_news_sm")
    
    texto = "requisitos de la matricula primer grado"
    doc = nlp(texto.lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_digit]
    texto_final = " ".join(tokens)
    
    vector = vectorizer.transform([texto_final])
    prediccion = modelo.predict(vector)[0]
    probabilidades = modelo.predict_proba(vector)
    
    print(f"Texto original: {texto}")
    print(f"Texto procesado: {texto_final}")
    print(f"Tokens: {tokens}")
    print(f"Prediccion: {prediccion}")
    
    classes = modelo.classes_
    for cls, prob in zip(classes, probabilidades[0]):
        print(f"  {cls}: {prob:.4f}")

if __name__ == "__main__":
    test_prediction()
