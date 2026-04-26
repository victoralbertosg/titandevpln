import spacy
import pandas as pd

print("Iniciando Fase 3: Preprocesamiento NLP...")

# Cargar el modelo de lenguaje español
nlp = spacy.load("es_core_news_sm")

def preprocesamiento_avanzado(texto):
    doc = nlp(str(texto))
    # Tokenización, Eliminación de Stopwords/Puntuación, Lematización
    tokens_limpios = [
        token.lemma_.lower()
        for token in doc 
        if not token.is_stop and not token.is_punct and not token.is_digit
    ]
    return " ".join(tokens_limpios)

# Aplicar al dataset etiquetado
df = pd.read_csv("dataset_etiquetado_v1.csv")

print(f"Procesando {len(df)} registros. Esto puede tardar un poco...")
df['texto_preprocesado'] = df['texto_limpio'].apply(preprocesamiento_avanzado)

df.to_csv("dataset_listo_para_entrenar.csv", index=False)

print("Preprocesamiento completado. Guardado como 'dataset_listo_para_entrenar.csv'.")
print("Ejemplo de transformación:")
if len(df) > 0:
    print(f"Original: {df['texto_limpio'].iloc[0]}")
    print(f"Preprocesado: {df['texto_preprocesado'].iloc[0]}")
