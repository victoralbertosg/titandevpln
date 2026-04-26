import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

print("Iniciando Fase 4: Entrenamiento del Modelo...")

# 1. Cargar el dataset etiquetado y preprocesado
df = pd.read_csv("dataset_listo_para_entrenar.csv")

# Filtrar posibles nulos generados en la limpieza
df = df.dropna(subset=['texto_preprocesado', 'label_sugerida'])

# 2. División de datos (80% entrenamiento / 20% evaluación)
X_train, X_test, y_train, y_test = train_test_split(
    df['texto_preprocesado'], 
    df['label_sugerida'], 
    test_size=0.20, 
    random_state=42,
    stratify=df['label_sugerida']
)

# 3. Vectorización TF-IDF
vectorizador = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
X_train_tfidf = vectorizador.fit_transform(X_train)
X_test_tfidf = vectorizador.transform(X_test)

print(f"Dimensiones de entrenamiento: {X_train_tfidf.shape}")

# 4. Entrenamiento del Modelo (Random Forest)
print("Entrenando RandomForest...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X_train_tfidf, y_train)

# 5. Exportación de los modelos
joblib.dump(modelo, 'modelo_admision_marista.pkl')
joblib.dump(vectorizador, 'vectorizador_tfidf.pkl')

# Guardamos los datos de test para la Fase 5
import pickle
with open('test_data.pkl', 'wb') as f:
    pickle.dump((X_test_tfidf, y_test), f)

print("Entrenamiento completado exitosamente. Modelos guardados ('modelo_admision_marista.pkl', 'vectorizador_tfidf.pkl').")
