import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC # Uso SVC con probability=True para tener predict_proba
import pickle

print("Entrenando múltiples modelos para comparación en la App...")

# 1. Cargar el dataset
df = pd.read_csv("dataset_listo_para_entrenar.csv")
df = df.dropna(subset=['texto_preprocesado', 'label_sugerida'])

# 2. División de datos
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

# 4. Entrenamiento de Modelos
modelos_config = {
    "Logistic_Regression": LogisticRegression(class_weight='balanced', random_state=42),
    "Naive_Bayes": MultinomialNB(),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

modelos_entrenados = {}
for nombre, modelo in modelos_config.items():
    print(f"Entrenando {nombre}...")
    modelo.fit(X_train_tfidf, y_train)
    modelos_entrenados[nombre] = modelo
    # Guardar cada uno
    joblib.dump(modelo, f'modelo_{nombre.lower()}.pkl')

# Guardar vectorizador y datos de test
joblib.dump(vectorizador, 'vectorizador_tfidf.pkl')
with open('test_data.pkl', 'wb') as f:
    pickle.dump((X_test_tfidf, y_test), f)

print("Todos los modelos han sido entrenados y guardados.")
