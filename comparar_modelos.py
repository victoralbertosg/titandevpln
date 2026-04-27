import pandas as pd
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time

print("Iniciando Comparación de Modelos de Clasificación...")

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

# 4. Definición de modelos a comparar
modelos = {
    "Logistic Regression (Actual)": LogisticRegression(class_weight='balanced', random_state=42),
    "Naive Bayes (Multinomial)": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine (LinearSVC)": LinearSVC(random_state=42)
}

resultados = []

for nombre, modelo in modelos.items():
    print(f"Evaluando {nombre}...")
    
    # Medir tiempo de entrenamiento
    inicio_entrenamiento = time.time()
    modelo.fit(X_train_tfidf, y_train)
    fin_entrenamiento = time.time()
    tiempo_entrenamiento = fin_entrenamiento - inicio_entrenamiento
    
    # Realizar predicciones
    inicio_prediccion = time.time()
    y_pred = modelo.predict(X_test_tfidf)
    fin_prediccion = time.time()
    tiempo_prediccion = fin_prediccion - inicio_prediccion
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    resultados.append({
        "Modelo": nombre,
        "Accuracy": f1_score(y_test, y_pred, average='weighted'), # Uso F1 como proxy de performance balanceada
        "Precisión": precision,
        "Recall": recall,
        "F1-Score": f1,
        "T. Entrenamiento (s)": round(tiempo_entrenamiento, 4),
        "T. Predicción (s)": round(tiempo_prediccion, 4)
    })

# 5. Crear Cuadro Comparativo
df_resultados = pd.DataFrame(resultados)

# Guardar resultados en CSV primero
df_resultados.to_csv("comparativa_modelos.csv", index=False)

print("\n### CUADRO COMPARATIVO DE MODELOS ###")
try:
    print(df_resultados.to_markdown(index=False))
except ImportError:
    print(df_resultados.to_string(index=False))

print("\nComparativa guardada en 'comparativa_modelos.csv'")
