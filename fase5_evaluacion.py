import joblib
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Iniciando Fase 5: Evaluación del Modelo...")

# 1. Cargar el modelo, el vectorizador y los datos de prueba
modelo = joblib.load('modelo_admision_marista.pkl')
vectorizador = joblib.load('vectorizador_tfidf.pkl')

with open('test_data.pkl', 'rb') as f:
    X_test_tfidf, y_test = pickle.load(f)

# 2. Realizar las predicciones con los datos que el modelo NO conoce
y_pred = modelo.predict(X_test_tfidf)

# 3. Generar el reporte de métricas detallado
print("### REPORTE DE CLASIFICACIÓN ###")
reporte = classification_report(y_test, y_pred)
print(reporte)

# 4. Calcular la Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
etiquetas = sorted(y_test.unique())

# 5. Visualización Gráfica de la Matriz de Confusión
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=etiquetas, yticklabels=etiquetas)

plt.title('Matriz de Confusión: Clasificación de Admisión')
plt.xlabel('Predicción del Sistema')
plt.ylabel('Valor Real (Padre de Familia)')
plt.savefig('matriz_confusion.png')
print("Gráfico de Matriz de Confusión guardado como 'matriz_confusion.png'")

# 6. Mostrar exactitud global
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud Global (Accuracy): {accuracy:.2%}")
