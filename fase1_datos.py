import pandas as pd
import random
import re

print("Iniciando Fase 1: Generación y Limpieza de Datos...")

# 1. Generación de datos sintéticos
patrones = [
    {"fuente": "Email", "plantilla": "Estimados, quisiera saber los precios de {grado} para mi hijo {nombre}."},
    {"fuente": "WhatsApp", "plantilla": "Hola hay vacantes para {grado}? Mi niño se llama {nombre}."},
    {"fuente": "Web", "plantilla": "Requisitos para traslado a {grado}. Postulante: {nombre}."},
    {"fuente": "Email", "plantilla": "Solicito información de pensiones y cuota de ingreso para {grado}."},
    {"fuente": "WhatsApp", "plantilla": "Saber si hay cupos en {grado} para {nombre}. Gracias."},
    {"fuente": "Web", "plantilla": "Cronograma de admision y charlas para {grado}."}
]

grados = ["inicial 3 años", "inicial 4 años", "inicial 5 años", "1ero primaria", "6to primaria", "1ero secundaria", "5to secundaria"]
nombres = ["Mateo", "Lucia", "Sebastian", "Sofia", "Thiago", "Valentina", "Nicolas", "Elena", "Gabriel", "Camila"]

datos_sinteticos = []
# Generamos 5000 registros
for _ in range(5000):
    p = random.choice(patrones)
    mensaje = p["plantilla"].format(grado=random.choice(grados), nombre=random.choice(nombres))
    # Introducimos un poco de ruido a veces para mayor realismo (ej. mayúsculas, html)
    if random.random() < 0.2:
        mensaje = f"<p><b>{mensaje}</b></p>"
    elif random.random() < 0.2:
        mensaje = mensaje.upper()
        
    datos_sinteticos.append({"fuente": p["fuente"], "contenido": mensaje})

df_masivo = pd.DataFrame(datos_sinteticos)
df_masivo.to_csv("datos_admision_masivos.csv", index=False, encoding='utf-8')
print("Archivo 'datos_admision_masivos.csv' generado con 5,000 registros.")

# 2. Limpieza de datos
def limpiar_texto(texto):
    if pd.isna(texto): return ""
    texto = re.sub(r'<.*?>', '', str(texto)) # Quitar HTML
    texto = texto.lower() # Minúsculas
    texto = re.sub(r'[^\w\s]', '', texto) # Quitar puntuación
    texto = " ".join(texto.split()) # Quitar espacios extra
    return texto

df = pd.read_csv("datos_admision_masivos.csv")
df['texto_limpio'] = df['contenido'].apply(limpiar_texto)
df = df.drop_duplicates(subset=['texto_limpio'])

df.to_csv("dataset_admision_limpio.csv", index=False)
print(f"Dataset limpio guardado como 'dataset_admision_limpio.csv' con {len(df)} registros únicos.")
