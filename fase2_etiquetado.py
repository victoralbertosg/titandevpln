import pandas as pd

print("Iniciando Fase 2: Auto-Etiquetado...")
df = pd.read_csv("dataset_admision_limpio.csv")

def asignar_etiqueta_inicial(texto):
    reglas = {
        'ADMIS_COSTOS': ['pension', 'cuota', 'precio', 'pagar', 'costo', 'descuento', 'mensualidad', 'pago'],
        'ADMIS_VACANTES': ['vacante', 'cupo', 'disponibilidad', 'espacio', 'lugar', 'ingreso', 'matricula', 'disponible'],
        'ADMIS_REQUIS_DOC': ['documento', 'papel', 'requisito', 'partida', 'dni', 'traslado', 'postulante', 'documentacion'],
        'ADMIS_CITAS': ['entrevista', 'visita', 'reunion', 'cita', 'charla', 'cronograma']
    }
    
    for etiqueta, palabras in reglas.items():
        if any(palabra in str(texto) for palabra in palabras):
            return etiqueta
    return 'OTROS'

df['label_sugerida'] = df['texto_limpio'].apply(asignar_etiqueta_inicial)

df.to_csv("dataset_etiquetado_v1.csv", index=False)
print("Etiquetado inicial completado. Guardado como 'dataset_etiquetado_v1.csv'.")
print("Distribución de etiquetas:")
print(df['label_sugerida'].value_counts())
