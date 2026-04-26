import pandas as pd

print("Iniciando Fase 2: Auto-Etiquetado...")
df = pd.read_csv("dataset_admision_limpio.csv")

def asignar_etiqueta_inicial(texto):
    texto_str = str(texto).lower()
    
    # Order matters: check most specific categories first
    # ADMIS_COSTOS: keywords related to money/payments
    if any(p in texto_str for p in ['pension', 'cuota', 'precio', 'pagar', 'costo', 'descuento', 'mensualidad', 'pago', 'tarifario']):
        return 'ADMIS_COSTOS'
        
    # ADMIS_VACANTES: keywords related to availability/seats (removed ambiguous "ingreso" and "matricula")
    if any(p in texto_str for p in ['vacante', 'cupo', 'disponibilidad', 'espacio', 'lugar', 'disponible', 'hay lugar', 'queda lugar']):
        return 'ADMIS_VACANTES'
        
    # ADMIS_REQUIS_DOC: keywords related to documentation/requirements
    if any(p in texto_str for p in ['documento', 'papel', 'requisito', 'partida', 'dni', 'traslado', 'postulante', 'documentacion', 'certificado', 'matricula']):
        return 'ADMIS_REQUIS_DOC'
        
    # ADMIS_CITAS: keywords related to meetings/visits
    if any(p in texto_str for p in ['entrevista', 'visita', 'reunion', 'cita', 'charla', 'cronograma', 'agendar']):
        return 'ADMIS_CITAS'
        
    return 'OTROS'

df['label_sugerida'] = df['texto_limpio'].apply(asignar_etiqueta_inicial)

df.to_csv("dataset_etiquetado_v1.csv", index=False)
print("Etiquetado inicial completado. Guardado como 'dataset_etiquetado_v1.csv'.")
print("Distribución de etiquetas:")
print(df['label_sugerida'].value_counts())
