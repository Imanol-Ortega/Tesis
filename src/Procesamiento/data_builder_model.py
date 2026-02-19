import numpy as np
import pandas as pd
import os
import glob
import re
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessor import DataPreprocessor
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_RAW_PLANETS = os.path.join(BASE_DIR, 'data', 'mast', 'csv')
PATH_RAW_ANOMALIES = os.path.join(BASE_DIR, 'data', 'mast', 'anomalies_csv')
PATH_META_PLANETS = os.path.join(BASE_DIR, 'data', 'nea', 'confirmados', 'nasa_exoplanets.csv')
PATH_META_ANOMALIES_DIR = os.path.join(BASE_DIR, 'data', 'nea', 'anomalias')
DIR_OUTPUT = os.path.join(BASE_DIR, 'data', 'processed', 'test')
KP_EXTREME = 0.10
KT_EXTREME = 0.05
TEST_SAMPLES_PER_FILE = 1
def load_metadata_planets():
    if not os.path.exists(PATH_META_PLANETS): return None
    try:
        df = pd.read_csv(PATH_META_PLANETS, comment='#')
        df.columns = [c.strip().lower() for c in df.columns]
        if 'pl_name' in df.columns:
            df['clean_name'] = df['pl_name'].astype(str).apply(lambda x: re.sub(r'[\s_\-]', '', x).lower())
        return df
    except: return None
def load_metadata_anomalies():
    files = glob.glob(os.path.join(PATH_META_ANOMALIES_DIR, "TOI*.csv"))
    if not files: return None
    latest_file = sorted(files)[-1]
    print(f"Usando catálogo de anomalías: {os.path.basename(latest_file)}")
    try:
        df = pd.read_csv(latest_file, comment='#')
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except: return None
def get_params(filename, df_meta, is_anomaly=False):
    clean_filename = filename.replace('.csv', '')
    if is_anomaly:
        match = re.search(r'TIC_(\d+)', clean_filename)
        if not match: return None
        tic_id = int(match.group(1))
        col_id = next((c for c in df_meta.columns if 'tic' in c or 'tid' == c), None)
        if not col_id: return None
        row = df_meta[df_meta[col_id] == tic_id]
    else:
        clean_target = re.sub(r'[\s_\-]', '', clean_filename).lower()
        row = df_meta[df_meta['clean_name'] == clean_target]
    if row.empty: return None
    try:
        col_p = next((c for c in row.columns if 'period' in c or 'orbper' in c), None)
        P = float(row.iloc[0][col_p])
        col_t0 = next((c for c in row.columns if 'epoch' in c or 'tranmid' in c or 'orbtper' in c), None)
        T0 = float(row.iloc[0][col_t0])
        return P, T0
    except: return None
def build_test_dataset():
    print("GENERANDO DATASET DE PRUEBA (FASE 3)...")
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    processor = DataPreprocessor()
    df_planets = load_metadata_planets()
    df_anomalies = load_metadata_anomalies()
    if df_planets is None:
        print(" ERROR CRÍTICO: No se pudo cargar el catálogo de metadatos (nasa_exoplanets.csv).")
        return
    X_test = []
    y_test = []

    files_planets = glob.glob(os.path.join(PATH_RAW_PLANETS, "*.csv"))
    files_planets = files_planets[-int(len(files_planets)*0.3):]
    if len(files_planets) == 0:
        print("ADVERTENCIA: La lista de archivos para test está vacía. Usando TODOS los archivos disponibles como fallback.")
        files_planets = glob.glob(os.path.join(PATH_RAW_PLANETS, "*.csv"))
    print(f" Analizando {len(files_planets)} archivos candidatos para el Test Set...")
    count_c0 = 0
    print("   [1/4] Procesando Clase 0 (Planetas Confirmados)...")
    for filepath in files_planets:
        if df_planets is None: break
        P, T0 = get_params(os.path.basename(filepath), df_planets) or (None, None) # Obtiene los parámetros P y T0 para el planeta actual
        if not P: continue
        try:
            df = pd.read_csv(filepath, comment='#') # Lee el archivo CSV, ignorando líneas de comentario
            time = df.iloc[:, 1].values # Asume que la primera columna es flujo
            flux = df.iloc[:, 0].values # Asume que la segunda columna es tiempo
            median_val = np.nanmedian(flux) #Calcula la mediana ignorando los NaN para evitar que afecten la normalización
            if median_val > 0: # Evita división por cero o valores no numéricos
                depth = 1.0 - (np.min(flux) / median_val) # Calcula la profundidad del tránsito como la caída relativa al valor mediano
                if depth < 0.005: # Filtra tránsitos muy superficiales que podrían no ser detectables
                    continue # Si la profundidad es menor al umbral, se considera que no hay un tránsito claro
            curve = processor.process_curve_phase_folding(time, flux, P, T0, smooth=False) # Procesa la curva sin suavizado para la clase de prueba
            if curve is not None:
                X_test.append(curve) # Agrega la curva procesada a los datos de prueba
                y_test.append(0) # Etiqueta 0 para planetas confirmados (clase normal)
                count_c0 += 1 # Contador de muestras de clase 0

                print("   [4/4] Procesando Clase 1 (Error Extremo)...")
                P_bad = P * (1 + np.random.choice([1, -1]) * KP_EXTREME) #  Genera un período ruidoso con un error extremo
                T0_bad = T0 * (1 + np.random.choice([1, -1]) * KT_EXTREME) # Genera una época de tránsito ruidosa con un error
                curve_bad = processor.process_curve_phase_folding(time, flux, P_bad, T0_bad, smooth=False) # Procesa la curva con los parámetros ruidosos para simular un mal plegado
                if curve_bad is not None:
                    X_test.append(curve_bad) # Agrega la curva mal plegada a los datos de prueba
                    y_test.append(1) # Etiqueta 1 para curvas mal plegadas (clase anómala)
        except: continue
    print("   [2/4] Generando Clase 0 (Ruido Estelar Plano)...")
    for _ in range(count_c0 // 2): # Genera una cantidad de curvas de ruido plano igual a la mitad de las muestras de clase 0 para mantener un balance razonable
        noise = np.random.normal(0, 0.005, 2048) # Ruido gaussiano con media 0 y desviación estándar de 0.005 para simular variabilidad estelar leve
        flat_curve = 0.5 + noise # Curva plana centrada en 0.5 con ruido para simular estrellas sin tránsitos
        flat_curve = np.clip(flat_curve, 0, 1) # Limita el rango de la curva a [0, 1] para mantener la consistencia con las curvas reales
        X_test.append(flat_curve) # Agrega la curva de ruido plano a los datos de prueba
        y_test.append(0) # Etiqueta 0 para ruido estelar plano (clase normal)

    print("   [3/4] Procesando Clase 1 (Anomalías Reales / Falsos Positivos)...")
    files_anoms = glob.glob(os.path.join(PATH_RAW_ANOMALIES, "*.csv"))
    count_c1_b = 0
    if df_anomalies is not None:
        for filepath in files_anoms: # Itera sobre los archivos de anomalías
            P, T0 = get_params(os.path.basename(filepath), df_anomalies, is_anomaly=True) or (None, None) # Obtiene los parámetros P y T0 para la anomalía
            if not P: continue
            try:
                df = pd.read_csv(filepath) # Lee el archivo CSV de la anomalía
                time = df['timecorr'].values # Asume que la columna de tiempo corregido se llama 'timecorr'
                flux = df['pdcsap_flux'].values # Asume que la columna de flujo corregido se llama 'pdcsap_flux'
                curve = processor.process_curve_phase_folding(time, flux, P, T0, smooth=False) # Procesa la curva de la anomalía sin suavizado
                if curve is not None:
                    X_test.append(curve) # Agrega la curva de la anomalía a los datos de prueba
                    y_test.append(1)# Etiqueta 1 para anomalías reales (clase anómala)
                    count_c1_b += 1 # Contador de muestras de clase 1 reales
            except: continue
    else:
        print("No se encontró catálogo TOI. Saltando Tipo B.")
    # Guardar
    X_test = np.array(X_test)[..., np.newaxis] # Convierte la lista de curvas a un array de NumPy
    y_test = np.array(y_test)
    np.save(os.path.join(DIR_OUTPUT, "X_test.npy"), X_test)
    np.save(os.path.join(DIR_OUTPUT, "y_test.npy"), y_test)

    print(f"\nRESUMEN DATASET DE PRUEBA:")
    print(f"   Total Muestras: {len(y_test)}")
    print(f"   Clase 0 (Normal): {np.sum(y_test == 0)} (Planetas: {count_c0}, Ruido: {count_c0//2})")
    print(f"   Clase 1 (Anómalo): {np.sum(y_test == 1)} (Mal Plegado: {count_c0}, Reales: {count_c1_b})")

    print(f"Archivos guardados en {DIR_OUTPUT}")
if __name__ == "__main__":
    build_test_dataset()
