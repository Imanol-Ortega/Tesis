import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import sys

# --- CONFIGURACIÓN DE RUTAS E IMPORTACIONES ---
BASE_DIR_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR_SRC)

from Procesamiento.preprocessor import DataPreprocessor

BASE_DIR = os.path.dirname(BASE_DIR_SRC)
PATH_RAW_PLANETS = os.path.join(BASE_DIR, 'data', 'mast', 'csv')
PATH_RAW_ANOMALIES = os.path.join(BASE_DIR, 'data', 'mast', 'anomalies_csv')
PATH_META_PLANETS = os.path.join(BASE_DIR, 'data', 'nea', 'confirmados', 'nasa_exoplanets.csv')
PATH_META_ANOMALIES_DIR = os.path.join(BASE_DIR, 'data', 'nea', 'anomalias')
DIR_RESULTS = os.path.join(BASE_DIR, 'results', 'validacion_fase3')

KP_EXTREME = 0.10
KT_EXTREME = 0.05

# --- FUNCIONES DE CARGA DE METADATOS ---
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
    files = glob.glob(os.path.join(PATH_META_ANOMALIES_DIR, "*.csv"))
    if not files:
        print(f"⚠️ No se encontraron metadatos de anomalías en: {PATH_META_ANOMALIES_DIR}")
        return None
    latest_file = sorted(files)[-1]
    try:
        df = pd.read_csv(latest_file, comment='#')
        df.columns = [c.strip().lower() for c in df.columns]
        # Crear clean_name por si falla la búsqueda por TIC/TOI
        col_name = next((c for c in df.columns if 'name' in c or 'id' in c), df.columns[0])
        df['clean_name'] = df[col_name].astype(str).apply(lambda x: re.sub(r'[\s_\-]', '', x).lower())
        return df
    except Exception as e:
        print(f"⚠️ Error cargando metadatos de anomalías: {e}")
        return None

def get_params(filename, df_meta, is_anomaly=False):
    clean_filename = filename.replace('.csv', '')
    row = pd.DataFrame() # DataFrame vacío por defecto

    if is_anomaly:
        # Intenta buscar TIC o TOI
        match = re.search(r'(?:TIC|TOI)[_]?(\d+)', clean_filename, re.IGNORECASE)
        if match:
            obj_id = int(match.group(1))
            col_id = next((c for c in df_meta.columns if 'tic' in c or 'tid' in c or 'toi' in c), None)
            if col_id:
                row = df_meta[df_meta[col_id] == obj_id]

        # Si falló la expresión regular, busca por el nombre limpio
        if row.empty:
            clean_target = re.sub(r'[\s_\-]', '', clean_filename).lower()
            if 'clean_name' in df_meta.columns:
                row = df_meta[df_meta['clean_name'] == clean_target]
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

# --- FUNCIÓN PRINCIPAL DE GENERACIÓN ---
def generate_visual_examples():
    print("Generando figura de ejemplos visuales...")
    os.makedirs(DIR_RESULTS, exist_ok=True)
    processor = DataPreprocessor()
    df_planets = load_metadata_planets()
    df_anomalies = load_metadata_anomalies()

    examples = {}

    # 1. BUSCAR PLANETA CONFIRMADO Y GENERAR ERROR DE PLEGADO
    files_planets = glob.glob(os.path.join(PATH_RAW_PLANETS, "*.csv"))
    if files_planets and df_planets is not None:
        for filepath in files_planets:
            P, T0 = get_params(os.path.basename(filepath), df_planets) or (None, None)
            if not P: continue
            try:
                df = pd.read_csv(filepath, comment='#')
                time = df.iloc[:, 1].values
                flux = df.iloc[:, 0].values

                median_val = np.nanmedian(flux)
                if median_val > 0:
                    depth = 1.0 - (np.min(flux) / median_val)
                    if depth < 0.005: continue

                curve_normal = processor.process_curve_phase_folding(time, flux, P, T0, smooth=False)
                P_bad = P * (1 + KP_EXTREME)
                T0_bad = T0 * (1 + KT_EXTREME)
                curve_error = processor.process_curve_phase_folding(time, flux, P_bad, T0_bad, smooth=False)

                if curve_normal is not None and curve_error is not None:
                    examples['Planeta Confirmado'] = curve_normal
                    examples['Error de Plegado'] = curve_error
                    print(f"   -> Ejemplo Planeta/Error obtenido de: {os.path.basename(filepath)}")
                    break
            except: continue

    # 2. GENERAR RUIDO PLANO (Sintético)
    noise = np.random.normal(0, 0.005, 2048)
    flat_curve = 0.5 + noise
    flat_curve = np.clip(flat_curve, 0, 1)
    examples['Ruido Plano'] = flat_curve
    print("   -> Ejemplo Ruido Plano generado.")

    # 3. BUSCAR FALSO POSITIVO REAL
    files_anoms = glob.glob(os.path.join(PATH_RAW_ANOMALIES, "*.csv"))
    if not files_anoms:
        print("❌ No se encontraron archivos de anomalías en la ruta especificada.")
    elif df_anomalies is not None:
        for filepath in files_anoms:
            filename = os.path.basename(filepath)
            P, T0 = get_params(filename, df_anomalies, is_anomaly=True) or (None, None)
            if not P:
                continue # No encontró metadatos
            try:
                # SE AGREGÓ comment='#' AQUÍ PARA EVITAR ERRORES DE LECTURA
                df = pd.read_csv(filepath, comment='#')
                df.columns = [c.strip() for c in df.columns] # Limpiar espacios en columnas

                time = df['timecorr'].values if 'timecorr' in df.columns else df.iloc[:, 0].values
                flux = df['pdcsap_flux'].values if 'pdcsap_flux' in df.columns else df.iloc[:, 1].values

                curve_anom = processor.process_curve_phase_folding(time, flux, P, T0, smooth=False)

                # Validar que devolvió algo válido y que no sea una línea totalmente plana
                if curve_anom is not None and np.std(curve_anom) > 0.001:
                    examples['Falso Positivo Real'] = curve_anom
                    print(f"   -> Ejemplo Falso Positivo obtenido de: {filename}")
                    break
            except Exception as e:
                print(f"   -> Error procesando anomalía {filename}: {e}")
                continue

    # --- GRAFICACIÓN ---
    for k in ['Planeta Confirmado', 'Error de Plegado', 'Ruido Plano', 'Falso Positivo Real']:
        if k not in examples:
            print(f"⚠️ Advertencia: No se pudo generar {k}. Se usará una línea base en 0.5.")
            examples[k] = np.full(2048, 0.5) # Reemplazado zeros por full(0.5) para que no sea 0.0

    x_axis = np.linspace(-0.5, 0.5, 2048)

    plots_config = [
        ('Planeta Confirmado', 'green', 'Clase 0 (Normal)', 'Tránsito alineado en el centro', 'ejemplo_planeta_confirmado.png'),
        ('Ruido Plano', 'blue', 'Clase 0 (Normal)', 'Variabilidad estelar sin eventos', 'ejemplo_ruido_plano.png'),
        ('Error de Plegado', 'red', 'Clase 1 (Anómalo)', 'Desfase por periodo incorrecto', 'ejemplo_error_plegado.png'),
        ('Falso Positivo Real', 'orange', 'Clase 1 (Anómalo)', 'Forma similar a tránsito (ej. Binaria)', 'ejemplo_falso_positivo.png')
    ]

    print("\nGenerando gráficos individuales...")

    for title, color, label_class, desc, filename in plots_config:
        plt.figure(figsize=(8, 6))
        data = examples[title]
        plt.plot(x_axis, data, color=color, alpha=0.8, linewidth=1.5)
        plt.title(f"{title}\n{label_class}", fontsize=14, fontweight='bold')

        plt.text(0.5, 0.05, desc, transform=plt.gca().transAxes, ha='center', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))

        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.5, 0.5)
        plt.xlabel("Fase Orbital", fontsize=12)
        plt.ylabel("Flujo Normalizado", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')

        save_path = os.path.join(DIR_RESULTS, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   -> Guardado: {filename}")

    print(f"\n✅ Gráficos guardados exitosamente en:\n{DIR_RESULTS}")

if __name__ == "__main__":
    generate_visual_examples()
