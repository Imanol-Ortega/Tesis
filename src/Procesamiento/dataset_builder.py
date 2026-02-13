import numpy as np
import pandas as pd
import os
import glob
import re
from preprocessor import DataPreprocessor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PATH_RAW = os.path.join(BASE_DIR, 'data', 'mast', 'csv')


PATH_META = os.path.join(BASE_DIR, 'data', 'nea', 'confirmados', 'nasa_exoplanets.csv')


DIR_OUTPUT = os.path.join(BASE_DIR, 'data', 'processed', 'entrenamiento')


KP = 0.001
KT = 0.0005
SAMPLES_PER_CURVE = 5

def load_metadata():
    """Carga el catálogo y crea una columna de búsqueda limpia"""
    if not os.path.exists(PATH_META):
        print(f"Error CRÍTICO: No encuentro {PATH_META}")
        print(f"   (Verifica que la carpeta 'nea' exista en 'data')")
        return None

    try:
        df = pd.read_csv(PATH_META, comment='#')

        df.columns = [c.strip().lower() for c in df.columns]


        if 'pl_name' in df.columns:
            df['clean_name'] = df['pl_name'].astype(str).apply(lambda x: re.sub(r'[\s_\-]', '', x).lower())

        print(f"Catálogo cargado correctamente: {len(df)} exoplanetas.")
        return df
    except Exception as e:
        print(f"Error leyendo metadatos: {e}")
        return None

def get_params_from_filename(filename, df_meta):
    """Busca P y T0 cruzando el nombre del archivo con el catálogo"""

    clean_filename = filename.replace('.csv', '')
    clean_target = re.sub(r'[\s_\-]', '', clean_filename).lower()


    row = df_meta[df_meta['clean_name'] == clean_target]

    if row.empty:
        return None


    try:
        P = float(row.iloc[0]['pl_orbper'])


        if 'pl_tranmid' in row.columns:
            T0 = float(row.iloc[0]['pl_tranmid'])
        elif 'pl_orbtper' in row.columns:
            T0 = float(row.iloc[0]['pl_orbtper'])
        else:
            return None

        return P, T0
    except:
        return None

def build_dataset():
    print("INICIANDO DATASET BUILDER (Tesis Fase 1)...")

    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)

    processor = DataPreprocessor()
    df_meta = load_metadata()
    if df_meta is None: return

    files = glob.glob(os.path.join(PATH_RAW, "*.csv"))
    print(f"Archivos encontrados: {len(files)}")

    X_input = []
    X_target = []

    processed_count = 0
    skipped_count = 0

    for filepath in files:
        filename = os.path.basename(filepath)


        params = get_params_from_filename(filename, df_meta)
        if params is None:
            skipped_count += 1
            continue

        P_real, T0_real = params
        if np.isnan(P_real) or np.isnan(T0_real) or P_real == 0:
            skipped_count += 1
            continue

        try:

            df_lc = pd.read_csv(filepath, comment='#')
            col_t = next((c for c in df_lc.columns if c.lower() in ['time', 'bjd', 'timecorr']), None)
            col_f = next((c for c in df_lc.columns if c.lower() in ['flux', 'pdcsap_flux', 'sap_flux']), None)

            if not col_t or not col_f:
                skipped_count += 1
                continue

            time = df_lc[col_t].values
            flux = df_lc[col_f].values


            curve_target = processor.process_curve_phase_folding(
                time, flux, P_real, T0_real, smooth=True
            )

            if curve_target is None:
                skipped_count += 1
                continue


            for _ in range(SAMPLES_PER_CURVE):
                sigma_p = P_real * KP
                sigma_t0 = T0_real * KT
                noise_p = np.random.normal(0, sigma_p)
                noise_t0 = np.random.normal(0, sigma_t0)

                curve_input = processor.process_curve_phase_folding(
                    time, flux, P_real + noise_p, T0_real + noise_t0, smooth=False
                )

                if curve_input is not None:
                    X_input.append(curve_input)
                    X_target.append(curve_target)

            processed_count += 1
            print(f"Procesados: {processed_count}...", end='\r')

        except Exception as e:
            skipped_count += 1
            continue

    print(f"\nFinalizado: {processed_count} procesados, {skipped_count} saltados.")

    if len(X_input) > 0:
        X_input = np.array(X_input)[..., np.newaxis]
        X_target = np.array(X_target)[..., np.newaxis]

        print(f"GUARDANDO DATOS:")
        print(f"   Inputs: {X_input.shape}")
        print(f"   Targets: {X_target.shape}")

        np.save(os.path.join(DIR_OUTPUT, "X_input.npy"), X_input)
        np.save(os.path.join(DIR_OUTPUT, "X_target.npy"), X_target)
    else:
        print("\nNo se generaron datos. Revisa tus archivos CSV.")

if __name__ == "__main__":
    build_dataset()
