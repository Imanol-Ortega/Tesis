import pandas as pd
import lightkurve as lk
import os
import warnings
import glob
import re
from tqdm import tqdm
# -- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEA_DIR = os.path.join(BASE_DIR, 'data', 'nea', 'confirmados')
CSV_DIR = os.path.join(BASE_DIR, 'data', 'mast', 'pl_confirmados_csv')
def obtener_catalogo_nea():
    archivos = glob.glob(os.path.join(NEA_DIR, "*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No se encontró el catálogo en {NEA_DIR}")
    return archivos[0]
def procesar_planeta(row):
    try:
        col_tic = next((c for c in row.index if 'tic' in c and 'id' in c), None)
        raw_id = str(row[col_tic])
        clean_id_str = re.sub(r"[^0-9]", "", raw_id)

        if not clean_id_str: return None
        tic_id = int(clean_id_str)
        planet_name = row['pl_name']
        safe_name = str(planet_name).replace(" ", "_").replace("/", "-")
        csv_output = os.path.join(CSV_DIR, f"{safe_name}.csv")
        if os.path.exists(csv_output):
            return f"Salteado: {planet_name}"
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120)
        if len(search) == 0:
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
        if len(search) == 0:
            return None
        try:
            lc_collection = search[0].download(quality_bitmask='hard')
        except Exception as e:
            return f"Error descarga: {planet_name} ({str(e)})"
        if lc_collection is None:
            return f"Error descarga (Vacío): {planet_name}"
        fits_path = lc_collection.filename
        lc = lc_collection.remove_nans()
        lc_df = pd.DataFrame({
            'pdcsap_flux': lc.pdcsap_flux.value,
            'timecorr': lc.time.value,
            'quality': lc.quality.value
        })
        lc_df.to_csv(csv_output, index=False)
        csv_size = os.path.getsize(csv_output) / (1024 * 1024)
        try:
            if os.path.exists(fits_path):
                os.remove(fits_path)
        except:
            pass
        return f"{planet_name} ({csv_size:.2f} MB)"
    except Exception as e:
        return f"🔥 Error crítico en {row.get('pl_name', 'Planeta')}: {str(e)}"
def descargar_curvas():
    print(f"\n>>> [ADQUISICIÓN DE DATOS SECUENCIAL]")
    os.makedirs(CSV_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")
    try:
        csv_path = obtener_catalogo_nea()
        df = pd.read_csv(csv_path, comment='#')
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        col_tic = next((c for c in df.columns if 'tic' in c and 'id' in c), None)
        df = df.dropna(subset=[col_tic, 'pl_name'])
        print(f"--- PROCESANDO: {len(df)} planetas ---")
    except Exception as e:
        print(f"Error inicializando: {e}")
        return
    rows = [row for _, row in df.iterrows()]
    print("\n--- INICIANDO DESCARGAS ---")
    for row in tqdm(rows, unit="planeta", ncols=100):
        result = procesar_planeta(row)
        if result:
            tqdm.write(str(result))
    print("\n>>> Descarga masiva finalizada.")
if __name__ == "__main__":
    descargar_curvas()
