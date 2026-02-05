import pandas as pd
import lightkurve as lk
import os
import warnings
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANOMALIES_CSV = os.path.join(BASE_DIR, 'data', 'nea', 'anomalias', 'TOI_2026.01.18_12.38.06.csv')

OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'mast', 'anomalies_csv')
FITS_DIR = os.path.join(BASE_DIR, 'data', 'mast', 'fits')

MAX_WORKERS = 1
def procesar_anomalia(row):
    try:

        col_tic = next((c for c in row.index if 'tic' in c.lower() or 'tid' == c.lower()), None)
        if not col_tic: return "❌ Error: No encontré columna TIC ID"
        raw_id = str(row[col_tic])

        clean_id_str = re.sub(r"[^0-9]", "", raw_id)
        if not clean_id_str: return None
        tic_id = int(clean_id_str)

        safe_name = f"TIC_{tic_id}_FP"
        csv_output = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")

        if os.path.exists(csv_output):
            return f"⏭️  Salteado: {safe_name}"

        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120)

        if len(search) == 0:
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
        if len(search) == 0:
            return None

        try:
            lc_collection = search[0].download(download_dir=FITS_DIR, quality_bitmask='hard')
        except Exception as e:
            return f"⚠️ Error descarga: {safe_name}"
        if lc_collection is None: return None

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
        return f"✅ {safe_name} ({csv_size:.2f} MB)"
    except Exception as e:
        return f"🔥 Error en TIC {row.get(col_tic, '?')}: {str(e)}"
def descargar_anomalias_full():
    print(f"\n>>> [ADQUISICIÓN DE ANOMALÍAS] Buscando impostores... Workers: {MAX_WORKERS}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FITS_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")
    try:

        if not os.path.exists(ANOMALIES_CSV):
            print(f"❌ ERROR CRÍTICO: No encuentro el archivo: {ANOMALIES_CSV}")
            print("   -> Asegúrate de guardarlo en 'data/nea/' con el nombre 'nasa_anomalies.csv'")
            return
        df = pd.read_csv(ANOMALIES_CSV, comment='#')
        print(f"--- Archivo cargado. Total de candidatos en lista: {len(df)} ---")

        col_disp = next((c for c in df.columns if 'disp' in c.lower() or 'tfop' in c.lower()), None)
        if col_disp:

            df_filtrado = df[df[col_disp].astype(str).str.contains('FP|EB', na=False, regex=True)]
            print(f"--- Filtrando por FP/EB: Descargaremos {len(df_filtrado)} curvas anómalas ---")
            df = df_filtrado
        else:
            print("⚠️ Advertencia: No encontré columna de disposición. Descargando TODO el archivo.")
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        return

    rows = [row for _, row in df.iterrows()]
    print("\n--- INICIANDO DESCARGAS ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(procesar_anomalia, row): row for row in rows}

        for future in tqdm(as_completed(futures), total=len(rows), unit="anomalia", ncols=100):
            result = future.result()
            if result:
                tqdm.write(str(result))
    print("\n>>> Descarga de anomalías finalizada.")
if __name__ == "__main__":
    descargar_anomalias_full()
