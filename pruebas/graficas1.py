import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import glob

# ================= CONFIGURACIÓN =================
PATH_METADATA = os.path.join('data', 'nea', 'confirmados', 'nasa_exoplanets.csv')
PATH_LIGHTCURVES = os.path.join('data', 'mast', 'csv', '*.csv')

def get_metadata(planet_name, df_meta):
    try:
        target = planet_name.replace(' ', '').lower()
        df_meta['temp_name'] = df_meta['pl_name'].str.replace(' ', '').str.lower()
        row = df_meta[df_meta['temp_name'] == target].iloc[0]
        return row['pl_orbper'], row['pl_tranmid']
    except:
        return None, None

def process_and_plot(file_path, df_meta):
    df = pd.read_csv(file_path)
    col_time = [c for c in df.columns if 'time' in c.lower()][0]
    col_flux = [c for c in df.columns if 'flux' in c.lower()][0]

    time, flux = df[col_time].values, df[col_flux].values
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time, flux = time[mask], flux[mask]
    flux_norm = flux / np.nanmedian(flux)

    planet_name = os.path.basename(file_path).replace('.csv', '').replace('_', ' ')
    P, T0 = get_metadata(planet_name, df_meta)

    if P is None: return False

    # --- RE-CENTRADO AUTOMÁTICO (Para que se vea como esperas) ---
    # Hacemos un plegado preliminar
    phase_tmp = ((time - T0) / P) % 1
    phase_tmp[phase_tmp > 0.5] -= 1

    # Buscamos el punto más bajo real (el tránsito) en una ventana
    # Esto corrige si el T0 del CSV de la NASA está un poco desfasado
    bins = np.linspace(-0.5, 0.5, 100)
    bin_means = pd.Series(flux_norm).groupby(pd.cut(phase_tmp, bins)).mean()
    t_shift = bins[np.argmin(bin_means.values)]

    # Aplicamos el desplazamiento para centrar la "U" en 0.0
    phase_final = phase_tmp - t_shift
    phase_final[phase_final > 0.5] -= 1
    phase_final[phase_final < -0.5] += 1

    # --- PROCESAMIENTO FINAL ---
    sort_idx = np.argsort(phase_final)
    p_sorted, f_sorted = phase_final[sort_idx], flux_norm[sort_idx]

    # Interpolación
    p_uni, u_idx = np.unique(p_sorted, return_index=True)
    f_uni = f_sorted[u_idx]
    x_new = np.linspace(-0.5, 0.5, 2048)
    f_interp = interp1d(p_uni, f_uni, kind='linear', fill_value=1.0, bounds_error=False)(x_new)

    f_final = np.clip((np.nan_to_num(f_interp, nan=1.0) - 1.0) * 15.0 + 0.5, 0.0, 1.0)

    # ================= GRÁFICO FINAL (EL "LINDO") =================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # A. Serie Temporal
    axes[0].scatter(time, flux_norm, s=1, color='black', alpha=0.3)
    axes[0].set_title(f"A. Datos Originales - {planet_name}", fontweight='bold')

    # B. Plegado (Ahora sí verás una sola línea o una U muy clara)
    axes[1].scatter(phase_final, flux_norm, s=2, color='#1f77b4', alpha=0.5)
    axes[1].set_title("B. Plegado de Fase", fontweight='bold')
    axes[1].set_xlim(-0.5, 0.5)

    # C. Tensor Final
    axes[2].plot(np.arange(2048), f_final, color='#d62728', linewidth=1.5)
    axes[2].axvline(1024, color='black', linestyle='--', alpha=0.3)
    axes[2].set_title("C. Vector de Entrada para la Red Neuronal", fontweight='bold')
    axes[2].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()
    return True

if __name__ == "__main__":
    df_meta = pd.read_csv(PATH_METADATA, comment='#')
    # Intenta buscar WASP-12 b o alguno que no sea AU Mic
    csv_files = glob.glob(PATH_LIGHTCURVES)
    for f in csv_files:
        if "AU_Mic" in f: continue # Saltamos el "feo"
        if process_and_plot(f, df_meta): break
