import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.Procesamiento.preprocessor import DataPreprocessor

# ================= CONFIGURACIÓN DE RUTAS =================
FILE_NEA = os.path.join('data', 'nea', 'confirmados', 'nasa_exoplanets.csv')
DIR_MAST = os.path.join('data', 'mast', 'csv')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANETA_BUSCADO = "BD-14 3065 b"

def visualizar_por_nombre(nombre_planeta):
    if not os.path.exists(FILE_NEA):
        print(f"Error: No se encuentra el archivo NEA en: {FILE_NEA}")
        return

    # Leer el archivo NEA saltando comentarios
    try:
        df_nea = pd.read_csv(FILE_NEA, comment='#')
        df_nea.columns = df_nea.columns.str.strip()
    except Exception as e:
        print(f"Error al leer el archivo NEA: {e}")
        return

    # Buscar la fila del planeta
    planeta_info = df_nea[df_nea['pl_name'] == nombre_planeta]

    if planeta_info.empty:
        print(f"Error: El planeta '{nombre_planeta}' no se encuentra en el registro.")
        return

    # Extraer parámetros para el preprocesamiento
    periodo = planeta_info.iloc[0]['pl_orbper']
    t0 = planeta_info.iloc[0]['pl_tranmid']

    print(f"Planeta encontrado: {nombre_planeta}")
    print(f"Parámetros: P={periodo}, T0={t0}")

    # --- CORRECCIÓN DE RUTA DE ARCHIVO ---
    # Reemplazamos espacios por guiones bajos para coincidir con tu carpeta MAST
    nombre_archivo_csv = nombre_planeta.replace(" ", "_") + ".csv"
    archivo_curva = os.path.join(DIR_MAST, nombre_archivo_csv)

    if not os.path.exists(archivo_curva):
        print(f"Error: No se encontró la curva de luz en: {archivo_curva}")
        # Listar archivos para ayudar a depurar
        print("Archivos disponibles en la carpeta (primeros 3):", os.listdir(DIR_MAST)[:3])
        return

    # 4. Leer datos de la curva de luz
    print(f"Leyendo archivo: {nombre_archivo_csv}")
    df_curva = pd.read_csv(archivo_curva, comment='#')

    # Detectar columnas de tiempo y flujo
    time_col = 'time' if 'time' in df_curva.columns else 'timecorr'
    flux_col = 'pdcsap_flux' if 'pdcsap_flux' in df_curva.columns else 'flux'

    time_data = df_curva[time_col].values
    flux_data = df_curva[flux_col].values

    # 5. Aplicar preprocesamiento de tu clase
    pp = DataPreprocessor()
    flux_noisy = pp.process_curve_phase_folding(time_data, flux_data, periodo, t0, smooth=False)
    flux_clean = pp.process_curve_phase_folding(time_data, flux_data, periodo, t0, smooth=True)

    if flux_noisy is None or flux_clean is None:
        print("Error: El preprocesador devolvió None. Verifica si hay NaNs o valores en 0.")
        return

    # 6. Graficación
    x_axis = np.linspace(-0.5, 0.5, 2048)
    plt.figure(figsize=(12, 6))

    plt.plot(x_axis, flux_noisy, color='black', alpha=0.3, label='Entrada Ruidosa ($X_{input}$)')
    plt.plot(x_axis, flux_clean, color='red', linewidth=2, label='Objetivo Limpio ($X_{target}$)')

    plt.title(f"Preprocesamiento: {nombre_planeta}", fontsize=14)
    plt.xlabel("Fase Orbital (Centrada en 0.0)")
    plt.ylabel("Amplitud [0, 1] (x15 + 0.5)")
    plt.xlim(-0.15, 0.15)
    plt.ylim(-0.1, 1.1)
    plt.axhline(0.5, color='blue', linestyle='--', alpha=0.3, label='Nivel Base')
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    # Guardar con el nombre del planeta
    plots_dir = os.path.join(BASE_DIR, 'graficos', 'preprocesamiento')
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f"prepro_{nombre_archivo_csv.replace('.csv', '.png')}")
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    visualizar_por_nombre(PLANETA_BUSCADO)
