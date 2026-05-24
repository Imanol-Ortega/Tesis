import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.Procesamiento.preprocessor import DataPreprocessor

# ================= CONFIGURACIÓN =================
FILE_NEA = os.path.join('data', 'nea', 'confirmados', 'nasa_exoplanets.csv')
DIR_MAST = os.path.join('data', 'mast', 'csv')

def buscar_mejor_u():
    pp = DataPreprocessor()
    df_nea = pd.read_csv(FILE_NEA, comment='#')
    df_nea.columns = df_nea.columns.str.strip()

    mejor_planeta = None
    max_profundidad = 1.0 # Buscamos el valor más bajo (cerca de 0)
    datos_ganadores = {}

    archivos = [f for f in os.listdir(DIR_MAST) if f.endswith('.csv')]
    print(f"Escaneando {len(archivos)} archivos para encontrar una 'U' visible...")

    for archivo in archivos:
        nombre_planeta = archivo.replace("_", " ").replace(".csv", "")
        planeta_info = df_nea[df_nea['pl_name'] == nombre_planeta]
        if planeta_info.empty: continue

        # Leer y procesar
        try:
            df = pd.read_csv(os.path.join(DIR_MAST, archivo))
            col_t = next(c for c in df.columns if 'time' in c.lower())
            col_f = next(c for c in df.columns if 'flux' in c.lower())

            # Procesamos con smooth=True (kernel 51 como dijiste)
            p = planeta_info.iloc[0]['pl_orbper']
            t0 = planeta_info.iloc[0]['pl_tranmid']

            x_target = pp.process_curve_phase_folding(df[col_t].values, df[col_f].values, p, t0, smooth=True)

            if x_target is None: continue

            # MÉTRICA: Buscamos el valor mínimo en el centro (fase -0.05 a 0.05)
            # Un buen tránsito bajará de 0.5 hacia 0.2 o 0.1
            centro = x_target[920:1128] # El centro de los 2048 puntos
            min_val = np.min(centro)

            if min_val < max_profundidad and min_val > 0.01: # Evitamos errores de 0 absoluto
                max_profundidad = min_val
                mejor_planeta = nombre_planeta
                datos_ganadores = {'x_input': pp.process_curve_phase_folding(df[col_t].values, df[col_f].values, p, t0, smooth=False),
                                   'x_target': x_target, 'name': nombre_planeta}
                print(f"Candidato: {nombre_planeta} con profundidad {min_val:.3f}")

        except:
            continue

    if mejor_planeta:
        graficar_resultado(datos_ganadores)
    else:
        print("No se encontró ningún tránsito claro.")

def graficar_resultado(d):
    x_axis = np.linspace(-0.5, 0.5, 2048)
    plt.figure(figsize=(12, 7))
    plt.plot(x_axis, d['x_input'], color='gray', alpha=0.3, label='Entrada Ruidosa')
    plt.plot(x_axis, d['x_target'], color='red', linewidth=3, label='Objetivo Limpio (U)')
    plt.title(f"Mejor Tránsito Encontrado: {d['name']}")
    plt.xlim(-0.15, 0.15)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.savefig("figura_tesis_ganadora.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    buscar_mejor_u()
