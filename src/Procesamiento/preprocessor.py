import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

class DataPreprocessor:
    def __init__(self):
        self.INPUT_LEN = 2048

    def process_curve_phase_folding(self, time, flux, P, T0, smooth=False):
        try:
            # 1. Normalización (Estrella = 1.0)
            median_val = np.nanmedian(flux)
            if median_val == 0 or np.isnan(median_val): return None
            flux_norm = flux / median_val

            # 2. Plegado
            phase = ((time - T0) / P) % 1
            phase[phase > 0.5] -= 1

            # 3. Ordenar
            sort_idx = np.argsort(phase)
            phase_sorted = phase[sort_idx]
            flux_sorted = flux_norm[sort_idx]

            # Limpiar NaNs antes de filtrar para evitar propagación de errores
            flux_sorted = np.nan_to_num(flux_sorted, nan=1.0)

            # 4. LIMPIEZA DE TARGET (Solo si smooth=True)
            if smooth:
                # Filtro fuerte para dejar la curva verde perfecta
                # REDUCIDO: 5 es el mínimo para quitar ruido 'salt & pepper' sin borrar tránsitos finos
                flux_sorted = medfilt(flux_sorted, kernel_size=5)

            # 5. Interpolación
            phase_unique, unique_indices = np.unique(phase_sorted, return_index=True)
            flux_unique = flux_sorted[unique_indices]

            x_new = np.linspace(-0.5, 0.5, self.INPUT_LEN)
            f_interp = interp1d(phase_unique, flux_unique, kind='linear', bounds_error=False, fill_value=1.0)
            flux_interpolated = f_interp(x_new)
            flux_final = np.nan_to_num(flux_interpolated, nan=1.0)

            # --- ESTRATEGIA MAESTRA: CENTRADO EN 0.5 ---
            # La tesis pide rango [0,1]. La Sigmoid funciona mejor en 0.5.
            # 1. Restamos 1.0 -> Estrella = 0.0
            flux_final = flux_final - 1.0

            # 2. Multiplicamos por 15 (Amplificar señal x15)
            # Un tránsito de 0.01 (1%) pasa a ser 0.15 (Muy visible para la red)
            flux_final = flux_final * 15.0

            # 3. Sumamos 0.5 -> Estrella = 0.5
            flux_final = flux_final + 0.5

            # 4. Clip final [0, 1]
            flux_final = np.clip(flux_final, 0.0, 1.0)

            return flux_final

        except:
            return None
