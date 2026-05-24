import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt
class DataPreprocessor:
    def __init__(self):
        self.INPUT_LEN = 2048
    def process_curve_phase_folding(self, time, flux, P, T0, smooth=False):
        try:
            median_val = np.nanmedian(flux) #Calcula la mediana
            if median_val == 0 or np.isnan(median_val): return None #Evita la división por cero o por un valor no numérico
            flux_norm = flux / median_val #Normalizización

            phase = ((time - T0) / P) % 1 #Plegado de Fase
            phase[phase > 0.5] -= 1 #Corrección de Fase

            #Ordenamiento
            sort_idx = np.argsort(phase) #Obtiene los indices ordenados de la fase
            phase_sorted = phase[sort_idx] #Ordena el vector de fase
            flux_sorted = flux_norm[sort_idx] #Reordena el flujo igual que la fase
            #Limpieza de Datos
            flux_sorted = np.nan_to_num(flux_sorted, nan=1.0) #Reemplaza los NaN por 1.0

            if smooth: #Suavizado aplicado solo a la curva objetivo
                flux_sorted = medfilt(flux_sorted, kernel_size=51)

            #Interpolación
            #Elimina duplicados para evitar problemas en la interpolación
            phase_unique, unique_indices = np.unique(phase_sorted, return_index=True)
            flux_unique = flux_sorted[unique_indices]
            #Interpolación lineal de 2048 puntos (INPUT_LEN = 2048) y rango de fase de -0.5 a 0.5
            x_new = np.linspace(-0.5, 0.5, self.INPUT_LEN)
            #Proyecta los datos originales a la nueva fase utilizando interpolación lineal, rellenando con 1.0 donde no hay datos
            f_interp = interp1d(phase_unique, flux_unique, kind='linear', bounds_error=False, fill_value=1.0)
            flux_interpolated = f_interp(x_new)
            #Transformación del flujo para resaltar las características de los tránsitos
            flux_final = np.nan_to_num(flux_interpolated, nan=1.0) #Reemplaza cualquier NaN resultante de la interpolación por 1.0
            flux_final = flux_final - 1.0 #Centra el flujo alrededor de 0
            flux_final = flux_final * 15.0 #Amplifica las caídas para mejorar la visibilidad de los tránsitos
            flux_final = flux_final + 0.5  # Desplaza el flujo para que el rango típico de los tránsitos esté alrededor de 0.5
            flux_final = np.clip(flux_final, 0.0, 1.0) # Limita el flujo final al rango [0,1]
            return flux_final

        except:

            return None
