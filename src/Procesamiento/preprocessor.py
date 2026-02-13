import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

class DataPreprocessor:
    def __init__(self):
        self.INPUT_LEN = 2048

    def process_curve_phase_folding(self, time, flux, P, T0, smooth=False):
        try:

            median_val = np.nanmedian(flux)
            if median_val == 0 or np.isnan(median_val): return None
            flux_norm = flux / median_val


            phase = ((time - T0) / P) % 1
            phase[phase > 0.5] -= 1


            sort_idx = np.argsort(phase)
            phase_sorted = phase[sort_idx]
            flux_sorted = flux_norm[sort_idx]


            flux_sorted = np.nan_to_num(flux_sorted, nan=1.0)


            if smooth:

                flux_sorted = medfilt(flux_sorted, kernel_size=5)


            phase_unique, unique_indices = np.unique(phase_sorted, return_index=True)
            flux_unique = flux_sorted[unique_indices]

            x_new = np.linspace(-0.5, 0.5, self.INPUT_LEN)
            f_interp = interp1d(phase_unique, flux_unique, kind='linear', bounds_error=False, fill_value=1.0)
            flux_interpolated = f_interp(x_new)
            flux_final = np.nan_to_num(flux_interpolated, nan=1.0)


            flux_final = flux_final - 1.0


            flux_final = flux_final * 15.0


            flux_final = flux_final + 0.5


            flux_final = np.clip(flux_final, 0.0, 1.0)

            return flux_final

        except:
            return None
