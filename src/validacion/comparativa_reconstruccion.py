import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'test')
DIR_RESULTS = os.path.join(BASE_DIR, 'results', 'comparativas')

# Intentar localizar el modelo en las rutas probables
PATH_MODEL_1 = os.path.join(BASE_DIR, 'models', 'CAE_1D.keras')
PATH_MODEL_2 = os.path.join(BASE_DIR, 'models_uniforme', 'CAE_1D.keras')
PATH_MODEL = PATH_MODEL_1 if os.path.exists(PATH_MODEL_1) else PATH_MODEL_2

# Definición de la función de pérdida personalizada (necesaria para cargar el modelo correctamente)
def weighted_mae(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    weights = tf.where(y_true < 0.48, 20.0, 1.0)
    return tf.reduce_mean(error * weights)

def main():
    print("--- GENERADOR DE COMPARATIVA DE RECONSTRUCCIÓN ---")
    os.makedirs(DIR_RESULTS, exist_ok=True)

    # 1. Cargar Datos
    try:
        X_test = np.load(os.path.join(PATH_DATA, 'X_test.npy'))
        y_test = np.load(os.path.join(PATH_DATA, 'y_test.npy'))
        print(f"✅ Datos cargados: {X_test.shape} muestras.")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        print("Asegúrate de haber ejecutado 'data_builder_model.py' primero.")
        return

    # 2. Cargar Modelo
    if not os.path.exists(PATH_MODEL):
        print(f"❌ No se encontró el modelo en: {PATH_MODEL}")
        return

    try:
        # compile=False es suficiente para inferencia, pero pasamos custom_objects por seguridad
        model = tf.keras.models.load_model(PATH_MODEL, custom_objects={'weighted_mae': weighted_mae}, compile=False)
        print(f"✅ Modelo cargado desde: {PATH_MODEL}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return

    # 3. Seleccionar Muestras (Una Normal y una Anómala si es posible)
    idx_normal = np.where(y_test == 0)[0]
    idx_anomalo = np.where(y_test == 1)[0]

    samples_to_plot = []

    if len(idx_normal) > 0:
        i = random.choice(idx_normal)
        samples_to_plot.append((X_test[i:i+1], f"Clase 0 (Normal) - Índice {i}"))

    if len(idx_anomalo) > 0:
        i = random.choice(idx_anomalo)
        samples_to_plot.append((X_test[i:i+1], f"Clase 1 (Anómalo) - Índice {i}"))

    # 4. Generar Reconstrucciones y Graficar
    x_axis = np.linspace(-0.5, 0.5, 2048)
    plt.figure(figsize=(12, 5 * len(samples_to_plot)))

    for i, (sample, label) in enumerate(samples_to_plot):
        # Predicción
        reconstruction = model.predict(sample, verbose=0)
        mse = np.mean(np.square(sample - reconstruction))

        # Aplanar para graficar
        orig = sample[0].flatten()
        recon = reconstruction[0].flatten()

        plt.subplot(len(samples_to_plot), 1, i+1)
        plt.plot(x_axis, orig, label='Original (Input)', color='black', alpha=0.5, linewidth=1)
        plt.plot(x_axis, recon, label='Reconstrucción (Output)', color='red', linewidth=2, linestyle='-')
        plt.title(f"{label} | MSE: {mse:.6f}")
        plt.xlabel("Fase Orbital")
        plt.ylabel("Flujo Normalizado")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

    save_path = os.path.join(DIR_RESULTS, 'comparativa_reconstruccion.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📸 Gráfico guardado en: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
