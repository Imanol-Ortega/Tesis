import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random

# --- RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'entrenamiento')
PATH_MODEL = os.path.join(BASE_DIR, 'models', 'CAE_1D.keras')
DIR_RESULTS = os.path.join(BASE_DIR, 'results')

def check_visual():
    print("🔍 Iniciando Inspección Visual del Modelo...")

    # 1. Cargar el Modelo
    if not os.path.exists(PATH_MODEL):
        print(f"❌ Error: No encuentro el modelo en {PATH_MODEL}")
        print("   ¿Ya ejecutaste train.py?")
        return

    try:
        # Usamos compile=False para evitar errores al buscar la función de pérdida personalizada (weighted_mae)
        model = tf.keras.models.load_model(PATH_MODEL, compile=False)
        print("✅ Modelo cargado exitosamente.")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return

    # 2. Cargar Datos (Solo necesitamos unos pocos para probar)
    try:
        X_input = np.load(os.path.join(PATH_DATA, 'X_input.npy'))
        X_target = np.load(os.path.join(PATH_DATA, 'X_target.npy'))
        print(f"📂 Datos disponibles: {X_input.shape[0]} muestras.")
    except:
        print("❌ No encuentro los archivos .npy en data/processed/entrenamiento")
        return

    # 3. Seleccionar Muestras Aleatorias
    # MODIFICACIÓN: Buscamos curvas que tengan tránsitos visibles (bajadas de luz)
    # La estrella está en 0.5. Si baja de 0.48, hay algo interesante.
    print("🔎 Buscando curvas con tránsitos claros para la prueba...")
    min_vals = np.min(X_target, axis=1)
    transit_indices = np.where(min_vals < 0.48)[0]

    num_samples = 5
    if len(transit_indices) >= num_samples:
        print(f"   ✅ Encontrados {len(transit_indices)} candidatos con tránsitos. Seleccionando {num_samples} al azar.")
        indices = random.sample(list(transit_indices), num_samples)
    else:
        print("⚠️ No se encontraron suficientes tránsitos profundos. Usando aleatorios.")
        indices = random.sample(range(len(X_input)), num_samples)

    samples_in = X_input[indices]
    samples_target = X_target[indices]

    # 4. PREDECIR (La Magia)
    print("🔮 Generando reconstrucciones...")
    predictions = model.predict(samples_in, verbose=0)

    # 5. Graficar
    if not os.path.exists(DIR_RESULTS): os.makedirs(DIR_RESULTS)

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 10))
    fig.suptitle(f'Verificación de Reconstrucción', fontsize=16)

    for i, idx in enumerate(indices):
        ax = axes[i]

        # Datos (quitamos la última dimensión para graficar: 2048, 1 -> 2048)
        input_curve = samples_in[i].flatten()
        target_curve = samples_target[i].flatten()
        pred_curve = predictions[i].flatten()

        # Eje X (Fase o Tiempo)
        x = np.linspace(-0.5, 0.5, len(input_curve))

        # Graficar
        ax.plot(x, input_curve, label='Input (Ruidoso)', color='salmon', alpha=0.5, linewidth=1)
        ax.plot(x, target_curve, label='Target (Perfecto)', color='green', linestyle='--', linewidth=1.5)
        ax.plot(x, pred_curve, label='Reconstrucción', color='blue', linewidth=2)

        ax.set_title(f"Muestra #{idx}", fontsize=10)
        ax.set_ylim(0.3, 0.7) # Zoom un poco más amplio para ver bien el pozo
        if i == 0: ax.legend(loc='upper right')

    plt.tight_layout()
    output_path = os.path.join(DIR_RESULTS, 'visual_check.png')
    plt.savefig(output_path)
    print(f"\n📸 Gráfica guardada en: {output_path}")
    print("Ábrela para ver si la línea AZUL coincide con la VERDE.")
    plt.show()

if __name__ == "__main__":
    check_visual()
