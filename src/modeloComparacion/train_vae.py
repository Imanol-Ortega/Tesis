import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from model_vae import build_vae

# Rutas ajustadas a tu estructura
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'entrenamiento')
PATH_MODELS = os.path.join(BASE_DIR, 'models')

def train_vae_reference():
    print("🚀 Entrenando VAE de Referencia (Hönes et al.)...")

    if not os.path.exists(PATH_MODELS): os.makedirs(PATH_MODELS)

    # 1. Cargar Datos LIMPIOS (X_target)
    # Tu tesis dice: "exceptuando los datos aumentados".
    # Usamos X_target (la versión perfecta) como Input Y Output.
    try:
        X_clean = np.load(os.path.join(PATH_DATA, 'X_target.npy'))
        print(f"   Datos cargados (Limpios/Target): {X_clean.shape}")
    except Exception as e:
        print(f"❌ Error cargando X_target.npy: {e}")
        return

    # 2. Split Train/Val
    X_train, X_val = train_test_split(X_clean, test_size=0.2, random_state=42)

    # 3. Construir y Compilar VAE
    vae = build_vae(input_len=2048)
    optimizer = Adam(learning_rate=0.0001)

    # La loss ya está añadida dentro del modelo (add_loss), así que usamos loss=None
    vae.compile(optimizer=optimizer)

    # 4. Entrenar (Input=Clean, Output=Clean implícito en la loss interna)
    history = vae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=100, # Los VAE suelen converger rápido con datos limpios
        batch_size=32,
        verbose=1
    )

    # 5. Guardar Modelo
    save_path = os.path.join(PATH_MODELS, 'VAE_Reference.keras')
    vae.save(save_path)
    print(f"✅ VAE de Referencia guardado en: {save_path}")

    # 6. Gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Entrenamiento VAE (Referencia Hönes)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE + KL)')
    plt.legend()
    plt.grid()
    output_plot = os.path.join(BASE_DIR, 'results', 'vae_training_history.png')
    plt.savefig(output_plot)
    print(f"📈 Gráfico guardado en {output_plot}")

if __name__ == "__main__":
    train_vae_reference()
