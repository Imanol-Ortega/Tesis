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

    try:
        X_clean = np.load(os.path.join(PATH_DATA, 'X_target.npy'))
        print(f"Datos cargados (Limpios/Target): {X_clean.shape}")
    except Exception as e:
        print(f"Error cargando X_target.npy: {e}")
        return


    X_train, X_val = train_test_split(X_clean, test_size=0.2, random_state=42)


    vae = build_vae(input_len=2048)
    optimizer = Adam(learning_rate=0.0001)


    vae.compile(optimizer=optimizer)


    history = vae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=32,
        verbose=1
    )


    save_path = os.path.join(PATH_MODELS, 'VAE_Reference.keras')
    vae.save(save_path)
    print(f"VAE de Referencia guardado en: {save_path}")


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
