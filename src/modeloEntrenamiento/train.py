import tensorflow as tf
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


from model_autoencoder import build_autoencoder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'entrenamiento')
PATH_MODELS = os.path.join(BASE_DIR, 'models')


MODEL_FILE = os.path.join(PATH_MODELS, 'CAE_1D.keras')

def weighted_mae(y_true, y_pred):
    """
    Función de pérdida personalizada que castiga más los errores en los tránsitos
    Si el valor real (y_true) es menor a 0.49 (hay un pozo), el peso es 20x.
    """
    error = tf.abs(y_true - y_pred)

    weights = tf.where(y_true < 0.49, 20.0, 1.0)
    return tf.reduce_mean(error * weights)

def train_model():
    print("🚀 Iniciando Entrenamiento (Configuración Tesis Fase 2)...")


    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
        print("🗑️ Modelo antiguo eliminado para reiniciar entrenamiento limpio.")

    if not os.path.exists(PATH_MODELS): os.makedirs(PATH_MODELS)


    print("📂 Cargando dataset...")
    try:
        X_input = np.load(os.path.join(PATH_DATA, 'X_input.npy'))
        X_target = np.load(os.path.join(PATH_DATA, 'X_target.npy'))
        print(f"   Datos cargados: {X_input.shape}")
        print(f"   Rango Input:  [{np.min(X_input):.3f}, {np.max(X_input):.3f}]")
        print(f"   Rango Target: [{np.min(X_target):.3f}, {np.max(X_target):.3f}]")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return


    X_train, X_val, y_train, y_val = train_test_split(
        X_input, X_target,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )


    model = build_autoencoder(input_len=2048)

    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=weighted_mae)


    callbacks = [

        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),


        ModelCheckpoint(MODEL_FILE, monitor='val_loss', save_best_only=True, verbose=1),


        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]


    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history)

def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(BASE_DIR, 'results', 'training_history.png'))
    print("\n📈 Gráfico guardado en results/training_history.png")

if __name__ == "__main__":
    train_model()
