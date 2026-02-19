import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from model_autoencoder import build_autoencoder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'entrenamiento')
PATH_MODELS = os.path.join(BASE_DIR, 'models')
MODEL_FILE = os.path.join(PATH_MODELS, 'CAE_1D.keras')

# Función de pérdida personalizada
def weighted_mae(y_true, y_pred):
    error = tf.abs(y_true - y_pred) #Error absoluto entre el valor real y el predicho
    weights = tf.where(y_true < 0.48, 20.0, 1.0) #Pesos: 20x para valores < 0.48, 1x para el resto
    return tf.reduce_mean(error * weights) # Promedio del error ponderado

def train_model():
    print("Iniciando Entrenamiento")
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
        print("Modelo antiguo eliminado para reiniciar entrenamiento limpio.")
    if not os.path.exists(PATH_MODELS): os.makedirs(PATH_MODELS)
    print("Cargando dataset...")
    try:
        X_input = np.load(os.path.join(PATH_DATA, 'X_input.npy'))
        X_target = np.load(os.path.join(PATH_DATA, 'X_target.npy'))
    except Exception as e:
        print(f" Error cargando datos: {e}")
        return
    # Dividir el dataset en entrenamiento y validación (80% - 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_input, X_target,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    model = build_autoencoder(input_len=2048) #Construcción del modelo
    optimizer = Adam(learning_rate=0.001) #Optimizador Adam con tasa de aprendizaje de 0.001
    model.compile(optimizer=optimizer, loss=weighted_mae) #Compilación del modelo con la función de pérdida personalizada
    #Callbacks para el entrenamiento
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1), #Detiene el entrenamiento si la pérdida de validación no mejora durante 30 épocas, restaurando los mejores pesos
        ModelCheckpoint(MODEL_FILE, monitor='val_loss', save_best_only=True, verbose=1), #Guarda el mejor modelo basado en la pérdida de validación
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1) #Reduce la tasa de aprendizaje a la mitad si la pérdida de validación no mejora durante 5 épocas, con un mínimo de 1e-6
    ]
    #Entrenar el modelo
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
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(BASE_DIR, 'results', 'training_history.png'))
    print("\n📈 Gráfico guardado en results/training_history.png")
if __name__ == "__main__":
    train_model()
