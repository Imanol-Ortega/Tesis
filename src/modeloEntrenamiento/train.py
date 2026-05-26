import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
import numpy as np
from src.modeloEntrenamiento.model_autoencoder import build_autoencoder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'entrenamiento')
PATH_MODELS = os.path.join(BASE_DIR, 'models')
PATH_MODELS = os.path.join(BASE_DIR, 'models_uniforme')
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
        #Cargar los datos de entrenamiento
        X_input = np.load(os.path.join(PATH_DATA, 'X_input.npy')) #
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
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1), #Detencion temprana del entrenamiento
        ModelCheckpoint(MODEL_FILE, monitor='val_loss', save_best_only=True, verbose=1), #Guarda el mejor modelo basado en la pérdida de validación
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1) #Reducción de la tasa de aprendizaje
    ]
    #Entrenar el modelo
    inicio = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    #Tiempo total de entrenamiento
    fin = time.time()
    tiempo_total = (fin - inicio) / 60
    plot_history(history,tiempo_total) #Gráfica Pérdida de entrenamiento y validación
def plot_history(history, tiempo_total):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    # Identificar la mejor época
    best_epoch = np.argmin(val_loss)
    best_val_loss = val_loss[best_epoch]
    best_epoch_num = epochs[best_epoch]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'y', label='Pérdida de entrenamiento', alpha=0.8)
    plt.plot(epochs, val_loss, 'r', label='Pérdida de validación', linewidth=2)
    # Marcador de la mejor época
    plt.scatter(best_epoch_num, best_val_loss, color='blue', s=100, zorder=5, label='Mejor Época')
    # 1. Colocamos la leyenda arriba a la derecha
    plt.legend(loc='upper right')
    # 2. CUADRO DE INFORMACIÓN DEBAJO DE LA LEYENDA
    info_text = (f'Mejor Época: {best_epoch_num}\n'
                 f'Mejor Loss: {best_val_loss:.4f}\n'
                 f'Tiempo Total: {tiempo_total:.2f} min')
    # Ajustamos y=0.75 para que esté debajo de la leyenda (que suele ocupar hasta el 0.85/0.90)
    # horizontalalignment='right' y verticalalignment='top' para que se alinee con la leyenda
    plt.gca().text(0.95, 0.75, info_text, transform=plt.gca().transAxes,
                   fontsize=10, fontweight='bold',
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
    plt.title('Pérdida de entrenamiento y validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.grid(True, linestyle='--', alpha=0.6)
    # Guardar gráfico
    output_dir = os.path.join(BASE_DIR, 'graficos', 'resultado_entrenamiento')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_path)
    plt.show()
if __name__ == "__main__":
    train_model()
