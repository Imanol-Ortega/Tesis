import numpy as np
import tensorflow as tf
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score
import matplotlib.pyplot as plt
def weighted_mae(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    weights = tf.where(y_true < 0.49, 20.0, 1.0)
    return tf.reduce_mean(error * weights)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_TEST_X = os.path.join(BASE_DIR, 'data', 'processed', 'test', 'X_test.npy')
PATH_TEST_Y = os.path.join(BASE_DIR, 'data', 'processed', 'test', 'y_test.npy')
PATH_CAE = os.path.join(BASE_DIR, 'models', 'CAE_1D.keras')
PATH_VAE = os.path.join(BASE_DIR, 'models', 'VAE_Reference.keras')
DIR_RESULTS = os.path.join(BASE_DIR, 'results')
def get_best_metrics(y_true, mse_scores):
    """Calcula umbral óptimo F1 y devuelve métricas"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, mse_scores)
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    y_pred = (mse_scores > best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    final_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    final_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {
        'F1-Score': best_f1,
        'Precision': final_prec,
        'Recall': final_rec,
        'Threshold': best_thresh
    }
def compare():
    print("INICIANDO ANÁLISIS COMPARATIVO (FASE 4)...")
    if not os.path.exists(PATH_TEST_X):
        print("❌ No encuentro X_test.npy. ¿Ejecutaste la fase 3?")
        return
    X_test = np.load(PATH_TEST_X)
    y_test = np.load(PATH_TEST_Y)
    print(f"Datos de prueba cargados: {X_test.shape}")
    print("🔹 Cargando CAE (Propuesto)...")
    try:
        cae = tf.keras.models.load_model(PATH_CAE, custom_objects={'weighted_mae': weighted_mae})
    except:
        print("Advertencia: Cargando CAE sin compilar (para evitar error de loss custom)...")
        cae = tf.keras.models.load_model(PATH_CAE, compile=False)
    print("🔸 Cargando VAE (Referencia Hönes)...")
    class Sampling(tf.keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            return z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(tf.shape(z_mean))
    class VAELossLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            true_inputs, reconstruction, _, _ = inputs
            return reconstruction
    vae = tf.keras.models.load_model(
        PATH_VAE,
        custom_objects={'Sampling': Sampling, 'VAELossLayer': VAELossLayer},
        compile=False
    )
    print("Generando predicciones...")
    rec_cae = cae.predict(X_test, verbose=0)
    mse_cae = np.mean(np.square(X_test - rec_cae), axis=(1, 2))
    metrics_cae = get_best_metrics(y_test, mse_cae)
    rec_vae = vae.predict(X_test, verbose=0)
    mse_vae = np.mean(np.square(X_test - rec_vae), axis=(1, 2))
    metrics_vae = get_best_metrics(y_test, mse_vae)
    df = pd.DataFrame({
        'Métrica': ['F1-Score', 'Precisión', 'Recall (Sensibilidad)', 'Umbral MSE'],
        'CAE (Propuesto)': [
            f"{metrics_cae['F1-Score']:.4f}",
            f"{metrics_cae['Precision']:.4f}",
            f"{metrics_cae['Recall']:.4f}",
            f"{metrics_cae['Threshold']:.6f}"
        ],
        'VAE (Referencia)': [
            f"{metrics_vae['F1-Score']:.4f}",
            f"{metrics_vae['Precision']:.4f}",
            f"{metrics_vae['Recall']:.4f}",
            f"{metrics_vae['Threshold']:.6f}"
        ]
    })
    print("\nTABLA COMPARATIVA FINAL")
    print("=============================================")
    print(df.to_string(index=False))
    print("=============================================")
    csv_path = os.path.join(DIR_RESULTS, 'comparative_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"   -> Guardado en {csv_path}")
if __name__ == "__main__":
    compare()
