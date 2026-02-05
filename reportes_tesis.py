import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix
from tensorflow.keras import layers, models, backend as K

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATA_TRAIN = os.path.join(BASE_DIR, 'data', 'processed', 'entrenamiento')
PATH_DATA_TEST_X = os.path.join(BASE_DIR, 'data', 'processed', 'test', 'X_test.npy')
PATH_DATA_TEST_Y = os.path.join(BASE_DIR, 'data', 'processed', 'test', 'y_test.npy')
PATH_RESULTS = os.path.join(BASE_DIR, 'results')

PATH_MODEL_CAE = os.path.join(BASE_DIR, 'models', 'CAE_1D.keras')
PATH_MODEL_VAE = os.path.join(BASE_DIR, 'models', 'VAE_Reference.keras')

if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)

# Estilo
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.3)

# ==========================================
# 2. OBJETOS PERSONALIZADOS (Para cargar modelos)
# ==========================================
def weighted_mae(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    weights = tf.where(y_true < 0.48, 20.0, 1.0)
    return tf.reduce_mean(error * weights)

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        return z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(tf.shape(z_mean))

class VAELossLayer(layers.Layer):
    def call(self, inputs):
        true_inputs, reconstruction, z_mean, z_log_var = inputs
        return reconstruction

# ==========================================
# 3. UTILIDADES
# ==========================================
def get_best_metrics(y_true, mse_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, mse_scores)
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    best_idx = np.argmax(f1_scores)

    y_pred = (mse_scores > thresholds[best_idx]).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'f1': f1_scores[best_idx], 'prec': precisions[best_idx], 'rec': recalls[best_idx],
        'thresh': thresholds[best_idx], 'cm': cm, 'curve_p': precisions, 'curve_r': recalls
    }

# ==========================================
# 4. CARGA
# ==========================================
print("📂 Cargando datos...")
X_input_train = np.load(os.path.join(PATH_DATA_TRAIN, 'X_input.npy'))
X_target_train = np.load(os.path.join(PATH_DATA_TRAIN, 'X_target.npy'))
X_test = np.load(PATH_DATA_TEST_X)
y_test = np.load(PATH_DATA_TEST_Y)

print("🤖 Cargando modelos...")
try:
    cae = tf.keras.models.load_model(PATH_MODEL_CAE, custom_objects={'weighted_mae': weighted_mae})
except:
    cae = tf.keras.models.load_model(PATH_MODEL_CAE, compile=False)

try:
    vae = tf.keras.models.load_model(PATH_MODEL_VAE, custom_objects={'Sampling': Sampling, 'VAELossLayer': VAELossLayer}, compile=False)
except:
    print("❌ Error cargando VAE.")
    exit()

# ==========================================
# 5. GENERACIÓN DE GRÁFICOS
# ==========================================

# --- GRÁFICO 1: DATOS (Búsqueda Inteligente) ---
print("📊 Generando Gráfico 1: Datos...")
best_idx = 0
found_nice = False
candidates = [2021, 3012, 1760, 2332]
for idx in candidates:
    if idx < len(X_target_train) and np.min(X_target_train[idx]) < 0.45:
        best_idx = idx
        found_nice = True
        break
if not found_nice:
    for i in range(len(X_target_train)):
        if np.min(X_target_train[i]) < 0.46 and np.std(X_target_train[i]) < 0.02:
            best_idx = i
            break

plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(X_target_train[best_idx], color='#2ca02c', linewidth=2, label='Target (Ideal)')
plt.title(f"A. Curva de Luz Original Plegada (Muestra #{best_idx})", fontweight='bold')
plt.ylabel("Flujo")
plt.ylim(0.3, 0.6)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(X_input_train[best_idx], color='#d62728', alpha=0.6, linewidth=0.8, label='Input (Ruidoso)')
plt.plot(X_target_train[best_idx], color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.8, label='Referencia')
plt.title("B. Curva de Luz con Ruido (Input)", fontweight='bold')
plt.xlabel("Fase")
plt.ylabel("Flujo")
plt.ylim(0.3, 0.6)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PATH_RESULTS, '1_datos_original_vs_aumentado.png'), dpi=300)
plt.close()


# --- GRÁFICO 2: RECONSTRUCCIÓN ---
print("📊 Generando Gráfico 2: Reconstrucción...")
indices_plot = [best_idx, 1760, 3012]
indices_plot = [i for i in indices_plot if i < len(X_input_train)]
if len(indices_plot) < 3: indices_plot = [best_idx, best_idx+1, best_idx+2]
samples = X_input_train[indices_plot]
targets = X_target_train[indices_plot]
reconstructions = cae.predict(samples, verbose=0)

plt.figure(figsize=(10, 9))
for i, idx in enumerate(indices_plot):
    plt.subplot(3, 1, i+1)
    plt.plot(samples[i], color='red', alpha=0.25, label='Input')
    plt.plot(targets[i], color='green', linestyle='--', linewidth=1.5, label='Target')
    plt.plot(reconstructions[i], color='blue', linewidth=2, label='CAE')
    plt.title(f"Reconstrucción Muestra #{idx}")
    plt.ylabel("Flujo")
    plt.ylim(0.3, 0.65)
    if i == 0: plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PATH_RESULTS, '2_reconstruccion_cae.png'), dpi=300)
plt.close()


# --- CÁLCULO MÉTRICAS ---
print("⚙️ Calculando métricas...")
rec_cae = cae.predict(X_test, verbose=0)
mse_cae = np.mean(np.square(X_test - rec_cae), axis=(1, 2))
metrics_cae = get_best_metrics(y_test, mse_cae)

rec_vae = vae.predict(X_test, verbose=0)
mse_vae = np.mean(np.square(X_test - rec_vae), axis=(1, 2))
metrics_vae = get_best_metrics(y_test, mse_vae)


# --- GRÁFICO 3: DETALLE CAE ---
print("📊 Generando Gráfico 3: Métricas CAE...")
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(metrics_cae['curve_r'], metrics_cae['curve_p'], label=f'CAE (F1={metrics_cae["f1"]:.3f})', linewidth=2)
plt.scatter(metrics_cae['rec'], metrics_cae['prec'], color='red', s=100, label='Punto Óptimo', zorder=5)
plt.title(f"Curva Precision-Recall (CAE)\nUmbral MSE: {metrics_cae['thresh']:.6f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.heatmap(metrics_cae['cm'], annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred: Normal', 'Pred: Anómalo'], yticklabels=['Real: Normal', 'Real: Anómalo'])
plt.title(f"Matriz de Confusión CAE\nF1-Score: {metrics_cae['f1']:.4f}")
plt.tight_layout()
plt.savefig(os.path.join(PATH_RESULTS, '3_metricas_cae_detalle.png'), dpi=300)
plt.close()


# --- GRÁFICO 4: COMPARATIVA (Con Tabla de Umbrales) ---
print("📊 Generando Gráfico 4: Comparativa Final (Con Tabla)...")
fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 3)

# SUBPLOT 1: TABLA COMPARATIVA (Reemplaza al gráfico de barras)
ax0 = fig.add_subplot(gs[0, 0])
ax0.axis('off') # Ocultar ejes
ax0.set_title("Tabla Comparativa de Rendimiento", fontweight='bold', pad=20)

# Datos de la tabla
table_data = [
    ["Métrica", "CAE (Propuesto)", "VAE (Referencia)"],
    ["F1-Score", f"{metrics_cae['f1']:.4f}", f"{metrics_vae['f1']:.4f}"],
    ["Sensibilidad (Recall)", f"{metrics_cae['rec']:.4f}", f"{metrics_vae['rec']:.4f}"],
    ["Precisión", f"{metrics_cae['prec']:.4f}", f"{metrics_vae['prec']:.4f}"],
    ["Umbral MSE", f"{metrics_cae['thresh']:.6f}", f"{metrics_vae['thresh']:.6f}"]
]

# Crear tabla visual
table = ax0.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2) # Escalar altura de filas

# Colorear encabezados
for i in range(3):
    table[(0, i)].set_facecolor('#40466e') # Azul oscuro
    table[(0, i)].set_text_props(color='white', weight='bold')
# Resaltar fila de Umbral (la última)
for i in range(3):
    table[(4, i)].set_facecolor('#f0f0f0') # Gris claro para destacar diferencia
    if i > 0: table[(4, i)].set_text_props(weight='bold', color='#d62728') # Rojo para los números

# SUBPLOT 2: Matriz CAE
ax1 = fig.add_subplot(gs[0, 1])
sns.heatmap(metrics_cae['cm'], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
            xticklabels=['Norm', 'Anom'], yticklabels=['Norm', 'Anom'])
ax1.set_title("CAE (Propuesto)")

# SUBPLOT 3: Matriz VAE
ax2 = fig.add_subplot(gs[0, 2])
sns.heatmap(metrics_vae['cm'], annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax2,
            xticklabels=['Norm', 'Anom'], yticklabels=['', ''])
ax2.set_title("VAE (Referencia)")

plt.tight_layout()
plt.savefig(os.path.join(PATH_RESULTS, '4_comparativa_final.png'), dpi=300)
plt.close()

print("\n✅ ¡Reporte Generado! Revisa la imagen '4_comparativa_final.png' para ver los umbrales.")
