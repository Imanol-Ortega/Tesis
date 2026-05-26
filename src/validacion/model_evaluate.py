import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, accuracy_score
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DATA_TEST = os.path.join(BASE_DIR, 'data', 'processed', 'test')
PATH_MODEL = os.path.join(BASE_DIR, 'models', 'CAE_1D.keras')
DIR_RESULTS = os.path.join(BASE_DIR, 'graficos', 'metricas_cae')
def evaluate():
    print("INICIANDO EVALUACIÓN DE RENDIMIENTO...")
    os.makedirs(DIR_RESULTS, exist_ok=True)
    try:
        # Se cargan los datos de prueba y el modelo entrenado
        X_test = np.load(os.path.join(PATH_DATA_TEST, 'X_test.npy'))
        y_test = np.load(os.path.join(PATH_DATA_TEST, 'y_test.npy'))
        if X_test.shape[0] == 0:
            print("Error: El dataset de prueba está vacío (0 muestras).")
            return
        model = tf.keras.models.load_model(PATH_MODEL, compile=False)
        print(f"Datos cargados: {X_test.shape} muestras.")
    except Exception as e:
        print(f"Error cargando recursos: {e}")
        return

    # Se predicen las reconstrucciones para el conjunto de prueba
    reconstructions = model.predict(X_test, verbose=1)
    # Se calcula el MSE para cada muestra (error de reconstrucción)
    mse_scores = np.mean(np.square(X_test - reconstructions), axis=1).flatten()
    print("Calculando Curva Precision-Recall...")
    precisions, recalls, thresholds = precision_recall_curve(y_test, mse_scores)
    print("Calculando F1 Scores para cada umbral...")
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10) # 1e-10 Evita división por cero
    best_idx = np.argmax(f1_scores) # Índice del umbral con mejor F1
    best_threshold = thresholds[best_idx] # Umbral óptimo basado en F1
    best_f1 = f1_scores[best_idx] # Mejor F1 Score
    # Se predice clase 1 (anómalo) si el MSE es mayor que el umbral óptimo, de lo contrario clase 0 (normal)
    y_pred = (mse_scores > best_threshold).astype(int)
    # Se calcula la precisión global del modelo con el umbral seleccionado
    final_acc = accuracy_score(y_test, y_pred)
    # Se calcula la matriz de confusión para evaluar el desempeño del modelo
    cm = confusion_matrix(y_test, y_pred)



    plt.figure(figsize=(10, 6))
    plt.hist(mse_scores[y_test==0], bins=50, alpha=0.6, color='green', label='Clase 0 (Normal)', density=True)
    plt.hist(mse_scores[y_test==1], bins=50, alpha=0.6, color='red', label='Clase 1 (Anómalo)', density=True)
    plt.axvline(best_threshold, color='black', linestyle='--', linewidth=2, label=f'Umbral ({best_threshold:.4f})')
    plt.title("Distribución del Error de Reconstrucción (MSE)")
    plt.xlabel("MSE Score")
    plt.ylabel("Densidad")
    plt.legend()
    plt.savefig(os.path.join(DIR_RESULTS, 'histograma_errores.png'))
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.', label=f'CAE (F1={best_f1:.3f})')
    plt.scatter(recalls[best_idx], precisions[best_idx], marker='o', color='red', s=100, label='Punto Óptimo (Max F1)', zorder=5)
    plt.title("Curva Precision-Recall")
    plt.xlabel("Recall (Sensibilidad)")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DIR_RESULTS, 'curva_pr.png'))
    labels = np.array([['VN', 'FP'], ['FN', 'VP']])
    # Crear anotaciones que combinen el número con la sigla técnica
    annot = np.array([[f"{val}\n({label})" for val, label in zip(row_val, row_label)]
                      for row_val, row_label in zip(cm, labels)])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=annot, fmt="", cmap='Blues', cbar=False,
                xticklabels=['Pred: Normal\n(Clase 0)', 'Pred: Anómalo\n(Clase 1)'],
                yticklabels=['Real: Normal\n(Clase 0)', 'Real: Anómalo\n(Clase 1)'])
    plt.title(f"Matriz de Confusión\nAcc: {final_acc:.2%}")
    plt.xlabel("Etiquetas Predichas")
    plt.ylabel("Etiquetas Reales")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_RESULTS, 'matriz_confusion.png'))
    print(f"\nReporte generado en: {DIR_RESULTS}")
    print("   - histograma_errores.png")
    print("   - curva_pr.png")
    print("   - matriz_confusion.png")
if __name__ == "__main__":
    evaluate()
