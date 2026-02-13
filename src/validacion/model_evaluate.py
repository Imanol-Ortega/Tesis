import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, accuracy_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DATA_TEST = os.path.join(BASE_DIR, 'data', 'processed', 'test')
PATH_MODEL = os.path.join(BASE_DIR, 'models', 'CAE_1D.keras')
DIR_RESULTS = os.path.join(BASE_DIR, 'results', 'validacion_fase3')

def evaluate():
    print("INICIANDO EVALUACIÓN DE RENDIMIENTO (FASE 3)...")
    os.makedirs(DIR_RESULTS, exist_ok=True)

    try:
        X_test = np.load(os.path.join(PATH_DATA_TEST, 'X_test.npy'))
        y_test = np.load(os.path.join(PATH_DATA_TEST, 'y_test.npy'))

        if X_test.shape[0] == 0:
            print("❌ Error: El dataset de prueba está vacío (0 muestras).")
            print("   -> Baja el filtro de profundidad en 'data_builder_model.py' y regenera el dataset.")
            return


        model = tf.keras.models.load_model(PATH_MODEL, compile=False)
        print(f"Datos cargados: {X_test.shape} muestras.")
    except Exception as e:
        print(f"❌ Error cargando recursos: {e}")
        return


    print("🧠 Ejecutando inferencia en Test Set...")
    reconstructions = model.predict(X_test, verbose=1)


    mse_scores = np.mean(np.square(X_test - reconstructions), axis=1).flatten()


    print("Calculando Curva Precision-Recall...")
    precisions, recalls, thresholds = precision_recall_curve(y_test, mse_scores)

    precisions = precisions[:-1]
    recalls = recalls[:-1]


    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)


    beta = 2
    f2_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-10)


    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx]
    best_f2 = f2_scores[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\nUMBRAL ÓPTIMO (Max F2 - Prioridad Recall): {best_threshold:.6f}")
    print(f"   F2-Score: {best_f2:.4f} (Usado para optimizar)")
    print(f"   F1-Score: {best_f1:.4f} (Referencia)")


    y_pred = (mse_scores > best_threshold).astype(int)


    final_acc = accuracy_score(y_test, y_pred)
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
    plt.plot(recalls, precisions, marker='.', label=f'CAE (F2={best_f2:.3f})')
    plt.scatter(recalls[best_idx], precisions[best_idx], marker='o', color='red', s=100, label='Punto Óptimo (Max F2)', zorder=5)
    plt.title("Curva Precision-Recall")
    plt.xlabel("Recall (Sensibilidad)")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DIR_RESULTS, 'curva_pr.png'))


    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred: Normal', 'Pred: Anómalo'],
                yticklabels=['Real: Normal', 'Real: Anómalo'])
    plt.title(f"Matriz de Confusión\nAcc: {final_acc:.2%}")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_RESULTS, 'matriz_confusion.png'))

    print(f"\nReporte generado en: {DIR_RESULTS}")
    print("   - histograma_errores.png")
    print("   - curva_pr.png")
    print("   - matriz_confusion.png")

if __name__ == "__main__":
    evaluate()
