import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Cargar el historial
history_path = "models/training_history.npy"

if not os.path.exists(history_path):
    print("❌ No encuentro el archivo training_history.npy")
    exit()

history = np.load(history_path, allow_pickle=True).item()

# 2. Extraer datos
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss) + 1)

# 3. Graficar
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, 'b-', label='Entrenamiento (Loss)', linewidth=2)
plt.plot(epochs, val_loss, 'r--', label='Validación (Val Loss)', linewidth=2)

plt.title('Convergencia del Autoencoder: Loss vs Épocas', fontsize=16)
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 4. Guardar
os.makedirs("results", exist_ok=True)
plt.savefig("results/grafico_entrenamiento.png")
print("✅ Gráfico guardado en: results/grafico_entrenamiento.png")
plt.show()
