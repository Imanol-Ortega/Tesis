import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# GESTIÓN DE DATOS SIMULADOS (Mismos datos)
# ==========================================
np.random.seed(42)
tiempo = np.linspace(0, 20, 1000)
periodo_P = 5.0
epoca_T0 = 2.5

# Crear ruido de fondo
flujo = np.ones_like(tiempo) + np.random.normal(0, 0.003, len(tiempo))

# Simular los tránsitos (caídas periódicas ocultas)
for epoch in range(4):
    tiempo_transito = epoca_T0 + epoch * periodo_P
    mask = np.abs(tiempo - tiempo_transito) < 0.3
    flujo[mask] -= 0.015 * np.exp(-((tiempo[mask] - tiempo_transito)**2) / (2 * 0.08**2))

# ======================================================
# FIGURA 1: BLOQUE DE PASOS 1 Y 2 (Entrada y Períodos)
# ======================================================
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# --- PANEL 1: PASO 1 (Datos Crudos) ---
ax1.scatter(tiempo, flujo, s=5, color='gray', alpha=0.7)
ax1.set_title("Paso 1: Serie Temporal de Entrada (Datos Crudos)", fontweight='bold')
ax1.set_xlabel("Tiempo de observación (días)")
ax1.set_xlim(0, 20)
ax1.grid(True, linestyle=':', alpha=0.4)

# --- PANEL 2: PASO 2 (Identificación de Períodos) ---
ax2.scatter(tiempo, flujo, s=5, color='gray', alpha=0.7)
ax2.set_title("Paso 2: Identificación de Periodos (Cortes para el plegado)", fontweight='bold')
ax2.set_xlabel("Tiempo de observación (días)")
ax2.set_xlim(0, 20)
ax2.grid(True, linestyle=':', alpha=0.4)

# Marcar las líneas divisorias de los periodos
colors = ['red', 'blue', 'green', 'purple']
for i in range(4):
    ax2.axvline(x=epoca_T0 + i * periodo_P, color=colors[i], linestyle='--', alpha=0.6, label=f"Periodo {i+1}")

# Etiquetas para T0 y T0+P
ax2.text(epoca_T0 - 0.2, 1.008, '$T_0$', color='red', fontweight='bold', ha='right')
ax2.text(epoca_T0 + periodo_P - 0.2, 1.008, '$T_0 + P$', color='blue', fontweight='bold', ha='right')
ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show() # Muestra la primera imagen (Pasos 1 y 2)

# ======================================================
# FIGURA 2: BLOQUE DE PASOS 3 Y 4 (Operación y Final)
# ======================================================
# Cálculos matemáticos
fase_cruda = ((tiempo - epoca_T0) / periodo_P) % 1.0
fase_centrada = fase_cruda.copy()
fase_centrada[fase_centrada > 0.5] -= 1.0

fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))

# --- PANEL 3: PASO 3 (Visualización de la operación Módulo) ---
# Mostramos cómo se verían los periodos cortados pero alineados por colores.
colors_seg = ['red', 'blue', 'green', 'purple']
for i in range(4):
    # Crear una máscara para cada periodo completo
    mask_seg = (tiempo >= (epoca_T0 + (i - 0.5) * periodo_P)) & \
               (tiempo < (epoca_T0 + (i + 0.5) * periodo_P))
    if i == 0: mask_seg = (tiempo >= 0) & (tiempo < (epoca_T0 + (i + 0.5) * periodo_P))
    if i == 3: mask_seg = (tiempo >= (epoca_T0 + (i - 0.5) * periodo_P)) & (tiempo <= 20)

    # Tiempo relativo a T0+n*P para alinear los segmentos
    t_seg = tiempo[mask_seg] - (epoca_T0 + i * periodo_P)
    f_seg = flujo[mask_seg]
    ax3.scatter(t_seg, f_seg, s=10, color=colors_seg[i], alpha=0.3, label=f"Segmento {i+1}")

ax3.set_title("Paso 3: Superposición de Segmentos (Resultado de la operación Módulo)", fontweight='bold')
ax3.set_xlabel("Tiempo relativo a $T_0 + n \cdot P$ (días)")
ax3.set_xlim(-periodo_P/2, periodo_P/2)
ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, linestyle=':', alpha=0.4)

# --- PANEL 4: PASO 4 (Diagrama de Fase Final) ---
ax4.scatter(fase_centrada, flujo, s=8, color='black', alpha=0.8)
ax4.set_title("Paso 4: Diagrama de Fase Final (Plegado Completo)", fontweight='bold')
ax4.set_xlabel("Fase ($\phi$)")
ax4.set_xlim(-0.5, 0.5)
ax4.grid(True, linestyle=':', alpha=0.5)
ax4.axvline(x=0, color='red', linestyle='-', alpha=0.5, label='Época central ($T_0$)')

plt.tight_layout()
plt.show() # Muestra la segunda imagen (Pasos 3 y 4)
