import argparse
import os
import sys
# --- IMPORTACIONES ---
from src.obtencion import dw_pl_confirmados       # Tu descargador de Planetas (Buenos)
from src.obtencion import dw_anomalias
from src.Procesamiento import dataset_builder # Tu descargador de Anomalías (Malos)
def main():
    parser = argparse.ArgumentParser(description="Pipeline de Tesis: Detección de Anomalías en Exoplanetas")
    # 1. Argumento principal: ¿Qué paso vamos a hacer?
    parser.add_argument('--step', type=str, required=True,
                        choices=['download', 'preprocess', 'train'],
                        help='Paso a ejecutar: download, preprocess, train')
    # 2. Argumentos opcionales (Modificadores)
    parser.add_argument('--full', action='store_true',
                        help='Ejecutar descarga completa (sin límites)')
    parser.add_argument('--anomalies', action='store_true',
                        help='Si se activa, descarga Anomalías (FP) en lugar de Planetas')
    args = parser.parse_args()
    # --- LÓGICA DEL PIPELINE ---
    if args.step == 'download':
        # OPCIÓN A: Descargar Anomalías (Tu archivo dw_anomalias.py)
        if args.anomalies:
            print("\n🔵 MODO: Descarga de Anomalías (Falsos Positivos/Binarias)")
            # Llamamos a la función principal de tu nuevo archivo
            dw_anomalias.descargar_anomalias_full()
        # OPCIÓN B: Descargar Planetas Confirmados (Tu archivo dw_pl_confirmados.py)
        else:
            print("\n🟢 MODO: Descarga de Planetas Confirmados")
            # Configuración de límites
            if args.full:
                print("   -> Descarga COMPLETA activada")
                dw_pl_confirmados.descargar_curvas(test_mode=False)
            else:
                print("   -> Modo PRUEBA (Limitado a 5 planetas)")
                dw_pl_confirmados.descargar_curvas(test_mode=True, limit=5)
    elif args.step == 'preprocess':
        print("\n⚙️ MODO: Preprocesamiento y Creación de Dataset")
        dataset_builder.construir_dataset()
    elif args.step == 'train':
        print("🚧 [PENDIENTE] El módulo de entrenamiento se integrará al final.")
if __name__ == "__main__":
    main()
