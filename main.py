import argparse
import os
import sys
# --- IMPORTACIONES ---
from src.obtencion import dw_pl_confirmados
from src.obtencion import dw_anomalias
from src.Procesamiento import dataset_builder_pl
from src.Procesamiento import dataset_builder_an
from src.modeloEntrenamiento import train
from src.modeloComparacion import train_vae
from src.validacion import model_evaluate
from src.modeloComparacion import compare_models
from src.test import pruebas
# Se importa el módulo de reportes (asumiendo que está en el path o carpeta raíz)
try: from pruebas import reportes_tesis
except ImportError: reportes_tesis = None

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Tesis: Detección de Anomalías en Exoplanetas")
    parser.add_argument('--step', type=str, required=True,
                        choices=['download', 'preprocess', 'preprocess-test', 'train', 'train-vae', 'evaluate', 'compare', 'visual', 'report'],
                        help='Paso a ejecutar: download, preprocess, preprocess-test, train, train-vae, evaluate, compare, visual, report')
    parser.add_argument('--full', action='store_true',
                        help='Ejecutar descarga completa (sin límites)')
    parser.add_argument('--anomalies', action='store_true',
                        help='Si se activa, descarga Anomalías (FP) en lugar de Planetas')
    args = parser.parse_args()
    if args.step == 'download':
        if args.anomalies:
            print("\nMODO: Descarga de Anomalías (Falsos Positivos)")
            print("   -> Iniciando descarga completa")
            dw_anomalias.descargar_anomalias_full()
        else:
            print("\nMODO: Descarga de Planetas Confirmados")
            print("   -> Iniciando descarga completa")
            dw_pl_confirmados.descargar_curvas()
    elif args.step == 'preprocess':
        print("\nMODO: Preprocesamiento y Creación de Dataset")
        dataset_builder_pl.build_dataset()
    elif args.step == 'preprocess-test':
        print("\nMODO: Preprocesamiento - Dataset de Prueba (Anomalías y Errores)")
        dataset_builder_an.build_test_dataset()
    elif args.step == 'train':
        print("\nMODO: Entrenamiento del Modelo Convolutional Autoencoder")
        train.train_model()
    elif args.step == 'train-vae':
        print("\nMODO: Entrenamiento del VAE de Referencia (Hönes et al.)")
        train_vae.train_vae_reference()
    elif args.step == 'evaluate':
        print("\nMODO: Evaluación de Rendimiento y Cálculo de Umbral")
        model_evaluate.evaluate()
    elif args.step == 'compare':
        print("\nMODO: Análisis Comparativo (CAE vs VAE)")
        compare_models.compare()
    elif args.step == 'visual':
        print("\nMODO: Inspección Visual de Reconstrucciones")
        pruebas.check_visual()
    elif args.step == 'report':
        print("\nMODO: Generación de Reportes y Gráficas de Tesis")
        if reportes_tesis is None:
            print("Error: El módulo 'reportes_tesis' no pudo ser cargado.")

if __name__ == "__main__":
    main()
