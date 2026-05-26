import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Tesis: Detección de Anomalías en Exoplanetas")
    parser.add_argument('--step', type=str, required=True,
                        choices=['download', 'preprocess', 'preprocess-test', 'train-cae', 'train-vae', 'evaluate-model-cae', 'compare-models', 'visual-reconstructions'],
                        help='Paso a ejecutar: download, preprocess, preprocess-test, train-cae, train-vae, evaluate-model-cae, compare-models, visual-reconstructions')
    parser.add_argument('--full', action='store_true',
                        help='Ejecutar descarga completa (sin límites)')
    parser.add_argument('--anomalies', action='store_true',
                        help='Si se activa, descarga Anomalías (FP) en lugar de Planetas')
    args = parser.parse_args()
    if args.step == 'download':
        from src.obtencion import dw_pl_confirmados, dw_anomalias
        if args.anomalies:
            print("\nMODO: Descarga de Anomalías (Falsos Positivos)")
            print("   -> Iniciando descarga completa")
            dw_anomalias.descargar_anomalias_full()
        else:
            print("\nMODO: Descarga de Planetas Confirmados")
            print("   -> Iniciando descarga completa")
            dw_pl_confirmados.descargar_curvas()
    elif args.step == 'preprocess':
        from src.Procesamiento import dataset_builder_train
        print("\nMODO: Preprocesamiento y Creación de Dataset")
        dataset_builder_train.build_dataset()
    elif args.step == 'preprocess-test':
        from src.Procesamiento import dataset_builder_test
        print("\nMODO: Preprocesamiento - Dataset de Prueba (Anomalías y Errores)")
        dataset_builder_test.build_test_dataset()
    elif args.step == 'train-cae':
        from src.modeloEntrenamiento import train
        print("\nMODO: Entrenamiento del Modelo Convolutional Autoencoder")
        train.train_model()
    elif args.step == 'train-vae':
        from src.modeloComparacion import train_vae
        print("\nMODO: Entrenamiento del VAE de Referencia (Hönes et al.)")
        train_vae.train_vae_reference()
    elif args.step == 'evaluate-model-cae':
        from src.validacion import model_evaluate
        print("\nMODO: Evaluación de Rendimiento y Cálculo de Umbral")
        model_evaluate.evaluate()
    elif args.step == 'compare-models':
        from pruebas import reportes_tesis
        print("\nMODO: Análisis Comparativo (CAE vs VAE)")
        reportes_tesis.generar_reporte()
    elif args.step == 'visual-reconstructions':
        from src.test import pruebas
        print("\nMODO: Inspección Visual de Reconstrucciones")
        pruebas.check_visual()


if __name__ == "__main__":
    main()
