# Configuración centralizada del proyecto GANIA 2.0 (FastGAN & Aumento Multiclase)

CONFIG = {
    "FASTGAN": {
        "BATCH_SIZE": 32,
        "TAMANO_IMAGEN": 64,
        "LEARNING_RATE": 0.0002,
        "MAX_EPOCAS": 50,
        "DIM_LATENTE": 100,
        "INTERVALO_CHECKPOINT": 5,
        # Lotes por época (muestreo con reemplazo). Con 80 imgs y batch 32, sin esto
        # una época serían solo 3 iteraciones. 500 épocas x 50 = 25.000 iteraciones.
        "ITERACIONES_POR_EPOCA": 50,
        "INTERVALO_METRICAS": 10,       # Calcular FID/KID/Diversidad cada N épocas
        "MUESTRAS_METRICAS": 256,       # Imágenes sintéticas usadas para las métricas
        "EMA_BETA": 0.999,              # Promedio móvil de pesos del generador
        "POLITICA_DIFFAUGMENT": "color,translation,cutout",
    },
    "AUMENTO": {
        "CANTIDAD_SINTETICAS": 500,  # Cantidad de imágenes sintéticas a generar por clase
        "MEZCLAR_CON_ORIGINAL": True
    },
    "CLASIFICADOR": {
        "ARQUITECTURA": "ResNet18",
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.001,
        "MAX_EPOCAS": 15,
        "PORCENTAJE_TEST": 0.2
    },
    "RUTAS": {
        "DATASETS_BASE": "./datos/datasets",
        "DATASET_ORIGINAL": "./datos/datasets/original",
        "DATASET_PREPARADO": "./datos/datasets/dataset_prepared",
        "DATASET_ZIP_EXTRAIDO": "./datos/datasets/zip_extraido",
        "DATASET_URL_EXTRAIDO": "./datos/datasets/url_extraido",
        "SINTETICAS_BASE": "./datos/datasets/synthetic",
        "AMPLIADO_BASE": "./datos/datasets/augmented",
        "RESULTADOS": "./datos/resultados",
        "CHECKPOINTS_FASTGAN": "./datos/resultados/checkpoints_fastgan",
        "CHECKPOINTS_CLASIFICADOR": "./datos/resultados/checkpoints_clasificador",
        "LOGS": "./datos/resultados/log_resultados.txt"
    }
}
