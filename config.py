# Configuración centralizada del proyecto GANIA 2.0 (FastGAN & Aumento Multiclase por Porcentaje)

CONFIG = {
    "FASTGAN": {
        "BATCH_SIZE": 32,
        "TAMANO_IMAGEN": 64,
        "LEARNING_RATE": 0.0002,
        "MAX_EPOCAS": 100000,
        "DIM_LATENTE": 100,
        "INTERVALO_CHECKPOINT": 25,
        "ITERACIONES_POR_EPOCA": 50,
        "INTERVALO_METRICAS": 10,
        "MUESTRAS_METRICAS": 256,
        "EMA_BETA": 0.999,
        "POLITICA_DIFFAUGMENT": "color,translation,cutout",
    },
    "AUMENTO": {
        "CANTIDAD_SINTETICAS": 500,  # Cantidad por defecto
        "MEZCLAR_CON_ORIGINAL": True
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
        "LOGS": "./datos/resultados/log_resultados.txt"
    }
}
