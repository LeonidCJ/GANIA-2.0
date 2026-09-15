# Configuración centralizada del proyecto GANIA 2.0

CONFIG = {
    "ENTRENAMIENTO": {
        "BATCH_SIZE": 32,  # Reducido de 64 para mayor estabilidad en PCs con menos recursos
        "TAMANO_IMAGEN": 64,
        "LEARNING_RATE": 0.0002,
        "MAX_EPOCAS": 50,
        "DIM_LATENTE": 100,
        "INTERVALO_CHECKPOINT": 5,
        "LABEL_SMOOTHING": 0.9
    },
    "RUTAS": {
        "DATASETS": "./datos/datasets",
        "RESULTADOS": "resultados",
        "CHECKPOINTS": "resultados/checkpoints",
        "LOGS": "resultados/log_resultados.txt"
    },
    "DATASETS": {
        "LIMITE_IMAGENES": 1000,
        "CIFAR10": "CIFAR-10",
        "ORANGE": "ORANGE"
    }
}
