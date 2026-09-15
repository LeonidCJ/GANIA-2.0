# -*- coding: utf-8 -*-
import logging
import os
from config import CONFIG

def configurar_logging():
    """Configura el sistema de logs global para el proyecto."""
    ruta_log = CONFIG["RUTAS"]["LOGS"]
    os.makedirs(os.path.dirname(ruta_log), exist_ok=True)

    # Formato profesional para tesis: Timestamp [Nivel] Modulo: Mensaje
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] GANIA.%(name)s: %(message)s',
        handlers=[
            logging.FileHandler(ruta_log, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("=== SISTEMA DE LOGS INICIALIZADO ===")

def obtener_logger(nombre):
    """Devuelve una instancia de logger para un módulo específico."""
    return logging.getLogger(nombre)
