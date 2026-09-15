# -*- coding: utf-8 -*-
import json
import os

RUTA_CONFIG = os.path.join("datos", "config_usuario.json")

def guardar_config_usuario(usuario_id, ruta_drive=""):
    """Guarda la configuración del usuario en un archivo JSON local."""
    config = {
        "usuario_id": usuario_id,
        "ruta_drive": ruta_drive
    }
    os.makedirs(os.path.dirname(RUTA_CONFIG), exist_ok=True)
    with open(RUTA_CONFIG, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def cargar_config_usuario():
    """Carga la configuración guardada si existe."""
    if os.path.exists(RUTA_CONFIG):
        try:
            with open(RUTA_CONFIG, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    return None
