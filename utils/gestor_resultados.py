import os
import datetime
from torchvision.utils import save_image

# Unificado con la estructura de la carpeta 'datos'
BASE_RESULTADOS = "datos/resultados"

def crear_estructura_resultados():
    """
    Crea las carpetas necesarias para guardar resultados dentro de datos/resultados.
    """
    carpetas = [
        os.path.join(BASE_RESULTADOS, "dcgan"),
        os.path.join(BASE_RESULTADOS, "cyclegan")
    ]
    for carpeta in carpetas:
        os.makedirs(carpeta, exist_ok=True)

def nombre_archivo_epoca(modelo, epoca):
    """
    Genera un nombre de archivo único para la imagen generada.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(BASE_RESULTADOS, modelo, f"{modelo}_epoca{epoca}_{timestamp}.png")

def guardar_imagen_tensor(tensor, ruta, nrow=8):
    """
    Guarda un tensor de imagen en disco como PNG.
    """
    save_image(tensor, ruta, normalize=True, nrow=nrow)

def registrar_log(modelo, epoca, ruta_imagen, archivo_log=None):
    """
    Registra una línea de log indicando que se generó y guardó una imagen.
    """
    if archivo_log is None:
        archivo_log = os.path.join(BASE_RESULTADOS, "log_resultados.txt")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{timestamp}] {modelo} - Época {epoca} - Imagen guardada: {ruta_imagen}\n"
    os.makedirs(os.path.dirname(archivo_log), exist_ok=True)
    with open(archivo_log, "a", encoding="utf-8") as f:
        f.write(linea)
