from datetime import datetime
import os
from pathlib import Path
import zipfile

from PIL import Image


BASE_DATOS = Path("datos")
DIR_DATASETS = BASE_DATOS / "datasets"
DIR_RESULTADOS = BASE_DATOS / "resultados"
DIR_RESULTADOS_GUARDADOS = BASE_DATOS / "resultados_guardados"


def crear_estructura_datos():
    for ruta in (DIR_DATASETS, DIR_RESULTADOS, DIR_RESULTADOS_GUARDADOS):
        ruta.mkdir(parents=True, exist_ok=True)


def guardar_imagen_numpy(imagen, nombre_modelo):
    crear_estructura_datos()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = DIR_RESULTADOS_GUARDADOS / f"{nombre_modelo.lower()}_{timestamp}.png"
    Image.fromarray(imagen).save(ruta)
    return ruta


def preparar_reporte_resultados():
    crear_estructura_datos()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_zip = DIR_RESULTADOS / f"reporte_resultados_{timestamp}.zip"

    with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_DEFLATED) as archivo_zip:
        for ruta in DIR_RESULTADOS_GUARDADOS.glob("*"):
            if ruta.is_file():
                archivo_zip.write(ruta, arcname=f"resultados_guardados/{ruta.name}")

    return ruta_zip
