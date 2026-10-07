# aplicacion/gestor_aumento.py
import os
import shutil
from pathlib import Path
from PIL import Image
import zipfile
from utils.cargador_datos import resolver_carpeta_train_o_base, validar_imagen_valida

class GestorAumentoDataset:
    """
    Servicio de gestión y combinación de datasets multiclase (Original + Sintético).
    """
    def __init__(self, ruta_original="./datos/datasets/original",
                 ruta_sinteticas="./datos/datasets/synthetic",
                 ruta_ampliado="./datos/datasets/augmented"):
        self.ruta_original = Path(ruta_original)
        self.ruta_sinteticas = Path(ruta_sinteticas)
        self.ruta_ampliado = Path(ruta_ampliado)

    def copiar_dataset_original(self):
        """Copia todas las carpetas e imágenes del dataset original al dataset ampliado."""
        if not self.ruta_original.exists():
            print(f"[Aumento] Ruta original {self.ruta_original} no existe.")
            return 0

        ruta_base = Path(resolver_carpeta_train_o_base(str(self.ruta_original)))
        copiadas = 0

        for root, dirs, files in os.walk(ruta_base):
            for file in files:
                filepath = Path(root) / file
                if validar_imagen_valida(str(filepath)):
                    rel_path = filepath.relative_to(ruta_base)
                    dest = self.ruta_ampliado / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(filepath, dest)
                    copiadas += 1

        print(f"[Aumento] Copiadas {copiadas} imágenes originales preservando la jerarquía por clase.")
        return copiadas

    def integrar_imagenes_sinteticas_clase(self, entrenador_dcgan, cantidad, clase_nombre="clase_01", callback_progreso=None):
        """
        Genera N imágenes sintéticas para una clase específica y las integra en:
        1. synthetic/clase_nombre/synthetic_001.png
        2. augmented/clase_nombre/synthetic_001.png
        """
        # Guardar en folder synthetic/clase_nombre
        dir_sinteticas = self.ruta_sinteticas / clase_nombre
        dir_ampliado = self.ruta_ampliado / clase_nombre

        dir_sinteticas.mkdir(parents=True, exist_ok=True)
        dir_ampliado.mkdir(parents=True, exist_ok=True)

        print(f"[Aumento] Generando {cantidad} imágenes sintéticas para la clase '{clase_nombre}'...")

        # Generar en folder sintéticas
        rutas_sinteticas = entrenador_dcgan.generar_lote_sintetico(
            cantidad=cantidad,
            ruta_destino=str(self.ruta_sinteticas),
            clase_nombre=clase_nombre,
            callback_progreso=callback_progreso
        )

        # Copiar también al dataset ampliado
        for src_path in rutas_sinteticas:
            filename = os.path.basename(src_path)
            dest_path = dir_ampliado / filename
            shutil.copy2(src_path, dest_path)

        return len(rutas_sinteticas)

    def exportar_zip_dataset_ampliado(self, ruta_zip_destino="./datos/resultados/dataset_ampliado.zip"):
        """Empaca el dataset ampliado en un archivo ZIP."""
        Path(ruta_zip_destino).parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(ruta_zip_destino, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(self.ruta_ampliado):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(self.ruta_ampliado.parent)
                    zip_file.write(file_path, arcname=arcname)

        print(f"[Aumento] ZIP exportado en: {ruta_zip_destino}")
        return ruta_zip_destino
