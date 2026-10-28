# aplicacion/gestor_aumento.py
import os
import shutil
from pathlib import Path
import zipfile
from utils.cargador_datos import resolver_carpeta_train_o_base, EXTENSIONES_VALIDAS


class GestorAumentoDataset:
    """
    Servicio de gestión y combinación de datasets multiclase (Original + Sintético FastGAN).
    Optimizado para evitar re-copias innecesarias de disco y limpiar residuos sintéticos previos.
    Todas las fases reportan progreso real (archivo por archivo) para que la barra de la
    interfaz refleje el tiempo de espera verdadero.
    """
    def __init__(self, ruta_original="./datos/datasets/original",
                 ruta_sinteticas="./datos/datasets/synthetic",
                 ruta_ampliado="./datos/datasets/augmented"):
        self.ruta_original = Path(ruta_original)
        self.ruta_sinteticas = Path(ruta_sinteticas)
        self.ruta_ampliado = Path(ruta_ampliado)

    # ------------------------------------------------------------------
    # Fase 1: sincronizar dataset original
    # ------------------------------------------------------------------
    def listar_copias_originales(self):
        """
        Devuelve la lista de pares (origen, destino) de imágenes originales
        que realmente necesitan copiarse (nuevas o con tamaño distinto).
        """
        if not self.ruta_original.exists():
            return [], 0

        ruta_base = Path(resolver_carpeta_train_o_base(str(self.ruta_original)))
        pendientes = []
        total = 0
        for root, _, files in os.walk(ruta_base):
            rel_dir = Path(root).relative_to(ruta_base)
            dest_dir = self.ruta_ampliado / rel_dir
            for file in files:
                if file.lower().endswith(EXTENSIONES_VALIDAS):
                    total += 1
                    src_file = Path(root) / file
                    dest_file = dest_dir / file
                    if not dest_file.exists() or dest_file.stat().st_size != src_file.stat().st_size:
                        pendientes.append((src_file, dest_file))
        return pendientes, total

    def copiar_dataset_original(self, pendientes=None, callback_progreso=None):
        """
        Copia las imágenes del dataset original al dataset ampliado.
        Omite imágenes que ya existan en el destino para acelerar ejecuciones sucesivas.
        """
        if pendientes is None:
            pendientes, _ = self.listar_copias_originales()

        total = len(pendientes)
        for i, (src_file, dest_file) in enumerate(pendientes, start=1):
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)
            if callback_progreso:
                callback_progreso(i, total)

        print(f"[Aumento] Copiadas {total} imágenes originales nuevas preservando la jerarquía por clase.")
        return total

    # ------------------------------------------------------------------
    # Fase 2 y 3: generar sintéticas y copiarlas al dataset ampliado
    # ------------------------------------------------------------------
    def _limpiar_sinteticas_previas_clase(self, dir_sinteticas: Path, dir_ampliado: Path):
        """
        Elimina únicamente las imágenes sintéticas previas (synthetic_*.png) de la clase actual
        para que si el usuario cambia el porcentaje (ej. de 100% a 25%), el conteo sea exacto.
        """
        if dir_sinteticas.exists():
            shutil.rmtree(dir_sinteticas, ignore_errors=True)

        if dir_ampliado.exists():
            for archivo in dir_ampliado.glob("synthetic_*.png"):
                try:
                    archivo.unlink()
                except OSError:
                    pass

    def integrar_imagenes_sinteticas_clase(
        self,
        entrenador_fastgan,
        cantidad,
        clase_nombre="clase_01",
        callback_progreso=None,
        callback_copia=None
    ):
        """
        Genera N imágenes sintéticas con FastGAN para una clase específica y las integra en:
        1. synthetic/clase_nombre/synthetic_001.png
        2. augmented/clase_nombre/synthetic_001.png
        """
        dir_sinteticas = self.ruta_sinteticas / clase_nombre
        dir_ampliado = self.ruta_ampliado / clase_nombre

        self._limpiar_sinteticas_previas_clase(dir_sinteticas, dir_ampliado)

        dir_sinteticas.mkdir(parents=True, exist_ok=True)
        dir_ampliado.mkdir(parents=True, exist_ok=True)

        print(f"[Aumento] Generando {cantidad} imágenes sintéticas FastGAN para la clase '{clase_nombre}'...")

        rutas_sinteticas = entrenador_fastgan.generar_lote_sintetico(
            cantidad=cantidad,
            ruta_destino=str(self.ruta_sinteticas),
            clase_nombre=clase_nombre,
            callback_progreso=callback_progreso
        )

        total = len(rutas_sinteticas)
        for i, src_path in enumerate(rutas_sinteticas, start=1):
            shutil.copy2(src_path, dir_ampliado / os.path.basename(src_path))
            if callback_copia:
                callback_copia(i, total)

        return total

    # ------------------------------------------------------------------
    # Fase 4: exportar ZIP
    # ------------------------------------------------------------------
    def contar_archivos_ampliado(self):
        return sum(len(files) for _, _, files in os.walk(self.ruta_ampliado))

    def exportar_zip_dataset_ampliado(self, ruta_zip_destino="./datos/resultados/dataset_ampliado.zip",
                                      callback_progreso=None):
        """
        Empaca el dataset ampliado en un ZIP sin recompresión (las imágenes .png/.jpg
        ya están comprimidas), reportando progreso archivo por archivo.
        """
        Path(ruta_zip_destino).parent.mkdir(parents=True, exist_ok=True)

        archivos = []
        for root, _, files in os.walk(self.ruta_ampliado):
            for file in files:
                archivos.append(Path(root) / file)

        total = len(archivos)
        with zipfile.ZipFile(ruta_zip_destino, "w", compression=zipfile.ZIP_STORED) as zip_file:
            for i, file_path in enumerate(archivos, start=1):
                arcname = file_path.relative_to(self.ruta_ampliado.parent)
                zip_file.write(file_path, arcname=arcname)
                if callback_progreso:
                    callback_progreso(i, total)

        print(f"[Aumento] ZIP exportado en: {ruta_zip_destino}")
        return ruta_zip_destino
