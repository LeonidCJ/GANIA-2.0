# aplicacion/sistema_gania.py
"""
Controlador de ejecución asíncrona (Workers en PyQT) para:
1. Entrenamiento de FastGAN por Clase
2. Generación Masiva por Clase y Dataset Ampliado
3. Clasificación y Evaluación Comparativa
"""
import os
import torch
import numpy as np
import torchvision.utils as vutils
from PyQt5.QtCore import QObject, pyqtSignal, QThread

from modelos.fastgan import EntrenadorFastGAN
from modelos.clasificador import EntrenadorClasificador
from aplicacion.gestor_aumento import GestorAumentoDataset
from utils.cargador_datos import crear_cargador_fastgan_clase, crear_cargadores_clasificacion
from utils.tensor_utils import tensor_to_numpy


class WorkerFastGAN(QThread):
    """Worker asíncrono para el entrenamiento de la red FastGAN en una clase específica."""
    finalizado_epoca = pyqtSignal(int, float, float, object)
    entrenamiento_completado = pyqtSignal(str)
    senal_log = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, modelo_fastgan, ruta_dataset, clase_seleccionada=None, batch_size=32, lr=0.0002, max_epocas=50):
        super().__init__()
        self.modelo = modelo_fastgan
        self.ruta_dataset = ruta_dataset
        self.clase_seleccionada = clase_seleccionada
        self.batch_size = batch_size
        self.lr = lr
        self.max_epocas = max_epocas
        self._abortar = False

    def run(self):
        try:
            from config import CONFIG
            from utils.cargador_datos import cargar_imagenes_reales_clase
            from utils.metricas_ia import calcular_metricas_calidad, extraer_caracteristicas_reales

            cfg = CONFIG["FASTGAN"]
            iters_epoca = cfg.get("ITERACIONES_POR_EPOCA", 50)
            intervalo_metricas = cfg.get("INTERVALO_METRICAS", 10)
            n_muestras_metricas = cfg.get("MUESTRAS_METRICAS", 256)
            dispositivo_str = str(self.modelo.dispositivo)

            self.senal_log.emit(f"[Worker FastGAN] Cargando clase '{self.clase_seleccionada or 'Todas'}'...")
            cargador = crear_cargador_fastgan_clase(
                self.ruta_dataset,
                clase_seleccionada=self.clase_seleccionada,
                batch_size=self.batch_size,
                iteraciones_por_epoca=iters_epoca
            )
            self.senal_log.emit(
                f"[Worker FastGAN] {iters_epoca} iteraciones/época x {self.max_epocas} épocas = "
                f"{iters_epoca * self.max_epocas:,} iteraciones totales (DiffAugment + EMA activos)."
            )

            # Imágenes reales (todas, sin aumentos) y sus features: se calculan una sola vez.
            reales_metricas = cargar_imagenes_reales_clase(self.ruta_dataset, self.clase_seleccionada)
            feats_reales = None
            if len(reales_metricas) >= 5:
                feats_reales = extraer_caracteristicas_reales(reales_metricas, dispositivo_str)

            for epoca in range(self.max_epocas):
                if self._abortar:
                    self.senal_log.emit("[Worker FastGAN] Entrenamiento cancelado por el usuario.")
                    return

                self.senal_log.emit(f"[Worker FastGAN] Procesando Época {epoca + 1}/{self.max_epocas} (Clase: {self.clase_seleccionada or 'Base'})...")
                def check_abort(): return self._abortar

                loss_G, loss_D = self.modelo.entrenar_epoca(cargador, epoca, self.max_epocas, abort_callback=check_abort)

                if self._abortar:
                    return

                # Muestra visual: Rejilla de 3x3 con el generador EMA en modo evaluación
                sample_tensors = self.modelo.generar(self.modelo.ruido_fijo[:9]).cpu()
                grid_tensor = vutils.make_grid(sample_tensors, nrow=3, normalize=True, value_range=(-1, 1), padding=2)
                img_muestra = tensor_to_numpy(grid_tensor)

                # Métricas (cada N épocas, en la primera y en la última)
                metricas = None
                num_ep = epoca + 1
                if num_ep == 1 or num_ep % intervalo_metricas == 0 or num_ep == self.max_epocas:
                    try:
                        falsas = []
                        while len(falsas) < n_muestras_metricas:
                            n = min(64, n_muestras_metricas - len(falsas))
                            z = torch.randn(n, self.modelo.dim_latente, 1, 1, device=self.modelo.dispositivo)
                            falsas.extend(tensor_to_numpy(t) for t in self.modelo.generar(z).cpu())
                        metricas = calcular_metricas_calidad(reales_metricas, falsas, dispositivo_str, feats_reales=feats_reales)
                        self.senal_log.emit(
                            f"[Métricas] Época {num_ep}: KID×1000={metricas.get('kid')} | FID={metricas.get('fid')} | "
                            f"Diversidad={metricas.get('diversidad')} ({len(falsas)} sintéticas vs {len(reales_metricas)} reales)"
                        )
                    except Exception as e:
                        self.senal_log.emit(f"[Métricas] Error: {e}")

                self.finalizado_epoca.emit(num_ep, loss_G, loss_D, {
                    "img": img_muestra,
                    "metricas": metricas
                })

                # Guardar checkpoint
                if num_ep % cfg.get("INTERVALO_CHECKPOINT", 5) == 0 or num_ep == self.max_epocas:
                    self.modelo.guardar_checkpoint(num_ep)

            self.entrenamiento_completado.emit(f"Entrenamiento FastGAN completado para clase '{self.clase_seleccionada or 'Base'}'.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Error en FastGAN: {str(e)}")

    def abortar(self):
        self._abortar = True


WorkerDCGAN = WorkerFastGAN  # Alias para compatibilidad


class WorkerGeneracionMasiva(QThread):
    """Worker asíncrono para generar N imágenes sintéticas de una clase y crear el Dataset Ampliado."""
    progreso = pyqtSignal(int, int)
    completado = pyqtSignal(str, str, object) # (Ruta dataset ampliado, Ruta ZIP, Rejilla Muestras)
    senal_log = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, entrenador_fastgan, ruta_original, clase_seleccionada="clase_01", cantidad_sinteticas=500):
        super().__init__()
        self.fastgan = entrenador_fastgan
        self.ruta_original = ruta_original
        self.clase_seleccionada = clase_seleccionada
        self.cantidad_sinteticas = cantidad_sinteticas

    def run(self):
        try:
            self.senal_log.emit(f"[Aumento] Generando {self.cantidad_sinteticas} imágenes sintéticas para la clase '{self.clase_seleccionada}'...")
            gestor = GestorAumentoDataset(ruta_original=self.ruta_original)

            # 1. Copiar dataset original preservando estructura multiclase
            self.senal_log.emit("[Aumento] Copiando estructura original al dataset ampliado...")
            gestor.copiar_dataset_original()

            # 2. Generar imágenes sintéticas en synthetic/clase_nombre y augmented/clase_nombre
            def cb_prog(actual, total):
                self.progreso.emit(actual, total)

            sinteticas_creadas = gestor.integrar_imagenes_sinteticas_clase(
                entrenador_fastgan=self.fastgan,
                cantidad=self.cantidad_sinteticas,
                clase_nombre=self.clase_seleccionada,
                callback_progreso=cb_prog
            )

            # Rejilla para vista previa de generadas en UI
            with torch.no_grad():
                sample_tensors = self.fastgan.generar(torch.randn(9, self.fastgan.dim_latente, 1, 1, device=self.fastgan.dispositivo)).cpu()
                grid_tensor = vutils.make_grid(sample_tensors, nrow=3, normalize=True, value_range=(-1, 1), padding=2)
                img_grid_sinteticas = tensor_to_numpy(grid_tensor)

            # 3. Exportar paquete ZIP
            self.senal_log.emit("[Aumento] Empacando Dataset Ampliado...")
            ruta_zip = gestor.exportar_zip_dataset_ampliado()

            self.completado.emit(str(gestor.ruta_ampliado), str(ruta_zip), img_grid_sinteticas)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Error en Aumento de Dataset: {str(e)}")


class WorkerClasificacion(QThread):
    """Worker asíncrono para entrenar y evaluar clasificadores (Módulo interno)."""
    progreso_epoca = pyqtSignal(str, int, int, float, float)
    completado = pyqtSignal(dict)
    senal_log = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, dispositivo, ruta_original, ruta_ampliado, arquitectura="ResNet18", batch_size=32, lr=0.001, max_epocas=15):
        super().__init__()
        self.dispositivo = dispositivo
        self.ruta_original = ruta_original
        self.ruta_ampliado = ruta_ampliado
        self.arquitectura = arquitectura
        self.batch_size = batch_size
        self.lr = lr
        self.max_epocas = max_epocas
        self._abortar = False

    def run(self):
        try:
            resultados = {}
            # Original
            self.senal_log.emit("\n=== [1/2] Entrenando Clasificador en Dataset ORIGINAL ===")
            loader_tr_orig, loader_te_orig, n_clases_orig = crear_cargadores_clasificacion(self.ruta_original, batch_size=self.batch_size)
            clf_orig = EntrenadorClasificador(self.dispositivo, self.arquitectura, max(2, n_clases_orig), self.lr)

            for ep in range(self.max_epocas):
                if self._abortar: return
                loss, acc = clf_orig.entrenar_epoca(loader_tr_orig)
                self.progreso_epoca.emit("Original", ep + 1, self.max_epocas, loss, acc)

            metricas_orig = clf_orig.evaluar(loader_te_orig)
            resultados["original"] = metricas_orig

            # Ampliado
            self.senal_log.emit("\n=== [2/2] Entrenando Clasificador en Dataset AMPLIADO ===")
            loader_tr_amp, loader_te_amp, n_clases_amp = crear_cargadores_clasificacion(self.ruta_ampliado, batch_size=self.batch_size)
            clf_amp = EntrenadorClasificador(self.dispositivo, self.arquitectura, max(2, n_clases_amp), self.lr)

            for ep in range(self.max_epocas):
                if self._abortar: return
                loss, acc = clf_amp.entrenar_epoca(loader_tr_amp)
                self.progreso_epoca.emit("Ampliado", ep + 1, self.max_epocas, loss, acc)

            metricas_amp = clf_amp.evaluar(loader_te_amp)
            resultados["ampliado"] = metricas_amp

            if not self._abortar:
                self.completado.emit(resultados)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Error en clasificación: {str(e)}")

    def abortar(self):
        self._abortar = True


class SistemaGania(QObject):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
