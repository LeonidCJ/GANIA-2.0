# aplicacion/sistema_gania.py
"""
Controlador de ejecución asíncrona (Workers en PyQT) para:
1. Entrenamiento de FastGAN por Clase
2. Generación Masiva por Clase y Dataset Ampliado por Porcentaje
"""
import os
import torch
import numpy as np
import torchvision.utils as vutils
from PyQt5.QtCore import QObject, pyqtSignal, QThread

from modelos.fastgan import EntrenadorFastGAN
from aplicacion.gestor_aumento import GestorAumentoDataset
from utils.cargador_datos import crear_cargador_fastgan_clase
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

            cfg = CONFIG.get("FASTGAN", {})
            dispositivo_str = str(self.modelo.dispositivo)
            es_cpu = (dispositivo_str == "cpu")

            if es_cpu:
                iters_epoca = min(cfg.get("ITERACIONES_POR_EPOCA", 50), 10)
                batch_efectivo = self.batch_size
                intervalo_metricas = max(cfg.get("INTERVALO_METRICAS", 10), 25)
                n_muestras_metricas = min(cfg.get("MUESTRAS_METRICAS", 256), 48)
                self.senal_log.emit(
                    f"[Modo CPU Ligero Activo] Ajustando a {iters_epoca} iters/época, "
                    f"batch_size={batch_efectivo} y {n_muestras_metricas} muestras para métricas."
                )
            else:
                iters_epoca = cfg.get("ITERACIONES_POR_EPOCA", 50)
                batch_efectivo = self.batch_size
                intervalo_metricas = cfg.get("INTERVALO_METRICAS", 10)
                n_muestras_metricas = cfg.get("MUESTRAS_METRICAS", 256)

            tam_img = getattr(self.modelo, "tamano_img", 64)
            self.senal_log.emit(f"[Worker FastGAN] Cargando clase '{self.clase_seleccionada or 'Todas'}' ({tam_img}x{tam_img} px)...")
            cargador = crear_cargador_fastgan_clase(
                self.ruta_dataset,
                clase_seleccionada=self.clase_seleccionada,
                batch_size=batch_efectivo,
                tamano_imagen=tam_img,
                iteraciones_por_epoca=iters_epoca
            )

            reales_metricas = cargar_imagenes_reales_clase(self.ruta_dataset, self.clase_seleccionada, tamano_imagen=tam_img)
            if es_cpu and len(reales_metricas) > 64:
                reales_metricas = reales_metricas[:64]

            feats_reales = None

            for epoca in range(self.max_epocas):
                if self._abortar:
                    self.senal_log.emit("[Worker FastGAN] Entrenamiento cancelado por el usuario.")
                    return

                num_ep = epoca + 1
                self.senal_log.emit(f"[Worker FastGAN] Procesando Época {num_ep}/{self.max_epocas} (Clase: {self.clase_seleccionada or 'Base'})...")

                def check_abort():
                    return self._abortar

                loss_G, loss_D = self.modelo.entrenar_epoca(cargador, epoca, self.max_epocas, abort_callback=check_abort)

                if self._abortar:
                    return

                # Muestra visual: Rejilla 3x3 en entrenamiento
                sample_tensors = self.modelo.generar(self.modelo.ruido_fijo[:9]).cpu()
                grid_tensor = vutils.make_grid(sample_tensors, nrow=3, normalize=True, value_range=(-1, 1), padding=2)
                img_muestra = tensor_to_numpy(grid_tensor)

                metricas = None
                calcular_ahora = (
                    (not es_cpu and num_ep == 1) or
                    (num_ep % intervalo_metricas == 0) or
                    (num_ep == self.max_epocas)
                )

                if calcular_ahora and len(reales_metricas) >= 5:
                    try:
                        self.senal_log.emit(f"[Métricas] Calculando FID/KID en época {num_ep}...")
                        if feats_reales is None:
                            feats_reales = extraer_caracteristicas_reales(reales_metricas, dispositivo_str)

                        falsas = []
                        while len(falsas) < n_muestras_metricas:
                            if self._abortar:
                                return
                            n = min(32 if es_cpu else 64, n_muestras_metricas - len(falsas))
                            z = torch.randn(n, self.modelo.dim_latente, 1, 1, device=self.modelo.dispositivo)
                            falsas.extend(tensor_to_numpy(t) for t in self.modelo.generar(z).cpu())

                        metricas = calcular_metricas_calidad(reales_metricas, falsas, dispositivo_str, feats_reales=feats_reales)
                    except Exception as e:
                        self.senal_log.emit(f"[Métricas] Error: {e}")

                self.finalizado_epoca.emit(num_ep, loss_G, loss_D, {
                    "img": img_muestra,
                    "metricas": metricas
                })

                intervalo_ckpt = cfg.get("INTERVALO_CHECKPOINT", 10 if es_cpu else 5)
                if num_ep % intervalo_ckpt == 0 or num_ep == self.max_epocas:
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
    """
    Worker asíncrono para generar N imágenes sintéticas de una clase y crear el Dataset Ampliado.

    La barra de progreso se calcula con "unidades de trabajo" ponderadas por el costo real
    de cada fase (copia, generación, copia al ampliado, ZIP y carga de galería), de modo que
    el 100% coincide con el momento en que el usuario ya tiene las imágenes en pantalla.
    """
    progreso = pyqtSignal(int, str)          # (porcentaje 0-100, descripción de la fase)
    completado = pyqtSignal(str, str, str)   # (Ruta dataset ampliado, Ruta ZIP, Ruta carpeta sintéticas)
    senal_log = pyqtSignal(str)
    error = pyqtSignal(str)

    # Costos relativos aproximados por elemento (en "unidades" ~ milisegundos)
    COSTO_COPIA = 2.0
    COSTO_ZIP = 1.5
    COSTO_GALERIA = 8.0
    COSTO_GUARDADO_PNG = 4.0

    def __init__(self, entrenador_fastgan, ruta_original, clase_seleccionada="clase_01", cantidad_sinteticas=500):
        super().__init__()
        self.fastgan = entrenador_fastgan
        self.ruta_original = ruta_original
        self.clase_seleccionada = clase_seleccionada
        self.cantidad_sinteticas = cantidad_sinteticas
        self.unidades_totales = 1.0
        self.unidades_hechas = 0.0
        self.unidades_galeria = 0.0
        self._ultimo_pct = -1

    def _costo_generacion_por_imagen(self):
        res = getattr(self.fastgan, "tamano_img", 64)
        factor_res = (res / 64.0) ** 2
        es_cpu = str(getattr(self.fastgan, "dispositivo", "cpu")) == "cpu"
        base = 12.0 if es_cpu else 2.0
        return base * factor_res + self.COSTO_GUARDADO_PNG * factor_res

    def _emitir(self, fase):
        pct = int(min(99, 100.0 * self.unidades_hechas / max(1.0, self.unidades_totales)))
        if pct != self._ultimo_pct:
            self._ultimo_pct = pct
            self.progreso.emit(pct, fase)

    def _crear_callback(self, costo_unitario, fase):
        base = self.unidades_hechas
        def cb(actual, total):
            self.unidades_hechas = base + actual * costo_unitario
            self._emitir(f"{fase} ({actual}/{total})")
        return cb

    def run(self):
        try:
            self.senal_log.emit(f"[Aumento] Generando {self.cantidad_sinteticas} imágenes sintéticas para la clase '{self.clase_seleccionada}'...")
            gestor = GestorAumentoDataset(ruta_original=self.ruta_original)

            # 0. Estimar el trabajo total antes de empezar
            self.progreso.emit(0, "Preparando...")
            pendientes, total_originales = gestor.listar_copias_originales()
            n = self.cantidad_sinteticas
            costo_gen = self._costo_generacion_por_imagen()
            archivos_zip_estimados = total_originales + n

            u_copia_orig = len(pendientes) * self.COSTO_COPIA
            u_gen = n * costo_gen
            u_copia_sint = n * self.COSTO_COPIA
            u_zip = archivos_zip_estimados * self.COSTO_ZIP
            self.unidades_galeria = n * self.COSTO_GALERIA
            self.unidades_totales = u_copia_orig + u_gen + u_copia_sint + u_zip + self.unidades_galeria

            # 1. Copiar dataset original preservando estructura multiclase
            self.senal_log.emit("[Aumento] Copiando estructura original al dataset ampliado...")
            gestor.copiar_dataset_original(
                pendientes, callback_progreso=self._crear_callback(self.COSTO_COPIA, "Copiando originales")
            )
            self.unidades_hechas = u_copia_orig

            # 2 y 3. Generar sintéticas y copiarlas al dataset ampliado
            cb_gen = self._crear_callback(costo_gen, "Generando imágenes")
            base_copia = u_copia_orig + u_gen

            def cb_copia(actual, total):
                self.unidades_hechas = base_copia + actual * self.COSTO_COPIA
                self._emitir(f"Integrando al dataset ({actual}/{total})")

            gestor.integrar_imagenes_sinteticas_clase(
                entrenador_fastgan=self.fastgan,
                cantidad=n,
                clase_nombre=self.clase_seleccionada,
                callback_progreso=cb_gen,
                callback_copia=cb_copia
            )
            self.unidades_hechas = base_copia + u_copia_sint

            dir_sinteticas_clase = os.path.join(str(gestor.ruta_sinteticas), self.clase_seleccionada)

            # 4. Exportar paquete ZIP (ajustando la estimación al número real de archivos)
            self.senal_log.emit("[Aumento] Empacando Dataset Ampliado...")
            archivos_zip_reales = gestor.contar_archivos_ampliado()
            self.unidades_totales += (archivos_zip_reales - archivos_zip_estimados) * self.COSTO_ZIP
            ruta_zip = gestor.exportar_zip_dataset_ampliado(
                callback_progreso=self._crear_callback(self.COSTO_ZIP, "Empaquetando ZIP")
            )

            # 5. La carga de la galería se hace en la interfaz y completa el 100%
            self.unidades_hechas = self.unidades_totales - self.unidades_galeria
            self._emitir("Cargando galería")
            self.completado.emit(str(gestor.ruta_ampliado), str(ruta_zip), dir_sinteticas_clase)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Error en Aumento de Dataset: {str(e)}")

    def porcentaje_inicio_galeria(self):
        """Porcentaje de la barra en el que comienza la carga de la galería en la interfaz."""
        return int(100.0 * (self.unidades_totales - self.unidades_galeria) / max(1.0, self.unidades_totales))


class SistemaGania(QObject):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
