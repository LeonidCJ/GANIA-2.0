# interfaces/panel_entrenamiento.py
import os
import math
import torch
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
                              QComboBox, QPushButton, QDoubleSpinBox, QCheckBox,
                              QProgressBar, QSpinBox, QFileDialog, QLineEdit,
                              QMessageBox, QSizePolicy)
import time
from PyQt5.QtCore import pyqtSignal, Qt, QUrl, QThread
from PyQt5.QtGui import QDesktopServices
import qtawesome as qta

from interfaces.controles_personalizados import NotaIcono
from utils.gestor_logs import obtener_logger
from utils.dispositivo import detectar_gpu
from utils.cargador_datos import (extraer_zip_dataset, descargar_extraer_url,
                                  preparar_y_analizar_dataset, obtener_clases_disponibles)

logger = obtener_logger("UI.Entrenamiento")


def icono_blanco(nombre):
    """Crea un icono de QtAwesome en color blanco puro constante para todos los estados."""
    return qta.icon(nombre, color="white", color_active="white", color_disabled="white")


def _ajustar_combo(combo):
    """Evita que textos largos en un QComboBox ensanchen el panel o generen scroll horizontal."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(10)
    combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


class WorkerCargaDataset(QThread):
    """
    Hilo en segundo plano para descomprimir, descargar y analizar datasets
    sin congelar ni bloquear la interfaz gráfica principal.
    """
    progreso_estado = pyqtSignal(str)
    completado = pyqtSignal(str, str, dict)  # (ruta_resuelta, ruta_preparada, resumen)
    error = pyqtSignal(str)

    def __init__(self, modo, origen):
        super().__init__()
        self.modo = modo      # 'carpeta' | 'zip' | 'url'
        self.origen = origen

    def run(self):
        try:
            if self.modo == 'zip':
                self.progreso_estado.emit("Extrayendo archivo comprimido en segundo plano...")
                ruta_resuelta = extraer_zip_dataset(self.origen)
            elif self.modo == 'url':
                self.progreso_estado.emit("Descargando y descomprimiendo dataset...")
                ruta_resuelta = descargar_extraer_url(self.origen, callback_log=lambda m: logger.info(m))
            else:
                ruta_resuelta = self.origen

            self.progreso_estado.emit("Analizando clases e imágenes del dataset...")
            ruta_preparada, resumen = preparar_y_analizar_dataset(ruta_resuelta)
            self.completado.emit(ruta_resuelta, ruta_preparada, resumen)
        except Exception as e:
            self.error.emit(str(e))


class PanelConfiguracionDataset(QGroupBox):
    def __init__(self):
        super().__init__("1. Dataset y Selección de Clase")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.ruta_dataset_resuelta = None
        self.ruta_dataset_preparada = None
        self.clases_detectadas = {}
        self.resumen_preparacion = {}
        self.worker_carga = None

        # 1. Selector de Método de Carga
        layout.addWidget(QLabel("Método de Carga de Dataset:"))
        self.selector_origen = QComboBox()
        self.selector_origen.addItems([
            "Carpeta Local",
            "Archivo Comprimido (ZIP/TGZ)",
            "Descargar desde URL / GitHub"
        ])
        self.selector_origen.setFixedHeight(35)
        _ajustar_combo(self.selector_origen)
        layout.addWidget(self.selector_origen)

        # Controles según el origen
        self.btn_carpeta_local = QPushButton(" Seleccionar Carpeta")
        self.btn_carpeta_local.setIcon(icono_blanco("fa5s.folder-open"))
        self.btn_carpeta_local.clicked.connect(self._seleccionar_carpeta)
        layout.addWidget(self.btn_carpeta_local)

        self.btn_zip_local = QPushButton(" Seleccionar Archivo .ZIP / .TGZ")
        self.btn_zip_local.setIcon(icono_blanco("fa5s.file-archive"))
        self.btn_zip_local.setVisible(False)
        self.btn_zip_local.clicked.connect(self._seleccionar_zip)
        layout.addWidget(self.btn_zip_local)

        self.container_url = QGroupBox("Descarga por URL Directa / GitHub")
        layout_url = QVBoxLayout(self.container_url)
        self.input_url = QLineEdit()
        self.input_url.setPlaceholderText("https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz")
        layout_url.addWidget(self.input_url)

        self.btn_descargar_url = QPushButton(" Descargar y Extraer")
        self.btn_descargar_url.setIcon(icono_blanco("fa5s.download"))
        self.btn_descargar_url.clicked.connect(self._descargar_url)
        layout_url.addWidget(self.btn_descargar_url)
        self.container_url.setVisible(False)
        layout.addWidget(self.container_url)

        # 2. Resumen de Preparación de Dataset
        self.lbl_resumen_preparacion = QLabel("Formato: Pendiente de análisis")
        self.lbl_resumen_preparacion.setWordWrap(True)
        self.lbl_resumen_preparacion.setStyleSheet(
            "font-size: 11px; color: #60A5FA; background: rgba(59, 130, 246, 0.1); "
            "padding: 8px; border-radius: 8px;"
        )
        layout.addWidget(self.lbl_resumen_preparacion)

        # 3. Selector de Clase para FastGAN
        layout.addWidget(QLabel("Clases Detectadas (Selecciona una):"))
        self.selector_clase = QComboBox()
        self.selector_clase.setFixedHeight(35)
        _ajustar_combo(self.selector_clase)
        self.selector_clase.currentIndexChanged.connect(self._notificar_cambio)
        layout.addWidget(self.selector_clase)

        # Estado de la Ruta
        self.lbl_ruta_estado = QLabel("Estado: Ningún dataset cargado")
        self.lbl_ruta_estado.setWordWrap(True)
        self.lbl_ruta_estado.setStyleSheet("font-size: 11px; color: #94A3B8; font-style: italic;")
        layout.addWidget(self.lbl_ruta_estado)

        self.selector_origen.currentTextChanged.connect(self._actualizar_visibilidad)
        self._actualizar_visibilidad(self.selector_origen.currentText())

    def _actualizar_visibilidad(self, texto):
        self.btn_carpeta_local.setVisible(texto == "Carpeta Local")
        self.btn_zip_local.setVisible(texto == "Archivo Comprimido (ZIP/TGZ)")
        self.container_url.setVisible(texto == "Descargar desde URL / GitHub")

    def _set_controles_carga_habilitados(self, habilitado):
        self.btn_carpeta_local.setEnabled(habilitado)
        self.btn_zip_local.setEnabled(habilitado)
        self.btn_descargar_url.setEnabled(habilitado)
        self.selector_origen.setEnabled(habilitado)
        self.selector_clase.setEnabled(habilitado)

    def _iniciar_worker_carga(self, modo, origen):
        if self.worker_carga is not None and self.worker_carga.isRunning():
            return
        self._set_controles_carga_habilitados(False)
        self.lbl_resumen_preparacion.setText("Procesando dataset en segundo plano...")
        self.worker_carga = WorkerCargaDataset(modo, origen)
        self.worker_carga.progreso_estado.connect(self.lbl_ruta_estado.setText)
        self.worker_carga.completado.connect(self._on_carga_completada)
        self.worker_carga.error.connect(self._on_carga_error)
        self.worker_carga.start()

    def _on_carga_completada(self, ruta_resuelta, ruta_preparada, resumen):
        self._set_controles_carga_habilitados(True)
        self.ruta_dataset_resuelta = ruta_resuelta
        self.ruta_dataset_preparada = ruta_preparada
        self.resumen_preparacion = resumen
        self.lbl_ruta_estado.setText(f"Carpeta: ...{ruta_resuelta[-35:]}")
        self._mostrar_resumen_preparacion()

    def _on_carga_error(self, msj_error):
        self._set_controles_carga_habilitados(True)
        self.lbl_ruta_estado.setText("Estado: Error al cargar dataset")
        self.lbl_resumen_preparacion.setText("Formato: Error en el análisis")
        QMessageBox.critical(self, "Error al Cargar Dataset", f"No se pudo procesar el dataset:\n{msj_error}")

    def _seleccionar_carpeta(self):
        ruta = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta del Dataset")
        if ruta:
            self._iniciar_worker_carga('carpeta', ruta)

    def _seleccionar_zip(self):
        ruta_zip, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Dataset Comprimido", "",
            "Archivos Comprimidos (*.zip *.tgz *.tar.gz *.tar)"
        )
        if ruta_zip:
            self._iniciar_worker_carga('zip', ruta_zip)

    def _descargar_url(self):
        url = self.input_url.text().strip()
        if not url.startswith("http"):
            QMessageBox.warning(self, "URL Inválida", "Ingresa una URL válida que comience con http:// o https://")
            return
        self._iniciar_worker_carga('url', url)

    def procesar_y_analizar_dataset(self):
        """Método síncrono de compatibilidad para llamadas directas."""
        if not self.ruta_dataset_resuelta or not os.path.exists(self.ruta_dataset_resuelta):
            return
        self.ruta_dataset_preparada, self.resumen_preparacion = preparar_y_analizar_dataset(self.ruta_dataset_resuelta)
        self._mostrar_resumen_preparacion()

    def _mostrar_resumen_preparacion(self):
        fmt = self.resumen_preparacion.get("formato", "Desconocido")
        train_map = self.resumen_preparacion.get("train", {})
        val_map = self.resumen_preparacion.get("val", {})
        test_map = self.resumen_preparacion.get("test", {})
        inv = self.resumen_preparacion.get("invalidas", 0)

        n_clases = len(train_map)
        n_train = sum(train_map.values())
        n_val = sum(val_map.values())
        n_test = sum(test_map.values())

        txt_resumen = (
            f"<b>Estructura:</b> {fmt}<br>"
            f"<b>Clases Totales:</b> {n_clases} | <b>Imágenes Válidas:</b> {n_train + n_val + n_test}<br>"
            f"- Train: {n_train} | Val: {n_val} | Test: {n_test}<br>"
            f"- Archivos inválidos/omitidos: {inv}"
        )
        self.lbl_resumen_preparacion.setText(txt_resumen)
        self.actualizar_clases_detectadas()

    def actualizar_clases_detectadas(self):
        """Puebla el selector de clases con las clases analizadas."""
        self.selector_clase.blockSignals(True)
        self.selector_clase.clear()

        train_map = self.resumen_preparacion.get("train", {})
        if not train_map:
            train_map = self.resumen_preparacion.get("total_por_clase", {})

        self.clases_detectadas = train_map

        if self.clases_detectadas:
            for clase_nom, cant in self.clases_detectadas.items():
                self.selector_clase.addItem(f"{clase_nom} ({cant} imágenes)", userData=clase_nom)
        else:
            self.selector_clase.addItem("Sin clases detectadas", userData=None)

        self.selector_clase.blockSignals(False)
        self._notificar_cambio()

    def obtener_ruta_dataset(self):
        return self.ruta_dataset_preparada or self.ruta_dataset_resuelta

    def obtener_clase_seleccionada(self):
        data = self.selector_clase.currentData()
        if data:
            return data
        txt = self.selector_clase.currentText()
        if txt and "(" in txt:
            return txt.split("(")[0].strip()
        return txt if txt else "clase_01"

    def obtener_cantidad_original_clase(self):
        clase_sel = self.obtener_clase_seleccionada()
        return self.clases_detectadas.get(clase_sel, 0)

    def _notificar_cambio(self):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "validar_inicio_entrenamiento"):
                parent.validar_inicio_entrenamiento()
                break
            parent = parent.parent()


class PanelControlFastGAN(QGroupBox):
    senal_iniciar_fastgan = pyqtSignal()
    senal_detener_fastgan = pyqtSignal()

    def __init__(self):
        super().__init__("2. Entrenamiento FastGAN por Clase (ICLR 2021)")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Fila compacta: Aceleración GPU + Chip de estado a la derecha
        self.info_gpu = detectar_gpu()
        es_gpu = self.info_gpu["disponible"]

        fila_hw = QHBoxLayout()
        fila_hw.setSpacing(8)
        self.chk_cuda = QCheckBox("Usar GPU (CUDA)")
        self.chk_cuda.setChecked(es_gpu)
        self.chk_cuda.setEnabled(es_gpu)
        self.chk_cuda.setToolTip(self.info_gpu["detalle"])
        self.chk_cuda.toggled.connect(self._al_cambiar_dispositivo_o_resolucion)

        self.lbl_dispositivo = NotaIcono(compacta=True, tam_icono=11)
        self.lbl_dispositivo.setToolTip(f"{self.info_gpu['mensaje']}\n{self.info_gpu['detalle']}")

        fila_hw.addWidget(self.chk_cuda)
        fila_hw.addStretch()
        fila_hw.addWidget(self.lbl_dispositivo)
        layout.addLayout(fila_hw)

        # 2. Selector de Resolución limpio + nota breve contextual
        layout.addWidget(QLabel("Resolución de salida"))
        self.combo_resolucion = QComboBox()
        self.combo_resolucion.addItem("64 x 64 px  -  Rápida (CPU)", userData=64)
        self.combo_resolucion.addItem("128 x 128 px  -  Media (GPU)", userData=128)
        self.combo_resolucion.addItem("256 x 256 px  -  Alta (GPU)", userData=256)
        self.combo_resolucion.setCurrentIndex(1 if es_gpu else 0)
        self.combo_resolucion.setFixedHeight(35)
        _ajustar_combo(self.combo_resolucion)
        self.combo_resolucion.currentIndexChanged.connect(self._al_cambiar_dispositivo_o_resolucion)
        layout.addWidget(self.combo_resolucion)

        self.lbl_aviso_resolucion = NotaIcono()
        layout.addWidget(self.lbl_aviso_resolucion)

        # 3. Tasa de Aprendizaje (LR)
        layout.addWidget(QLabel("Tasa de Aprendizaje (LR)"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(6)
        self.spin_lr.setRange(0.000001, 0.01)
        self.spin_lr.setValue(0.0002)
        self.spin_lr.setFixedHeight(35)
        layout.addWidget(self.spin_lr)

        # 4. Lote y Épocas en 2 columnas simétricas
        fila_params = QHBoxLayout()
        fila_params.setSpacing(10)

        col_batch = QVBoxLayout()
        col_batch.setSpacing(4)
        col_batch.addWidget(QLabel("Tamaño de Lote"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(4, 256)
        self.spin_batch.setValue(32 if es_gpu else 8)
        self.spin_batch.setFixedHeight(35)
        self.spin_batch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        col_batch.addWidget(self.spin_batch)

        col_epocas = QVBoxLayout()
        col_epocas.setSpacing(4)
        col_epocas.addWidget(QLabel("Épocas"))
        self.spin_epocas = QSpinBox()
        self.spin_epocas.setRange(1, 100000)
        self.spin_epocas.setSingleStep(50)
        self.spin_epocas.setValue(25)
        self.spin_epocas.setFixedHeight(35)
        self.spin_epocas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        col_epocas.addWidget(self.spin_epocas)

        fila_params.addLayout(col_batch, 1)
        fila_params.addLayout(col_epocas, 1)
        layout.addLayout(fila_params)

        # 5. Progreso
        self.barra_progreso = QProgressBar()
        self.barra_progreso.setFixedHeight(18)
        self.barra_progreso.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.barra_progreso)

        # 6. Botones de Acción (Simetría 1:1 exacta)
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        self.btn_iniciar = QPushButton(" Entrenar")
        self.btn_iniciar.setIcon(icono_blanco("fa5s.play"))
        self.btn_iniciar.setFixedHeight(40)
        self.btn_iniciar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_iniciar.clicked.connect(lambda: self.senal_iniciar_fastgan.emit())

        self.btn_detener = QPushButton(" Detener")
        self.btn_detener.setIcon(icono_blanco("fa5s.stop"))
        self.btn_detener.setFixedHeight(40)
        self.btn_detener.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_detener.setEnabled(False)
        self.btn_detener.clicked.connect(lambda: self.senal_detener_fastgan.emit())

        btn_layout.addWidget(self.btn_iniciar, 1)
        btn_layout.addWidget(self.btn_detener, 1)
        layout.addLayout(btn_layout)

        self._al_cambiar_dispositivo_o_resolucion()

    def usar_gpu(self):
        return bool(self.info_gpu.get("disponible") and self.chk_cuda.isChecked())

    def obtener_resolucion(self):
        val = self.combo_resolucion.currentData()
        return int(val) if val else 64

    def _al_cambiar_dispositivo_o_resolucion(self):
        gpu_disp = self.info_gpu.get("disponible", False)
        usando_gpu = self.usar_gpu()
        res = self.obtener_resolucion()

        # 1. Chip compacto de estado de GPU / CPU
        if gpu_disp and usando_gpu:
            nombre_corto = (self.info_gpu.get("nombre") or "GPU").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
            self.lbl_dispositivo.establecer("fa5s.microchip", nombre_corto, color="#059669",
                                            fondo="rgba(16, 185, 129, 0.14)", negrita=True)
        elif gpu_disp and not usando_gpu:
            self.lbl_dispositivo.establecer("fa5s.power-off", "GPU desactivada", color="#D97706",
                                            fondo="rgba(245, 158, 11, 0.14)")
        else:
            self.lbl_dispositivo.establecer("fa5s.desktop", "Modo CPU", color="#64748B",
                                            fondo="rgba(100, 116, 139, 0.12)")

        # 2. Nota breve bajo el selector de resolución
        if usando_gpu:
            self.lbl_aviso_resolucion.establecer("fa5s.bolt", f"Aceleración GPU activa para {res}x{res} px.",
                                                 color="#059669")
        elif res == 64:
            self.lbl_aviso_resolucion.establecer("fa5s.info-circle", "Ligera para CPU. Menor nitidez al ampliar.",
                                                 color="#64748B")
        elif res == 128:
            self.lbl_aviso_resolucion.establecer("fa5s.exclamation-triangle",
                                                 "En CPU tarda ~4x más por época. Recomendado: GPU o Colab.",
                                                 color="#D97706", fondo="rgba(245, 158, 11, 0.12)")
        else:
            self.lbl_aviso_resolucion.establecer("fa5s.exclamation-circle",
                                                 "En CPU tarda ~16x más y consume mucha RAM. Usa GPU o Colab.",
                                                 color="#DC2626", fondo="rgba(239, 68, 68, 0.12)")


# Alias de retrocompatibilidad
PanelControlDCGAN = PanelControlFastGAN


class PanelAumentoDataset(QGroupBox):
    senal_generar_aumento = pyqtSignal(int)

    def __init__(self):
        super().__init__("3. Generación Sintética y Dataset Ampliado")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.cant_original = 0
        self.cant_sintetica_calculada = 0
        self.ruta_sinteticas = "./datos/datasets/synthetic"
        self.ruta_ampliado = "./datos/datasets/augmented"

        # Estado del Modelo Entrenado
        self.lbl_estado_modelo = NotaIcono(tam_icono=14)
        self.lbl_estado_modelo.establecer("fa5s.times-circle", "Modelo FastGAN: No entrenado",
                                          color="#EF4444", negrita=True, tam_fuente=12)
        layout.addWidget(self.lbl_estado_modelo)

        # 1. Conteo Original
        layout.addWidget(QLabel("Imágenes originales en la clase:"))
        self.lbl_cant_original = QLabel("0 imágenes")
        self.lbl_cant_original.setStyleSheet("font-size: 14px; font-weight: bold; color: #3B82F6;")
        layout.addWidget(self.lbl_cant_original)

        # 2. Selector de Porcentaje
        layout.addWidget(QLabel("Porcentaje de aumento respecto al original:"))
        self.combo_porcentaje = QComboBox()
        self.combo_porcentaje.addItems(["10%", "25%", "50%", "75%", "100%", "200%"])
        self.combo_porcentaje.setCurrentText("25%")
        self.combo_porcentaje.setFixedHeight(35)
        _ajustar_combo(self.combo_porcentaje)
        self.combo_porcentaje.currentTextChanged.connect(self._recalcular_sinteticas)
        layout.addWidget(self.combo_porcentaje)

        # 3. Cantidad Sintética Calculada
        layout.addWidget(QLabel("Imágenes sintéticas a generar:"))
        self.lbl_cant_sintetica = QLabel("0 imágenes")
        self.lbl_cant_sintetica.setWordWrap(True)
        self.lbl_cant_sintetica.setStyleSheet("font-size: 14px; font-weight: bold; color: #10B981;")
        layout.addWidget(self.lbl_cant_sintetica)

        # 4. Barra de Progreso de Generación Sintética
        layout.addWidget(QLabel("Progreso de Generación Sintética:"))
        self.barra_progreso_aumento = QProgressBar()
        self.barra_progreso_aumento.setFixedHeight(22)
        self.barra_progreso_aumento.setAlignment(Qt.AlignCenter)
        self.barra_progreso_aumento.setValue(0)
        layout.addWidget(self.barra_progreso_aumento)

        # Fase actual y tiempo restante estimado
        fila_estado = QHBoxLayout()
        fila_estado.setSpacing(8)
        self.lbl_fase_aumento = NotaIcono(compacta=True, tam_icono=11)
        self.lbl_fase_aumento.establecer("fa5s.hourglass-start", "En espera", color="#94A3B8")
        self.lbl_tiempo_aumento = NotaIcono(compacta=True, tam_icono=11)
        self.lbl_tiempo_aumento.setVisible(False)
        fila_estado.addWidget(self.lbl_fase_aumento, 1)
        fila_estado.addWidget(self.lbl_tiempo_aumento, 0)
        layout.addLayout(fila_estado)
        self._t_inicio_aumento = None

        # Botón de Generación sin Reentrenamiento
        self.btn_generar_aumento = QPushButton(" Generar Sintéticas")
        self.btn_generar_aumento.setIcon(icono_blanco("fa5s.magic"))
        self.btn_generar_aumento.setFixedHeight(42)
        self.btn_generar_aumento.setEnabled(False)
        self.btn_generar_aumento.clicked.connect(self._al_generar)
        layout.addWidget(self.btn_generar_aumento)

        # Botones de Explorador de Archivos
        self.btn_abrir_sinteticas = QPushButton("Abrir Carpeta Sintéticas")
        self.btn_abrir_sinteticas.setIcon(icono_blanco("fa5s.folder-open"))
        self.btn_abrir_sinteticas.setFixedHeight(36)
        self.btn_abrir_sinteticas.clicked.connect(self._abrir_folder_sinteticas)
        layout.addWidget(self.btn_abrir_sinteticas)

        self.btn_abrir_ampliado = QPushButton("Abrir Dataset Ampliado")
        self.btn_abrir_ampliado.setIcon(icono_blanco("fa5s.folder-open"))
        self.btn_abrir_ampliado.setFixedHeight(36)
        self.btn_abrir_ampliado.clicked.connect(self._abrir_folder_ampliado)
        layout.addWidget(self.btn_abrir_ampliado)

    @staticmethod
    def _formatear_tiempo(segundos):
        segundos = max(0, int(round(segundos)))
        m, s = divmod(segundos, 60)
        return f"{m}:{s:02d} min" if m else f"{s} s"

    def iniciar_progreso_aumento(self):
        self._t_inicio_aumento = time.time()
        self.barra_progreso_aumento.setValue(0)
        self.lbl_fase_aumento.establecer("fa5s.hourglass-start", "Preparando...", color="#3B82F6")
        self.lbl_tiempo_aumento.establecer("fa5s.clock", "Calculando...", color="#64748B")
        self.lbl_tiempo_aumento.setVisible(True)

    def actualizar_progreso_aumento(self, porcentaje, fase=""):
        """Recibe el porcentaje global real (0-100) y la fase actual; estima el tiempo restante."""
        porcentaje = int(max(0, min(100, porcentaje)))
        self.barra_progreso_aumento.setValue(porcentaje)
        if fase:
            self.lbl_fase_aumento.establecer("fa5s.hourglass-half", fase, color="#3B82F6")

        if self._t_inicio_aumento is not None and porcentaje >= 3:
            transcurrido = time.time() - self._t_inicio_aumento
            restante = transcurrido * (100 - porcentaje) / porcentaje
            self.lbl_tiempo_aumento.establecer("fa5s.clock", f"Restante: ~{self._formatear_tiempo(restante)}",
                                               color="#64748B")

    def finalizar_progreso_aumento(self, exito=True):
        self.barra_progreso_aumento.setValue(100 if exito else self.barra_progreso_aumento.value())
        total = time.time() - self._t_inicio_aumento if self._t_inicio_aumento else 0
        if exito:
            self.lbl_fase_aumento.establecer("fa5s.check-circle", "Completado", color="#10B981", negrita=True)
            self.lbl_tiempo_aumento.establecer("fa5s.stopwatch", f"Total: {self._formatear_tiempo(total)}",
                                               color="#10B981")
        else:
            self.lbl_fase_aumento.establecer("fa5s.times-circle", "Error en la generación", color="#EF4444",
                                             negrita=True)
            self.lbl_tiempo_aumento.setVisible(False)
        self._t_inicio_aumento = None

    def actualizar_conteo_original(self, cantidad):
        self.cant_original = max(0, cantidad)
        self.lbl_cant_original.setText(f"{self.cant_original} imágenes")
        self._recalcular_sinteticas()

    def set_estado_modelo_listo(self, listo=True):
        if listo:
            self.lbl_estado_modelo.establecer("fa5s.check-circle", "Modelo FastGAN: Entrenado y listo",
                                              color="#10B981", negrita=True, tam_fuente=12)
            self.btn_generar_aumento.setEnabled(True)
        else:
            self.lbl_estado_modelo.establecer("fa5s.times-circle", "Modelo FastGAN: No entrenado",
                                              color="#EF4444", negrita=True, tam_fuente=12)
            self.btn_generar_aumento.setEnabled(False)

    def _recalcular_sinteticas(self):
        txt_pct = self.combo_porcentaje.currentText().replace("%", "").strip()
        try:
            pct = int(txt_pct)
        except ValueError:
            pct = 25

        if self.cant_original > 0:
            self.cant_sintetica_calculada = math.ceil(self.cant_original * pct / 100.0)
        else:
            self.cant_sintetica_calculada = 0

        total_ampliado = self.cant_original + self.cant_sintetica_calculada
        self.lbl_cant_sintetica.setText(f"{self.cant_sintetica_calculada} imágenes\n(Total ampliado: {total_ampliado})")

    def _al_generar(self):
        if self.cant_sintetica_calculada <= 0:
            QMessageBox.warning(self, "Aumento Nulo", "La cantidad de imágenes sintéticas a generar debe ser mayor a 0.")
            return
        self.iniciar_progreso_aumento()
        self.senal_generar_aumento.emit(self.cant_sintetica_calculada)

    def _abrir_folder_sinteticas(self):
        os.makedirs(self.ruta_sinteticas, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.ruta_sinteticas)))

    def _abrir_folder_ampliado(self):
        os.makedirs(self.ruta_ampliado, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.ruta_ampliado)))
