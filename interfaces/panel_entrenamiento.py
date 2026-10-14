# interfaces/panel_entrenamiento.py
import os
import math
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
                              QComboBox, QPushButton, QDoubleSpinBox, QCheckBox,
                              QProgressBar, QSpinBox, QFileDialog, QLineEdit, QMessageBox)
from PyQt5.QtCore import pyqtSignal, Qt, QUrl
from PyQt5.QtGui import QDesktopServices
import qtawesome as qta

from utils.gestor_logs import obtener_logger
from utils.dispositivo import detectar_gpu, obtener_dispositivo
from utils.cargador_datos import (extraer_zip_dataset, descargar_extraer_url,
                                  preparar_y_analizar_dataset, obtener_clases_disponibles)

logger = obtener_logger("UI.Entrenamiento")

def icono_blanco(nombre):
    """Crea un icono de QtAwesome en color blanco puro constante para todos los estados."""
    return qta.icon(nombre, color="white", color_active="white", color_disabled="white")


class PanelConfiguracionDataset(QGroupBox):
    def __init__(self):
        super().__init__("1. Dataset & Selección de Clase")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.ruta_dataset_resuelta = None
        self.ruta_dataset_preparada = None
        self.clases_detectadas = {}
        self.resumen_preparacion = {}

        # 1. Selector de Método de Carga
        layout.addWidget(QLabel("Método de Carga de Dataset:"))
        self.selector_origen = QComboBox()
        self.selector_origen.addItems([
            "Carpeta Local",
            "Archivo Comprimido (ZIP/TGZ)",
            "Descargar desde URL / GitHub"
        ])
        self.selector_origen.setFixedHeight(35)
        layout.addWidget(self.selector_origen)

        # Controles según el origen
        self.btn_carpeta_local = QPushButton(" Seleccionar Carpeta del Dataset")
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

        self.btn_descargar_url = QPushButton(" Descargar & Extraer")
        self.btn_descargar_url.setIcon(icono_blanco("fa5s.download"))
        self.btn_descargar_url.clicked.connect(self._descargar_url)
        layout_url.addWidget(self.btn_descargar_url)
        self.container_url.setVisible(False)
        layout.addWidget(self.container_url)

        # 2. Resumen de Preparación de Dataset
        self.lbl_resumen_preparacion = QLabel("Formato: Pendiente de análisis")
        self.lbl_resumen_preparacion.setWordWrap(True)
        self.lbl_resumen_preparacion.setStyleSheet("font-size: 11px; color: #60A5FA; background: rgba(59, 130, 246, 0.1); padding: 8px; border-radius: 8px;")
        layout.addWidget(self.lbl_resumen_preparacion)

        # 3. Selector de Clase para FastGAN
        layout.addWidget(QLabel("Clases Detectadas (Selecciona una para FastGAN):"))
        self.selector_clase = QComboBox()
        self.selector_clase.setFixedHeight(35)
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

    def _seleccionar_carpeta(self):
        ruta = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta del Dataset")
        if ruta:
            self.ruta_dataset_resuelta = ruta
            self.lbl_ruta_estado.setText(f"Carpeta: ...{ruta[-35:]}")
            self.procesar_y_analizar_dataset()

    def _seleccionar_zip(self):
        ruta_zip, _ = QFileDialog.getOpenFileName(self, "Seleccionar Dataset Comprimido", "", "Archivos Comprimidos (*.zip *.tgz *.tar.gz *.tar)")
        if ruta_zip:
            try:
                ruta_extraida = extraer_zip_dataset(ruta_zip)
                self.ruta_dataset_resuelta = ruta_extraida
                self.lbl_ruta_estado.setText(f"Extraído en: ...{ruta_extraida[-30:]}")
                QMessageBox.information(self, "Descompresión Exitosa", f"Dataset descompuesto correctamente en:\n{ruta_extraida}")
                self.procesar_y_analizar_dataset()
            except Exception as e:
                QMessageBox.critical(self, "Error al Descomprimir", f"No se pudo descomprimir el archivo:\n{e}")

    def _descargar_url(self):
        url = self.input_url.text().strip()
        if not url.startswith("http"):
            QMessageBox.warning(self, "URL Inválida", "Ingresa una URL válida que empiece con http:// o https://")
            return

        try:
            self.btn_descargar_url.setEnabled(False)
            self.lbl_ruta_estado.setText("Descargando y descomprimiendo dataset...")
            ruta_descargada = descargar_extraer_url(url, callback_log=lambda m: logger.info(m))
            self.ruta_dataset_resuelta = ruta_descargada
            self.lbl_ruta_estado.setText(f"URL Extraída en: ...{ruta_descargada[-30:]}")
            QMessageBox.information(self, "Descarga Completada", f"Dataset descargado y listo en:\n{ruta_descargada}")
            self.procesar_y_analizar_dataset()
        except Exception as e:
            QMessageBox.critical(self, "Error Descarga", f"No se pudo descargar o descomprimir el dataset:\n{e}")
        finally:
            self.btn_descargar_url.setEnabled(True)

    def procesar_y_analizar_dataset(self):
        """Ejecuta el módulo universal de preparación y actualiza la UI."""
        if not self.ruta_dataset_resuelta or not os.path.exists(self.ruta_dataset_resuelta):
            return

        self.ruta_dataset_preparada, self.resumen_preparacion = preparar_y_analizar_dataset(self.ruta_dataset_resuelta)

        # Formatear Resumen Visual
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
            f"• Train: {n_train} | Val: {n_val} | Test: {n_test}<br>"
            f"• Archivos inválidos/omitidos: {inv}"
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

    # Aliases de señales para retrocompatibilidad
    senal_iniciar_dcgan = senal_iniciar_fastgan
    senal_detener_dcgan = senal_detener_fastgan

    def __init__(self):
        super().__init__("2. Entrenamiento FastGAN por Clase (ICLR 2021)")
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Learning Rate
        layout.addWidget(QLabel("Tasa de Aprendizaje (LR)"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(6)
        self.spin_lr.setRange(0.000001, 0.01)
        self.spin_lr.setValue(0.0002)
        self.spin_lr.setFixedHeight(35)
        layout.addWidget(self.spin_lr)

        # Batch Size
        layout.addWidget(QLabel("Tamaño de Lote (Batch Size)"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(8, 256)
        self.spin_batch.setValue(32)
        self.spin_batch.setFixedHeight(35)
        layout.addWidget(self.spin_batch)

        # Épocas
        layout.addWidget(QLabel("Épocas de Entrenamiento"))
        self.spin_epocas = QSpinBox()
        self.spin_epocas.setRange(1, 500)
        self.spin_epocas.setValue(25)
        self.spin_epocas.setFixedHeight(35)
        layout.addWidget(self.spin_epocas)

        # Dispositivo de entrenamiento (GPU NVIDIA / CPU)
        self.chk_cuda = QCheckBox("Entrenar con GPU (CUDA)")
        self.lbl_estado_gpu = QLabel()
        self.lbl_estado_gpu.setWordWrap(True)
        layout.addWidget(self.chk_cuda)
        layout.addWidget(self.lbl_estado_gpu)
        self._configurar_gpu()

        # Progreso
        self.barra_progreso = QProgressBar()
        self.barra_progreso.setFixedHeight(20)
        self.barra_progreso.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.barra_progreso)

        # Botones
        btn_layout = QHBoxLayout()
        self.btn_iniciar = QPushButton(" Entrenar FastGAN")
        self.btn_iniciar.setIcon(icono_blanco("fa5s.play"))
        self.btn_iniciar.setFixedHeight(40)
        self.btn_iniciar.clicked.connect(lambda: self.senal_iniciar_fastgan.emit())

        self.btn_detener = QPushButton(" Detener")
        self.btn_detener.setIcon(icono_blanco("fa5s.stop"))
        self.btn_detener.setFixedHeight(40)
        self.btn_detener.setEnabled(False)
        self.btn_detener.clicked.connect(lambda: self.senal_detener_fastgan.emit())

        btn_layout.addWidget(self.btn_iniciar)
        btn_layout.addWidget(self.btn_detener)
        layout.addLayout(btn_layout)

    def _configurar_gpu(self):
        """Detecta CUDA: si hay GPU usable se activa la casilla; si no, queda desactivada."""
        info = detectar_gpu()
        self._gpu_disponible = info["disponible"]

        self.chk_cuda.setEnabled(self._gpu_disponible)
        self.chk_cuda.setChecked(self._gpu_disponible)
        self.chk_cuda.setToolTip(info["detalle"])
        self.lbl_estado_gpu.setToolTip(info["detalle"])

        if self._gpu_disponible:
            icono, color, fondo = "✔", "#10B981", "rgba(16, 185, 129, 0.12)"
        elif info["nombre"]:
            icono, color, fondo = "⚠", "#F59E0B", "rgba(245, 158, 11, 0.12)"   # GPU sin PyTorch-CUDA
        else:
            icono, color, fondo = "✖", "#EF4444", "rgba(239, 68, 68, 0.12)"
        self.lbl_estado_gpu.setText(f"{icono} {info['mensaje']}")
        self.lbl_estado_gpu.setStyleSheet(
            f"font-size: 11px; color: {color}; background: {fondo}; padding: 6px; border-radius: 6px;"
        )
        logger.info(f"[GPU] {info['mensaje']} | {info['detalle']}")

    def obtener_dispositivo(self):
        """torch.device a usar según la casilla (CUDA solo si está disponible y marcada)."""
        return obtener_dispositivo(usar_gpu=self.chk_cuda.isChecked())

    def set_entrenando(self, entrenando):
        """Bloquea el cambio de dispositivo mientras hay un entrenamiento en curso."""
        self.chk_cuda.setEnabled(self._gpu_disponible and not entrenando)


# Alias para retrocompatibilidad
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
        self.lbl_estado_modelo = QLabel("Modelo FastGAN: No entrenado")
        self.lbl_estado_modelo.setStyleSheet("font-weight: bold; color: #EF4444; font-size: 12px;")
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
        self.combo_porcentaje.currentTextChanged.connect(self._recalcular_sinteticas)
        layout.addWidget(self.combo_porcentaje)

        # 3. Cantidad Sintética Calculada
        layout.addWidget(QLabel("Imágenes sintéticas a generar:"))
        self.lbl_cant_sintetica = QLabel("0 imágenes")
        self.lbl_cant_sintetica.setWordWrap(True)
        self.lbl_cant_sintetica.setStyleSheet("font-size: 14px; font-weight: bold; color: #10B981;")
        layout.addWidget(self.lbl_cant_sintetica)

        # Botón de Generación sin Reentrenamiento
        self.btn_generar_aumento = QPushButton(" Generar Imágenes Sintéticas")
        self.btn_generar_aumento.setIcon(icono_blanco("fa5s.magic"))
        self.btn_generar_aumento.setFixedHeight(42)
        self.btn_generar_aumento.setEnabled(False)
        self.btn_generar_aumento.clicked.connect(self._al_generar)
        layout.addWidget(self.btn_generar_aumento)

        # Botones de Explorador de Archivos
        self.btn_abrir_sinteticas = QPushButton("Abrir Carpeta de Sintéticas")
        self.btn_abrir_sinteticas.setIcon(icono_blanco("fa5s.folder-open"))
        self.btn_abrir_sinteticas.setFixedHeight(36)
        self.btn_abrir_sinteticas.clicked.connect(self._abrir_folder_sinteticas)
        layout.addWidget(self.btn_abrir_sinteticas)

        self.btn_abrir_ampliado = QPushButton("Abrir Dataset Ampliado")
        self.btn_abrir_ampliado.setIcon(icono_blanco("fa5s.folder-open"))
        self.btn_abrir_ampliado.setFixedHeight(36)
        self.btn_abrir_ampliado.clicked.connect(self._abrir_folder_ampliado)
        layout.addWidget(self.btn_abrir_ampliado)

    def actualizar_conteo_original(self, cantidad):
        self.cant_original = max(0, cantidad)
        self.lbl_cant_original.setText(f"{self.cant_original} imágenes")
        self._recalcular_sinteticas()

    def set_estado_modelo_listo(self, listo=True):
        if listo:
            self.lbl_estado_modelo.setText("Modelo FastGAN: Entrenado y listo ✓")
            self.lbl_estado_modelo.setStyleSheet("font-weight: bold; color: #10B981; font-size: 12px;")
            self.btn_generar_aumento.setEnabled(True)
        else:
            self.lbl_estado_modelo.setText("Modelo FastGAN: No entrenado")
            self.lbl_estado_modelo.setStyleSheet("font-weight: bold; color: #EF4444; font-size: 12px;")
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
        self.senal_generar_aumento.emit(self.cant_sintetica_calculada)

    def _abrir_folder_sinteticas(self):
        os.makedirs(self.ruta_sinteticas, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.ruta_sinteticas)))

    def _abrir_folder_ampliado(self):
        os.makedirs(self.ruta_ampliado, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.ruta_ampliado)))


class PanelControlClasificador(QGroupBox):
    """Panel de Clasificación (Oculto temporalmente)."""
    senal_iniciar_clasificacion = pyqtSignal(str, int, float)

    def __init__(self):
        super().__init__("4. Clasificación Comparativa")
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(QLabel("Arquitectura del Clasificador:"))
        self.selector_modelo = QComboBox()
        self.selector_modelo.addItems(["ResNet18", "CNN_Simple"])
        self.selector_modelo.setFixedHeight(35)
        layout.addWidget(self.selector_modelo)

        layout.addWidget(QLabel("Épocas Clasificador:"))
        self.spin_epocas_clf = QSpinBox()
        self.spin_epocas_clf.setRange(1, 100)
        self.spin_epocas_clf.setValue(10)
        self.spin_epocas_clf.setFixedHeight(35)
        layout.addWidget(self.spin_epocas_clf)

        self.btn_clasificar = QPushButton(" Evaluar Clasificación")
        self.btn_clasificar.setIcon(icono_blanco("fa5s.chart-line"))
        self.btn_clasificar.setFixedHeight(45)
        self.btn_clasificar.setEnabled(False)
        self.btn_clasificar.clicked.connect(self._al_clasificar)
        layout.addWidget(self.btn_clasificar)

    def _al_clasificar(self):
        self.senal_iniciar_clasificacion.emit(
            self.selector_modelo.currentText(),
            self.spin_epocas_clf.value(),
            0.001
        )
