# interfaces/panel_entrenamiento.py
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
                              QComboBox, QPushButton, QDoubleSpinBox,
                              QProgressBar, QSpinBox, QFileDialog, QCheckBox)
from PyQt5.QtCore import pyqtSignal, Qt
import qtawesome as qta
import os
from utils.gestor_logs import obtener_logger

logger = obtener_logger("UI.Entrenamiento")

class PanelModelo(QGroupBox):
    def __init__(self):
        super().__init__("1. Objetivo Creativo")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Rutas locales
        self.ruta_dataset_a = None
        self.ruta_dataset_b = None

        layout.addWidget(QLabel("¿Qué quieres que haga la IA?"))
        self.selector_modelo = QComboBox()
        self.selector_modelo.addItems([
            "Explorar Formas (DCGAN)",
            "Transformar Estilo (CycleGAN)"
        ])
        self.selector_modelo.setFixedHeight(35)
        layout.addWidget(self.selector_modelo)

        self.lbl_explicacion_modelo = QLabel("")
        self.lbl_explicacion_modelo.setWordWrap(True)
        self.lbl_explicacion_modelo.setStyleSheet("color: #94A3B8; font-size: 11px; font-style: italic; background: rgba(0,0,0,0.1); padding: 8px; border-radius: 8px;")
        layout.addWidget(self.lbl_explicacion_modelo)

        layout.addWidget(QLabel("¿Cuál es tu fuente de inspiración?"))
        self.selector_origen = QComboBox()
        self.selector_origen.setFixedHeight(35)
        layout.addWidget(self.selector_origen)

        self.lbl_explicacion_dataset = QLabel("")
        self.lbl_explicacion_dataset.setWordWrap(True)
        self.lbl_explicacion_dataset.setStyleSheet("color: #94A3B8; font-size: 11px; font-style: italic; padding: 5px;")
        layout.addWidget(self.lbl_explicacion_dataset)

        # Botones de Carpeta Local
        self.btn_carpeta_a = QPushButton(" Seleccionar Carpeta de Imágenes")
        self.btn_carpeta_a.setIcon(qta.icon("fa5s.folder-open", color="white"))
        self.btn_carpeta_a.setVisible(False)
        self.btn_carpeta_a.clicked.connect(self._seleccionar_a)
        layout.addWidget(self.btn_carpeta_a)

        self.btn_carpeta_b = QPushButton(" Seleccionar Carpeta de Estilo")
        self.btn_carpeta_b.setIcon(qta.icon("fa5s.folder-open", color="white"))
        self.btn_carpeta_b.setVisible(False)
        self.btn_carpeta_b.clicked.connect(self._seleccionar_b)
        layout.addWidget(self.btn_carpeta_b)

        self.lbl_ruta_a = QLabel("No seleccionada")
        self.lbl_ruta_a.setStyleSheet("font-size: 10px; color: #94A3B8;")
        self.lbl_ruta_a.setVisible(False)
        layout.addWidget(self.lbl_ruta_a)

        self.selector_modelo.currentIndexChanged.connect(self.actualizar_explicacion)
        self.selector_modelo.currentIndexChanged.connect(self.actualizar_opciones_datasets)
        self.selector_modelo.currentIndexChanged.connect(self._actualizar_visibilidad)
        self.selector_origen.currentIndexChanged.connect(self._actualizar_visibilidad)
        self.selector_origen.currentTextChanged.connect(self._verificar_aviso_local)

        self.actualizar_explicacion()
        self.actualizar_opciones_datasets()
        self._actualizar_visibilidad()

    def actualizar_opciones_datasets(self):
        modelo = self.selector_modelo.currentText()
        self.selector_origen.blockSignals(True)
        self.selector_origen.clear()
        if "DCGAN" in modelo:
            self.selector_origen.addItems(["CIFAR-10", "CelebA (Rostros)", "Carpeta Local"])
        else:
            self.selector_origen.addItems(["Apple ↔ Orange", "Van Gogh ↔ Photo", "Summer ↔ Winter", "Carpeta Local"])
        self.selector_origen.blockSignals(False)
        self.actualizar_explicacion_dataset()

    def actualizar_explicacion(self):
        descripciones = {
            "Explorar Formas (DCGAN)": "Crea bocetos totalmente nuevos para formas desde el lienzo en blanco basándose en patrones aprendidos.",
            "Transformar Estilo (CycleGAN)": "Toma tus imágenes y cámbiales el 'vibe' o la textura para explorar variantes estéticas manteniendo la estructura original."
        }
        modelo = self.selector_modelo.currentText()
        self.lbl_explicacion_modelo.setText(descripciones.get(modelo, ""))

    def actualizar_explicacion_dataset(self):
        descripciones = {
            "CIFAR-10": "Muestras de objetos reales (autos, animales, barcos).",
            "CelebA (Rostros)": "Más de 200,000 rostros para entrenamiento de alta fidelidad.",
            "Apple ↔ Orange": "Ideal para estudiar texturas y transformaciones orgánicas.",
            "Van Gogh ↔ Photo": "El estándar para transferencia de estilo artístico.",
            "Summer ↔ Winter": "Cambios de atmósfera y clima en paisajes.",
            "Carpeta Local": "Usa tus propios archivos de imagen desde el disco."
        }
        dataset = self.selector_origen.currentText()
        if dataset:
            self.lbl_explicacion_dataset.setText(descripciones.get(dataset, ""))

    def _verificar_aviso_local(self, texto):
        if texto == "Carpeta Local":
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Requisitos de Inspiración Propia",
                "¡Excelente elección! Para que la IA aprenda bien de tus archivos, asegúrate de cumplir lo siguiente:\n\n"
                "• Cantidad: Al menos 100 imágenes (recomendado 500+).\n"
                "• Formato: Archivos .jpg o .png legibles.\n"
                "• Proporción: Se recomienda que las imágenes sean cuadradas.\n"
                "• Contenido: Trata de que las imágenes tengan un estilo visual similar.")

    def _actualizar_visibilidad(self):
        modelo = self.selector_modelo.currentText()
        origen = self.selector_origen.currentText()
        es_local = (origen == "Carpeta Local")
        es_cycle = ("CycleGAN" in modelo)

        self.btn_carpeta_a.setVisible(es_local)
        self.lbl_ruta_a.setVisible(es_local)
        self.btn_carpeta_b.setVisible(es_local and es_cycle)

        self._notificar_cambio()

    def _seleccionar_a(self):
        ruta = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Imágenes")
        if ruta:
            self.ruta_dataset_a = ruta
            self.lbl_ruta_a.setText(f"Ruta: ...{ruta[-30:]}")
            self._notificar_cambio()

    def _seleccionar_b(self):
        ruta = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Estilo")
        if ruta:
            self.ruta_dataset_b = ruta
            self._notificar_cambio()

    def _notificar_cambio(self):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "validar_inicio_entrenamiento"):
                parent.validar_inicio_entrenamiento()
                break
            parent = parent.parent()


class PanelEntrenamiento(QGroupBox):
    senal_iniciar = pyqtSignal()
    senal_detener = pyqtSignal()

    def __init__(self):
        super().__init__("2. Control de Evolución")
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 1. Learning Rate
        layout.addWidget(QLabel("Velocidad de Evolución (Learning Rate)"))
        self.spin_tasa_aprendizaje = QDoubleSpinBox()
        self.spin_tasa_aprendizaje.setDecimals(6)
        self.spin_tasa_aprendizaje.setRange(0.000001, 0.01)
        self.spin_tasa_aprendizaje.setSingleStep(0.0001)
        self.spin_tasa_aprendizaje.setValue(0.0002)
        self.spin_tasa_aprendizaje.setFixedHeight(35)
        layout.addWidget(self.spin_tasa_aprendizaje)

        # 2. Batch Size
        layout.addWidget(QLabel("Intensidad de Trabajo (Batch Size)"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 128)
        self.spin_batch.setValue(32)
        self.spin_batch.setFixedHeight(35)
        layout.addWidget(self.spin_batch)

        # 3. Épocas
        layout.addWidget(QLabel("Pasos de Inspiración (Épocas)"))
        self.spin_epocas = QSpinBox()
        self.spin_epocas.setRange(1, 1000)
        self.spin_epocas.setValue(50)
        self.spin_epocas.setFixedHeight(35)
        layout.addWidget(self.spin_epocas)

        # 4. Mode Collapse (Checkbox)
        self.chk_early_stopping = QCheckBox("Evitar que la IA se 'trabe' (Auto-stop)")
        self.chk_early_stopping.setChecked(True)
        self.chk_early_stopping.setStyleSheet("color: #64748B; font-size: 11px;")
        layout.addWidget(self.chk_early_stopping)

        self.barra_progreso = QProgressBar()
        self.barra_progreso.setFixedHeight(22)
        self.barra_progreso.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.barra_progreso)

        self.lbl_epoca = QLabel("Esperando señal creativa...")
        self.lbl_epoca.setAlignment(Qt.AlignCenter)
        self.lbl_epoca.setStyleSheet("font-size: 11px; color: #94A3B8;")
        layout.addWidget(self.lbl_epoca)

        self.boton_iniciar = QPushButton(" Iniciar Creación")
        self.boton_iniciar.setObjectName("BotonIniciar")
        self.boton_iniciar.setIcon(qta.icon("fa5s.play", color="white"))
        self.boton_iniciar.setFixedHeight(45)

        self.boton_detener = QPushButton(" Detener Proceso")
        self.boton_detener.setObjectName("BotonDetener")
        self.boton_detener.setIcon(qta.icon("fa5s.stop", color="white"))
        self.boton_detener.setFixedHeight(45)
        self.boton_detener.setEnabled(False)

        self.boton_iniciar.clicked.connect(self.iniciar)
        self.boton_detener.clicked.connect(self.detener)

        layout.addWidget(self.boton_iniciar)
        layout.addWidget(self.boton_detener)

    def iniciar(self):
        self.senal_iniciar.emit()

    def detener(self):
        self.senal_detener.emit()

    def get_tasa_aprendizaje(self):
        return self.spin_tasa_aprendizaje.value()

    def get_batch_size(self):
        return self.spin_batch.value()

    def set_habilitar_inicio(self, habilitar: bool):
        self.boton_iniciar.setEnabled(habilitar)

    def actualizar_progreso(self, epoca_actual, max_epocas):
        self.barra_progreso.setMaximum(max_epocas)
        self.barra_progreso.setValue(epoca_actual)
        self.lbl_epoca.setText(f"Paso Creativo {epoca_actual} de {max_epocas}")

    def reiniciar_progreso(self, max_epocas):
        self.barra_progreso.setMaximum(max_epocas)
        self.barra_progreso.setValue(0)
        self.lbl_epoca.setText("Iniciando exploración...")
