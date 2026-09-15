# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QGridLayout,
)
import qtawesome as qta
from utils.gestor_logs import obtener_logger

logger = obtener_logger("UI.Resultados")


class PanelVisualizacion(QGroupBox):
    def __init__(self, titulo="Visualización"):
        super().__init__(titulo)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.imagen_actual = None

        self.label_imagen = QLabel("Esperando imagen...")
        self.label_imagen.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_imagen)

    def mostrar_imagen(self, imagen_np):
        if imagen_np is None: return
        imagen_np = np.ascontiguousarray(imagen_np)

        alto, ancho, canales = imagen_np.shape
        bytes_por_linea = canales * ancho

        qimage = QImage(imagen_np.data, ancho, alto, bytes_por_linea, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        pixmap_escalado = pixmap.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_imagen.setPixmap(pixmap_escalado)
        self.imagen_actual = imagen_np


class PanelEstadisticas(QGroupBox):
    def __init__(self):
        super().__init__("Estadísticas del Entrenamiento")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Doble gráfico para Tesis
        self.figura = Figure(figsize=(8.5, 3.2), facecolor='none')
        self.canvas = FigureCanvas(self.figura)

        self.ax1 = self.figura.add_subplot(121)
        self.ax1.set_title("Curvas de Pérdidas")
        self.ax1.set_xlabel("Épocas")
        self.ax1.set_ylabel("Pérdida")

        self.ax2 = self.figura.add_subplot(122)
        self.ax2.set_title("Calidad (FID) y Diversidad")
        self.ax2.set_xlabel("Épocas")
        self.ax2.set_ylabel("Métrica")

        self.epochs = []
        self.losses_G = []
        self.losses_D = []
        self.epochs_eval = []
        self.fid_values = []
        self.div_values = []
        self.last_draw_time = 0

        self.layout.addWidget(self.canvas)
        self.figura.tight_layout()

    def limpiar(self):
        self.epochs = []
        self.losses_G = []
        self.losses_D = []
        self.epochs_eval = []
        self.fid_values = []
        self.div_values = []
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()

    def actualizar(self, epoca, loss_G, loss_D, fid=None, diversidad=None):
        self.epochs.append(epoca)
        self.losses_G.append(loss_G)
        self.losses_D.append(loss_D)

        if (time.time() - self.last_draw_time < 1.0) and (fid is None): return
        self.last_draw_time = time.time()

        self.ax1.clear()
        self.ax1.plot(self.epochs, self.losses_G, label="Generador", color="#3B82F6", linewidth=2)
        self.ax1.plot(self.epochs, self.losses_D, label="Discriminador", color="#EF4444", linewidth=2)
        self.ax1.legend(loc="upper right")
        self.ax1.grid(True, linestyle="--", alpha=0.5)

        if fid is not None or diversidad is not None:
            self.epochs_eval.append(epoca)
            if fid is not None:
                try: self.fid_values.append(float(fid))
                except: pass
            if diversidad is not None:
                try:
                    val = float(diversidad.replace("%", "")) if isinstance(diversidad, str) else float(diversidad)
                    self.div_values.append(val)
                except: pass

        self.ax2.clear()
        if self.fid_values: self.ax2.plot(self.epochs_eval, self.fid_values, 'o-', label="FID", color="#8B5CF6")
        if self.div_values: self.ax2.plot(self.epochs_eval, self.div_values, 's--', label="Diversidad %", color="#10B981")
        self.ax2.legend(loc="upper right")
        self.ax2.grid(True, linestyle="--", alpha=0.5)

        self.figura.tight_layout()
        self.canvas.draw()


class TarjetaImagen(QFrame):
    def __init__(self, imagen_np, texto, al_clic):
        super().__init__()
        self.imagen_np = imagen_np
        self.setObjectName("ModuloCard")
        self.setCursor(Qt.PointingHandCursor)
        self.al_clic = al_clic
        layout = QVBoxLayout(self)

        self.lbl_img = QLabel()
        alto, ancho, canales = imagen_np.shape
        qimage = QImage(imagen_np.data, ancho, alto, canales * ancho, QImage.Format_RGB888)
        self.lbl_img.setPixmap(QPixmap.fromImage(qimage).scaled(280, 280, Qt.KeepAspectRatio))
        self.lbl_img.setAlignment(Qt.AlignCenter)

        self.lbl_info = QLabel(texto)
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("font-weight: bold;")

        layout.addWidget(self.lbl_img)
        layout.addWidget(self.lbl_info)

    def mousePressEvent(self, event):
        self.al_clic(self)
        super().mousePressEvent(event)

    def set_seleccion(self, estado):
        if estado: self.setStyleSheet("QFrame#ModuloCard { border: 3px solid #3B82F6; }")
        else: self.setStyleSheet("")


class PanelResultadosCompleto(QWidget):
    senal_guardar_especifica = pyqtSignal(object, str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Métricas
        self.stats_layout = QHBoxLayout()
        self.widgets_metricas = {}
        for clave, label in [("fid", "FID Score"), ("fidelidad", "Calidad"), ("cantidad", "Muestras")]:
            box = QFrame(); box.setObjectName("ModuloCard")
            l = QVBoxLayout(box)
            val = QLabel("---"); val.setStyleSheet("font-size: 18px; font-weight: bold; color: #3B82F6;")
            l.addWidget(val, 0, Qt.AlignCenter); l.addWidget(QLabel(label), 0, Qt.AlignCenter)
            self.widgets_metricas[clave] = val
            self.stats_layout.addWidget(box)
        layout.addLayout(self.stats_layout)

        # Historial
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.contenedor = QWidget(); self.historial_layout = QGridLayout(self.contenedor)
        self.scroll.setWidget(self.contenedor)
        layout.addWidget(self.scroll, 1)

        # Botones
        self.btn_guardar_seleccionada = QPushButton(" Guardar Imagen Seleccionada")
        self.btn_guardar_seleccionada.setIcon(qta.icon("fa5s.save", color="white"))
        self.btn_guardar_seleccionada.setFixedHeight(50); self.btn_guardar_seleccionada.setEnabled(False)
        self.btn_guardar_seleccionada.clicked.connect(self._emitir_guardado)
        
        layout.addWidget(self.btn_guardar_seleccionada)

        self.tarjeta_seleccionada = None
        self.total_imagenes = 0

    def _manejar_seleccion(self, tarjeta):
        if self.tarjeta_seleccionada: self.tarjeta_seleccionada.set_seleccion(False)
        self.tarjeta_seleccionada = tarjeta
        tarjeta.set_seleccion(True)
        self.btn_guardar_seleccionada.setEnabled(True)

    def _emitir_guardado(self):
        if self.tarjeta_seleccionada:
            self.senal_guardar_especifica.emit(self.tarjeta_seleccionada.imagen_np, "manual")

    def mostrar_resultado_final(self, imagen_np):
        nueva = TarjetaImagen(np.copy(imagen_np), f"Muestra {self.total_imagenes+1}", self._manejar_seleccion)
        self.historial_layout.addWidget(nueva, self.total_imagenes // 3, self.total_imagenes % 3)
        self.total_imagenes += 1

    def limpiar_historial(self):
        self.total_imagenes = 0
        while self.historial_layout.count():
            item = self.historial_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def actualizar_metricas(self, datos):
        if "fid" in datos: self.widgets_metricas["fid"].setText(str(datos["fid"]))
        if "fidelidad" in datos: self.widgets_metricas["fidelidad"].setText(str(datos["fidelidad"]))
        if "cantidad" in datos: self.widgets_metricas["cantidad"].setText(str(datos["cantidad"]))


class PanelEvaluacionCompleto(QWidget):
    def __init__(self, callback_guardar=None):
        super().__init__()
        self.callback_guardar = callback_guardar
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Evaluación del Estímulo Visual"))
        
        self.grupos = {}
        for key, preg in [("utilidad", "1. Utilidad"), ("fluidez", "2. Fluidez"), ("estetica", "3. Satisfacción")]:
            box = QFrame(); box.setObjectName("ModuloCard")
            l = QVBoxLayout(box); l.addWidget(QLabel(preg))
            h = QHBoxLayout(); g = QButtonGroup(self); self.grupos[key] = g
            for i in range(1, 6):
                rb = QRadioButton(str(i)); g.addButton(rb, i); h.addWidget(rb)
            l.addLayout(h); layout.addWidget(box)

        self.txt_comentarios = QTextEdit(); self.txt_comentarios.setPlaceholderText("Comentarios..."); layout.addWidget(self.txt_comentarios)

        self.btn_guardar = QPushButton(" Registrar Evaluación")
        self.btn_guardar.setIcon(qta.icon("fa5s.clipboard-check", color="white"))
        self.btn_guardar.clicked.connect(self._al_guardar)

        layout.addWidget(self.btn_guardar)
        layout.addStretch()

    def _al_guardar(self):
        scores = {k: g.checkedId() for k, g in self.grupos.items()}
        if -1 in scores.values(): return
        if self.callback_guardar:
            self.callback_guardar(scores, self.txt_comentarios.toPlainText())
            self.txt_comentarios.clear()
