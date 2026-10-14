# interfaces/panel_resultados.py
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
    QFrame, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget, QScrollArea, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QApplication
)
import qtawesome as qta
from utils.gestor_logs import obtener_logger
from utils.metricas_ia import graficar_matriz_confusion_ax

logger = obtener_logger("UI.Resultados")


class PanelVisualizacion(QGroupBox):
    """Panel para mostrar imágenes generadas o de vista previa."""
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
        data = imagen_np.tobytes()
        qimage = QImage(data, ancho, alto, bytes_por_linea, QImage.Format_RGB888)
        qimage.bits = data
        pixmap = QPixmap.fromImage(qimage)
        pixmap_escalado = pixmap.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_imagen.setPixmap(pixmap_escalado)
        self.imagen_actual = imagen_np


class PanelEstadisticasFastGAN(QGroupBox):
    """Visualizador de Curvas de Pérdida de FastGAN y Métricas GAN (FID / Diversidad)."""
    def __init__(self):
        super().__init__("Pérdidas y Métricas FastGAN")
        layout = QVBoxLayout(self)

        self.figura = Figure(figsize=(8.5, 3.0), facecolor='none')
        self.canvas = FigureCanvas(self.figura)

        self.ax1 = self.figura.add_subplot(121)
        self.ax2 = self.figura.add_subplot(122)
        self.ax2b = self.ax2.twinx()  # Eje secundario para Diversidad (%)

        self.epochs = []
        self.losses_G = []
        self.losses_D = []
        self.epochs_fid = []
        self.fid_values = []
        self.epochs_kid = []
        self.kid_values = []
        self.epochs_div = []
        self.div_values = []
        self.last_draw_time = 0

        layout.addWidget(self.canvas)
        self.figura.tight_layout()

    def limpiar(self):
        self.epochs = []
        self.losses_G = []
        self.losses_D = []
        self.epochs_fid = []
        self.fid_values = []
        self.epochs_kid = []
        self.kid_values = []
        self.epochs_div = []
        self.div_values = []
        self.ax1.clear()
        self.ax2.clear()
        self.ax2b.clear()
        self.canvas.draw()

    def actualizar(self, epoca, loss_G, loss_D, fid=None, diversidad=None, kid=None):
        self.epochs.append(epoca)
        self.losses_G.append(loss_G)
        self.losses_D.append(loss_D)

        hay_metricas = fid is not None or kid is not None or diversidad is not None
        if (time.time() - self.last_draw_time < 0.4) and not hay_metricas:
            return
        self.last_draw_time = time.time()

        is_dark = True
        try:
            style = QApplication.instance().styleSheet()
            if "background-color: #F8FAFC" in style: is_dark = False
        except: pass

        color_main = '#F8FAFC' if is_dark else '#1E293B'
        color_sub = '#94A3B8' if is_dark else '#64748B'
        color_grid = '#334155' if is_dark else '#E2E8F0'
        bg_plots = '#0F172A' if is_dark else '#F1F5F9'

        self.figura.patch.set_alpha(0)

        # --- PLOT 1: PÉRDIDAS G y D ---
        self.ax1.clear()
        self.ax1.set_facecolor(bg_plots)
        self.ax1.plot(self.epochs, self.losses_G, label="Pérdida G", color="#3B82F6", linewidth=2)
        self.ax1.plot(self.epochs, self.losses_D, label="Pérdida D", color="#EF4444", linewidth=2)

        self.ax1.set_title("Pérdidas Generador / Discriminador", color=color_main, fontsize=10, fontweight='bold')
        self.ax1.set_xlabel("Épocas", color=color_sub, fontsize=8)
        self.ax1.tick_params(colors=color_sub, labelsize=7)
        self.ax1.grid(True, linestyle="--", alpha=0.3, color=color_grid)
        self.ax1.legend(loc="upper right", fontsize='x-small', framealpha=0.5)
        for s in self.ax1.spines.values(): s.set_color(color_grid)

        # --- PLOT 2: CALIDAD GAN (KID/FID & DIVERSIDAD) ---
        def _a_float(v):
            try:
                return float(v.replace("%", "")) if isinstance(v, str) else float(v)
            except (ValueError, TypeError, AttributeError):
                return None

        for valor, lista_ep, lista_val in ((fid, self.epochs_fid, self.fid_values),
                                           (kid, self.epochs_kid, self.kid_values),
                                           (diversidad, self.epochs_div, self.div_values)):
            if valor is not None:
                v = _a_float(valor)
                if v is not None:
                    lista_val.append(v)
                    lista_ep.append(epoca)

        self.ax2.clear()
        self.ax2b.clear()
        self.ax2b.patch.set_visible(False)
        self.ax2b.yaxis.tick_right()
        self.ax2b.yaxis.set_label_position("right")
        self.ax2.set_facecolor(bg_plots)
        lineas = []
        # KID es fiable con pocas muestras: se prioriza sobre FID en el gráfico.
        if self.kid_values:
            lineas += self.ax2.plot(self.epochs_kid, self.kid_values, 'o-', label="KID×1000 (↓)", color="#8B5CF6", markersize=4)
            self.ax2.set_ylabel("KID×1000", color=color_sub, fontsize=8)
        elif self.fid_values:
            lineas += self.ax2.plot(self.epochs_fid, self.fid_values, 'o-', label="FID (↓)", color="#8B5CF6", markersize=4)
            self.ax2.set_ylabel("FID", color=color_sub, fontsize=8)
        if self.div_values:
            lineas += self.ax2b.plot(self.epochs_div, self.div_values, 's--', label="Diversidad % (↑)", color="#10B981", markersize=4)
            self.ax2b.set_ylim(0, max(110, max(self.div_values) * 1.1))
            self.ax2b.set_ylabel("Diversidad %", color=color_sub, fontsize=8)

        self.ax2.set_title("Calidad (KID) y Diversidad", color=color_main, fontsize=10, fontweight='bold')
        self.ax2.set_xlabel("Épocas", color=color_sub, fontsize=8)
        self.ax2.tick_params(colors=color_sub, labelsize=7)
        self.ax2b.tick_params(colors=color_sub, labelsize=7)
        self.ax2.grid(True, linestyle="--", alpha=0.3, color=color_grid)
        if lineas:
            self.ax2.legend(lineas, [l.get_label() for l in lineas], loc="upper right", fontsize='x-small', framealpha=0.5)
        for s in self.ax2.spines.values(): s.set_color(color_grid)
        for s in self.ax2b.spines.values(): s.set_color(color_grid)

        self.figura.tight_layout()
        self.canvas.draw()


# Alias de retrocompatibilidad
PanelEstadisticasDCGAN = PanelEstadisticasFastGAN


class PanelEvaluacionClasificacion(QGroupBox):
    """Panel de Comparación Experimental: Clasificador en Dataset Original vs. Dataset Ampliado."""
    def __init__(self):
        super().__init__("Evaluación Experimental de Clasificación (Original vs. Ampliado)")
        layout = QVBoxLayout(self)

        # Tabla Comparativa de Métricas
        self.tabla = QTableWidget(2, 4)
        self.tabla.setHorizontalHeaderLabels(["Accuracy", "Precision", "Recall", "F1-Score"])
        self.tabla.setVerticalHeaderLabels(["Dataset Original", "Dataset Ampliado (FastGAN)"])
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.setFixedHeight(110)
        layout.addWidget(self.tabla)

        # Matrices de Confusión
        self.figura = Figure(figsize=(8.5, 3.5), facecolor='none')
        self.canvas = FigureCanvas(self.figura)
        self.ax_orig = self.figura.add_subplot(121)
        self.ax_amp = self.figura.add_subplot(122)

        layout.addWidget(self.canvas)
        self.figura.tight_layout()

    def actualizar_resultados(self, resultados):
        """
        Pinta la tabla comparativa y las matrices de confusión reales.
        """
        orig = resultados.get("original", {})
        amp = resultados.get("ampliado", {})

        # Llenar Tabla
        metrics_keys = ["accuracy", "precision", "recall", "f1_score"]
        for col, key in enumerate(metrics_keys):
            val_orig = f"{orig.get(key, 0.0)*100:.2f}%" if key in orig else "N/A"
            val_amp = f"{amp.get(key, 0.0)*100:.2f}%" if key in amp else "N/A"

            item_o = QTableWidgetItem(val_orig)
            item_a = QTableWidgetItem(val_amp)
            item_o.setTextAlignment(Qt.AlignCenter)
            item_a.setTextAlignment(Qt.AlignCenter)

            self.tabla.setItem(0, col, item_o)
            self.tabla.setItem(1, col, item_a)

        # Dibujar Matrices de Confusión
        is_dark = True
        try:
            style = QApplication.instance().styleSheet()
            if "background-color: #F8FAFC" in style: is_dark = False
        except: pass

        if "matriz_confusion" in orig:
            graficar_matriz_confusion_ax(self.ax_orig, orig["matriz_confusion"], titulo="Matriz: Dataset Original", is_dark=is_dark)

        if "matriz_confusion" in amp:
            graficar_matriz_confusion_ax(self.ax_amp, amp["matriz_confusion"], titulo="Matriz: Dataset Ampliado", is_dark=is_dark)

        self.figura.tight_layout()
        self.canvas.draw()
