# interfaces/panel_resultados.py
import os
import numpy as np
import matplotlib
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QImage, QPixmap, QDesktopServices
from PyQt5.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget, QScrollArea, QGridLayout,
    QApplication
)
import qtawesome as qta
from utils.gestor_logs import obtener_logger

logger = obtener_logger("UI.Resultados")


class TarjetaImagenSintetica(QFrame):
    """
    Tarjeta individual interactiva para cada imagen sintética generada.
    """
    def __init__(self, ruta_imagen, titulo_muestra, al_clic=None):
        super().__init__()
        self.ruta_imagen = ruta_imagen
        self.al_clic = al_clic
        self.setObjectName("ModuloCard")
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.lbl_img = QLabel()
        pixmap = QPixmap(ruta_imagen)
        if not pixmap.isNull():
            pixmap_escalado = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_img.setPixmap(pixmap_escalado)
        else:
            self.lbl_img.setText("Error al cargar")

        self.lbl_img.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_img)

        self.lbl_info = QLabel(titulo_muestra)
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("font-size: 11px; font-weight: bold; color: #3B82F6;")
        layout.addWidget(self.lbl_info)

    def mousePressEvent(self, event):
        if self.al_clic:
            self.al_clic(self.ruta_imagen)
        super().mousePressEvent(event)


class PanelGaleriaSinteticas(QGroupBox):
    """
    Galería desplegable e interactiva con cuadrícula (Grid) de imágenes sintéticas individuales.
    Reemplaza la vista fija de rejilla 3x3 por una cuadrícula desplazable de fotos independientes.
    """
    def __init__(self, titulo="Galería de Imágenes Sintéticas Generadas"):
        super().__init__(titulo)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 18, 12, 12)

        # Contador de resumen
        self.lbl_conteo_galeria = QLabel("Esperando generación de muestras...")
        self.lbl_conteo_galeria.setStyleSheet("font-size: 12px; font-weight: bold; color: #94A3B8; margin-bottom: 5px;")
        layout.addWidget(self.lbl_conteo_galeria)

        # Scroll Area para la galería
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.contenedor = QWidget()
        self.grid_layout = QGridLayout(self.contenedor)
        self.grid_layout.setContentsMargins(5, 5, 5, 5)
        self.grid_layout.setSpacing(12)

        self.scroll.setWidget(self.contenedor)
        layout.addWidget(self.scroll, 1)

    def limpiar(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.lbl_conteo_galeria.setText("Esperando generación de muestras...")

    def cargar_galeria_desde_carpeta(self, ruta_carpeta, max_columnas=3, callback_progreso=None):
        """
        Escanea la carpeta de imágenes sintéticas individuales (.png) y las renderiza en la rejilla.
        """
        self.limpiar()

        if not os.path.exists(ruta_carpeta):
            self.lbl_conteo_galeria.setText("No se encontró la carpeta de muestras sintéticas.")
            return

        extensiones = ('.png', '.jpg', '.jpeg', '.webp')
        archivos = [f for f in os.listdir(ruta_carpeta) if f.lower().endswith(extensiones)]
        archivos.sort()

        if not archivos:
            self.lbl_conteo_galeria.setText("La carpeta de sintéticas aún no contiene imágenes.")
            return

        self.lbl_conteo_galeria.setText(f"Mostrando {len(archivos)} imágenes sintéticas individuales generadas:")

        for idx, filename in enumerate(archivos):
            path_img = os.path.join(ruta_carpeta, filename)
            titulo = f"Sintética {idx + 1:03d}"

            tarjeta = TarjetaImagenSintetica(path_img, titulo, al_clic=self._al_hacer_clic_muestra)
            fila = idx // max_columnas
            columna = idx % max_columnas
            self.grid_layout.addWidget(tarjeta, fila, columna)

            # Reportar progreso real y refrescar la UI para que la barra avance mientras se dibuja
            if callback_progreso:
                callback_progreso(idx + 1, len(archivos))
            if (idx + 1) % 8 == 0:
                QApplication.processEvents()

    def _al_hacer_clic_muestra(self, ruta_img):
        """Abre la imagen individual en el visor predeterminado del sistema operativo."""
        if os.path.exists(ruta_img):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(ruta_img)))


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
        self.ax2b = self.ax2.twinx()

        self.epochs = []
        self.losses_G = []
        self.losses_D = []
        self.epochs_fid = []
        self.fid_values = []
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
        self.epochs_div = []
        self.div_values = []
        self.ax1.clear()
        self.ax2.clear()
        self.ax2b.clear()
        self.canvas.draw()

    def actualizar(self, epoca, loss_G, loss_D, fid=None, diversidad=None):
        self.epochs.append(epoca)
        self.losses_G.append(loss_G)
        self.losses_D.append(loss_D)

        if (time.time() - self.last_draw_time < 0.4) and (fid is None):
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

        # --- PLOT 2: CALIDAD GAN (FID & DIVERSIDAD) ---
        if fid is not None:
            try:
                val = float(fid)
                self.fid_values.append(val)
                self.epochs_fid.append(epoca)
            except (ValueError, TypeError):
                pass

        if diversidad is not None:
            try:
                val = float(diversidad.replace("%", "")) if isinstance(diversidad, str) else float(diversidad)
                self.div_values.append(val)
                self.epochs_div.append(epoca)
            except (ValueError, TypeError):
                pass

        self.ax2.clear()
        self.ax2b.clear()
        self.ax2b.patch.set_visible(False)
        self.ax2b.yaxis.tick_right()
        self.ax2b.yaxis.set_label_position("right")
        self.ax2.set_facecolor(bg_plots)
        lineas = []
        if len(self.fid_values) > 0:
            lineas += self.ax2.plot(self.epochs_fid, self.fid_values, 'o-', label="FID", color="#8B5CF6", markersize=4)
            self.ax2.set_ylabel("FID", color=color_sub, fontsize=8)
        if len(self.div_values) > 0:
            lineas += self.ax2b.plot(self.epochs_div, self.div_values, 's--', label="Diversidad %", color="#10B981", markersize=4)
            self.ax2b.set_ylim(0, max(110, max(self.div_values) * 1.1))
            self.ax2b.set_ylabel("Diversidad %", color=color_sub, fontsize=8)
            self.ax2b.tick_params(colors=color_sub, labelsize=7)
            for s in self.ax2b.spines.values(): s.set_color(color_grid)
        if lineas:
            self.ax2.legend(lineas, [l.get_label() for l in lineas], loc="upper right", fontsize='x-small', framealpha=0.5)






        self.ax2.set_title("Calidad (FID) y Diversidad", color=color_main, fontsize=10, fontweight='bold')
        self.ax2.set_xlabel("Épocas", color=color_sub, fontsize=8)
        self.ax2.tick_params(colors=color_sub, labelsize=7)
        self.ax2.grid(True, linestyle="--", alpha=0.3, color=color_grid)

        for s in self.ax2.spines.values(): s.set_color(color_grid)

        self.figura.tight_layout()
        self.canvas.draw()


# Aliases de retrocompatibilidad
PanelEstadisticasDCGAN = PanelEstadisticasFastGAN
