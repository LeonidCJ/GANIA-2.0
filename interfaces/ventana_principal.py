# -*- coding: utf-8 -*-
from datetime import datetime
import os
from PIL import Image
import numpy as np
import traceback
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QStatusBar, QTextEdit, QMessageBox, QStackedWidget, QFrame,
                             QLabel, QPushButton, QButtonGroup, QRadioButton, QScrollArea,
                             QGroupBox, QGraphicsDropShadowEffect)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize, QTimer
from PyQt5.QtGui import QColor
from interfaces.estilos import ESTILO_OSCURO, ESTILO_CLARO
from interfaces.controles_personalizados import ToggleTemaNeumorfico
from interfaces.panel_entrenamiento import PanelModelo, PanelEntrenamiento
from interfaces.panel_resultados import (PanelVisualizacion, PanelEstadisticas,
                                         PanelResultadosCompleto, PanelEvaluacionCompleto)
from interfaces.pantalla_login import PantallaLogin
from interfaces.pantalla_bienvenida import PantallaBienvenida
from utils.gestor_configuracion import cargar_config_usuario, guardar_config_usuario
from utils.gestor_logs import configurar_logging, obtener_logger
import torch
import qtawesome as qta

# Gestores de Aplicación (TESIS)
from aplicacion.gestor_evaluacion import guardar_evaluacion_estimulo
from aplicacion.gestor_generacion import guardar_imagen_numpy

# Inicializar logger raíz
configurar_logging()
logger = obtener_logger("Interfaz")

from modelos.dcgan import EntrenadorDCGAN, Generador
from modelos.cyclegan import EntrenadorCycleGAN
from utils.gestor_resultados import crear_estructura_resultados
from aplicacion.sistema_gania import SistemaGania, EntrenadorWorker
from utils.tensor_utils import tensor_to_numpy
from config import CONFIG

# --- COMPONENTE DE TARJETA ---
class Card(QFrame):
    def __init__(self, icon_name, titulo, descripcion, color_borde="#3B82F6", callback=None):
        super().__init__()
        self.callback = callback
        self.setObjectName("ModuloCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"QFrame#ModuloCard {{ border-left: 6px solid {color_borde}; }}")

        # Sombra dinámica
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(15)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(4)
        self.shadow.setColor(QColor(0, 0, 0, 50))
        self.setGraphicsEffect(self.shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(12)

        header = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(qta.icon(icon_name, color=color_borde).pixmap(40, 40))
        icon_lbl.setStyleSheet("background: transparent;")

        title_lbl = QLabel(titulo)
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; background: transparent;")

        header.addWidget(icon_lbl)
        header.addWidget(title_lbl)
        header.addStretch()
        layout.addLayout(header)

        desc_lbl = QLabel(descripcion)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("font-size: 13px; color: #64748B; background: transparent;")
        layout.addWidget(desc_lbl)

        footer = QHBoxLayout()
        footer.addStretch()
        arrow = QLabel("Acceder →")
        arrow.setStyleSheet("font-weight: bold; color: #3B82F6; background: transparent;")
        footer.addWidget(arrow)
        layout.addLayout(footer)

    def mousePressEvent(self, event):
        if self.callback:
            self.callback()
        super().mousePressEvent(event)


class VentanaGAN(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GANIA 2.0 - Capa de Diseño Humano")
        self.resize(1280, 850)

        # 1. Estado
        self.sistema = SistemaGania()
        self.dcgan = None
        self.cyclegan = None
        self.epoca = 0
        self.max_epocas = CONFIG["ENTRENAMIENTO"]["MAX_EPOCAS"]
        self.usuario_id = "Participante_Anonimo"
        self._buffer_imgs_reales = []
        self._buffer_imgs_falsas = []
        self.entrenamiento_activo = False

        # 2. UI
        self.configurar_interfaz()

        # 3. Login y Bienvenida
        config = cargar_config_usuario()
        self.pantalla_login = PantallaLogin(config)
        self.pantalla_login.senal_login_exitoso.connect(self.finalizar_login)
        self.pantalla_bienvenida = PantallaBienvenida()

        self.pantallas_stack.insertWidget(0, self.pantalla_login)
        self.pantallas_stack.insertWidget(1, self.pantalla_bienvenida)

        self.pantallas_stack.setCurrentIndex(0)
        self.header.hide()

        # 4. Señales
        self.sistema.senal_epoca_terminada.connect(self.on_epoca_terminada)
        self.sistema.senal_error.connect(lambda e: self.barra_estado.showMessage(f"Error: {e}"))

    def configurar_interfaz(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.layout_base = QVBoxLayout(central)
        self.layout_base.setContentsMargins(0, 0, 0, 0)
        self.layout_base.setSpacing(0)

        # Header
        self.header = QFrame()
        self.header.setFixedHeight(75)
        self.header.setObjectName("MainHeader")
        layout_h = QHBoxLayout(self.header)
        layout_h.setContentsMargins(25, 0, 25, 0)
        
        self.lbl_titulo_app = QLabel("GANIA 2.0")
        self.lbl_titulo_app.setStyleSheet("font-size: 22px; font-weight: 800; color: #3B82F6;")

        self.lbl_pantalla_actual = QLabel("|  DASHBOARD")
        self.lbl_pantalla_actual.setStyleSheet("font-size: 14px; font-weight: 500; color: #94A3B8; margin-left: 10px;")

        # Menú Superior
        self.btn_nav_home = QPushButton(" Inicio")
        self.btn_nav_home.setIcon(qta.icon("fa5s.home", color="white"))
        self.btn_nav_home.clicked.connect(lambda: self.cambiar_pantalla(2, "Dashboard Principal"))

        self.btn_nav_train = QPushButton(" Entrenamiento")
        self.btn_nav_train.setIcon(qta.icon("fa5s.magic", color="white"))
        self.btn_nav_train.clicked.connect(lambda: self.cambiar_pantalla(3, "Entrenamiento"))

        self.btn_nav_results = QPushButton(" Resultados")
        self.btn_nav_results.setIcon(qta.icon("fa5s.chart-bar", color="white"))
        self.btn_nav_results.clicked.connect(lambda: self.cambiar_pantalla(4, "Galería"))

        self.btn_nav_eval = QPushButton(" Evaluación")
        self.btn_nav_eval.setIcon(qta.icon("fa5s.clipboard-check", color="white"))
        self.btn_nav_eval.clicked.connect(lambda: self.cambiar_pantalla(5, "Evaluación"))

        self.nav_buttons = [self.btn_nav_home, self.btn_nav_train, self.btn_nav_results, self.btn_nav_eval]

        self.toggle_tema = ToggleTemaNeumorfico()
        self.toggle_tema.tema_cambiado.connect(self.cambiar_tema)

        layout_h.addWidget(self.lbl_titulo_app)
        layout_h.addWidget(self.lbl_pantalla_actual)
        layout_h.addStretch()
        for btn in self.nav_buttons:
            btn.setFixedWidth(130)
            layout_h.addWidget(btn)
        layout_h.addWidget(self.toggle_tema)
        self.layout_base.addWidget(self.header)

        self.pantallas_stack = QStackedWidget()
        self.layout_base.addWidget(self.pantallas_stack)

        # DASHBOARD (2)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        dash_content = QWidget(); layout_dash = QVBoxLayout(dash_content); layout_dash.setContentsMargins(40, 40, 40, 40); layout_dash.setSpacing(25)
        layout_dash.addWidget(QLabel("Laboratorio de Co-Creación", styleSheet="font-size: 28px; font-weight: bold; color: #1E293B;"))
        layout_dash.addWidget(Card("fa5s.magic", "Entrenamiento Creativo", "Configura e inicia procesos de IA.", "#3B82F6", lambda: self.cambiar_pantalla(3, "Entrenamiento")))
        layout_dash.addWidget(Card("fa5s.images", "Galería de Resultados", "Inspecciona y organiza muestras.", "#10B981", lambda: self.cambiar_pantalla(4, "Galería")))
        layout_dash.addWidget(Card("fa5s.clipboard-check", "Evaluación de Calidad", "Registra hallazgos técnicos de tesis.", "#8B5CF6", lambda: self.cambiar_pantalla(5, "Evaluación")))
        layout_dash.addStretch(); scroll.setWidget(dash_content); self.pantallas_stack.addWidget(scroll)

        # ENTRENAMIENTO (3)
        self.pantalla_entreno = QWidget(); layout_e = QHBoxLayout(self.pantalla_entreno)
        self.panel_controles = PanelModelo(); self.panel_entrenamiento = PanelEntrenamiento()
        self.panel_entrenamiento.senal_iniciar.connect(self.iniciar_entrenamiento)
        self.panel_entrenamiento.senal_detener.connect(self.detener_entrenamiento)
        izq = QVBoxLayout(); scroll_izq = QScrollArea(); scroll_izq.setFixedWidth(420); scroll_izq.setWidgetResizable(True)
        cont_izq = QWidget(); l_izq = QVBoxLayout(cont_izq); l_izq.addWidget(self.panel_controles); l_izq.addWidget(self.panel_entrenamiento); l_izq.addStretch()
        scroll_izq.setWidget(cont_izq); izq.addWidget(scroll_izq); layout_e.addLayout(izq)
        der = QVBoxLayout(); self.tabs = QTabWidget()
        self.p_dcgan = PanelVisualizacion("Formas"); self.p_cyclegan = PanelVisualizacion("Estilos")
        self.tabs.addTab(self.p_dcgan, "Formas"); self.tabs.addTab(self.p_cyclegan, "Estilos")
        self.stats = PanelEstadisticas()

        self.panel_inferior = QTabWidget(); self.panel_inferior.setFixedHeight(280)
        self.log_eventos = QTextEdit(); self.log_eventos.setReadOnly(True); self.log_eventos.setObjectName("BitacoraLog")
        self.panel_inferior.addTab(self.stats, "Gráficos de Convergencia")
        self.panel_inferior.addTab(self.log_eventos, "Bitácora del Sistema")

        der.addWidget(self.tabs, 1); der.addWidget(self.panel_inferior, 0); layout_e.addLayout(der, 1)
        self.pantallas_stack.addWidget(self.pantalla_entreno)

        # GALERÍA (4)
        self.pantalla_resultados = PanelResultadosCompleto()
        self.pantalla_resultados.senal_guardar_especifica.connect(self.procesar_guardado_galeria)
        self.pantallas_stack.addWidget(self.pantalla_resultados)

        # EVALUACIÓN (5)
        self.pantalla_evaluacion = PanelEvaluacionCompleto(callback_guardar=self.procesar_evaluacion_tesis)
        self.pantallas_stack.addWidget(self.pantalla_evaluacion)

        self.barra_estado = QStatusBar(); self.setStatusBar(self.barra_estado)
        self.cambiar_tema(False)

    def cambiar_pantalla(self, indice, titulo):
        self.pantallas_stack.setCurrentIndex(indice)
        self.lbl_pantalla_actual.setText(f"|  {titulo.upper()}")
        self.header.setVisible(indice >= 2)

    def cambiar_tema(self, oscuro):
        from interfaces.estilos import ESTILO_OSCURO, ESTILO_CLARO
        self.setStyleSheet(ESTILO_OSCURO if oscuro else ESTILO_CLARO)

    def escribir_log(self, mensaje):
        """Escribe los eventos en la bitácora debajo de la visualización."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        texto = f"[{timestamp}] {mensaje}"
        if hasattr(self, 'log_eventos'):
            self.log_eventos.append(texto)
            self.log_eventos.ensureCursorVisible()
        logger.info(f"UI Log: {mensaje}")

    def finalizar_login(self, usuario_id, ruta_drive):
        self.usuario_id = usuario_id
        guardar_config_usuario(usuario_id, ruta_drive)
        self.pantalla_bienvenida.set_usuario(usuario_id)
        self.pantallas_stack.setCurrentIndex(1)
        QTimer.singleShot(2500, lambda: self.cambiar_pantalla(2, "Dashboard Principal"))

    def validar_inicio_entrenamiento(self):
        if self.entrenamiento_activo: return
        habilitar = False
        o = self.panel_controles.selector_origen.currentText()
        if o == "CIFAR-10": habilitar = True
        elif self.panel_controles.ruta_dataset_a and os.path.exists(self.panel_controles.ruta_dataset_a): habilitar = True
        self.panel_entrenamiento.set_habilitar_inicio(habilitar)

    def iniciar_entrenamiento(self):
        crear_estructura_resultados(); self.entrenamiento_activo = True
        self.panel_entrenamiento.boton_iniciar.setEnabled(False)
        self.panel_entrenamiento.boton_detener.setEnabled(True)
        self.panel_controles.setEnabled(False)
        self.stats.limpiar(); self.pantalla_resultados.limpiar_historial()
        self.log_eventos.clear()
        self.escribir_log("Iniciando nueva sesión de entrenamiento...")

        tasa = self.panel_entrenamiento.get_tasa_aprendizaje()
        batch = self.panel_entrenamiento.get_batch_size()

        # --- SUGERENCIA DE HIPERPARÁMETROS (Técnica de Tesis) ---
        if tasa != 0.0002 or batch != 32:
            mensaje_pedagogico = (
                "¡Atención, Diseñador! Los ajustes actuales de velocidad e intensidad "
                "están fuera de los rangos de equilibrio recomendados.\n\n"
                "Esto podría causar que la IA genere formas caóticas o pierda la "
                "coherencia estética en sus diseños.\n\n"
                "¿Deseas continuar con tu configuración experimental?"
            )
            respuesta = QMessageBox.question(self, "Equilibrio Artístico",
                mensaje_pedagogico, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if respuesta == QMessageBox.No:
                self.entrenamiento_activo = False
                self.panel_entrenamiento.boton_iniciar.setEnabled(True)
                self.panel_entrenamiento.boton_detener.setEnabled(False)
                self.panel_controles.setEnabled(True)
                return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoca = 0
        try:
            from utils.cargador_datos import crear_par_datasets, crear_cargador_datos_diseno
            m = self.panel_controles.selector_modelo.currentText()
            o = self.panel_controles.selector_origen.currentText()
            fa = self.panel_controles.ruta_dataset_a if o == "Carpeta Local" else o
            if "DCGAN" in m:
                self.cargador = crear_cargador_datos_diseno(fa, batch_size=batch)
                self.cargadores = [self.cargador]; self.dcgan = EntrenadorDCGAN(device, lr=tasa)
            else:
                ca, cb = crear_par_datasets(fa, self.panel_controles.ruta_dataset_b, batch_size=batch)
                self.cargadores = [ca, cb]; self.cyclegan = EntrenadorCycleGAN(device, lr=tasa)
            self.ejecutar_siguiente_epoca()
        except Exception as e:
            self.escribir_log(f"Error: {e}")

    def ejecutar_siguiente_epoca(self):
        if not self.entrenamiento_activo or self.epoca >= self.max_epocas:
            self.escribir_log("Entrenamiento finalizado.")
            self.panel_entrenamiento.boton_iniciar.setEnabled(True)
            self.panel_entrenamiento.boton_detener.setEnabled(False)
            self.panel_controles.setEnabled(True)
            self.entrenamiento_activo = False
            return

        modelo = self.dcgan or self.cyclegan
        self.panel_entrenamiento.actualizar_progreso(self.epoca + 1, self.max_epocas)
        self.worker = EntrenadorWorker(modelo, self.cargadores, self.epoca, self.max_epocas)
        self.worker.senal_log.connect(self.escribir_log)
        self.worker.finalizado.connect(self.on_epoca_terminada)
        self.worker.start()

    def on_epoca_terminada(self, epoca, lg, ld, datos):
        img = datos.get('img') if isinstance(datos, dict) else datos
        if img is not None:
            if self.dcgan: self.p_dcgan.mostrar_imagen(img)
            else: self.p_cyclegan.mostrar_imagen(img)
            self.pantalla_resultados.mostrar_resultado_final(img)
            self.stats.actualizar(epoca+1, lg, ld)
        self.epoca += 1; self.ejecutar_siguiente_epoca()

    def detener_entrenamiento(self):
        if not self.entrenamiento_activo: return
        res = QMessageBox.question(self, "Detener", "¿Deseas detener el entrenamiento actual?", QMessageBox.Yes|QMessageBox.No)
        if res == QMessageBox.Yes:
            self.entrenamiento_activo = False
            if hasattr(self, 'worker') and self.worker.isRunning():
                self.worker.abortar()
            self.barra_estado.showMessage("Entrenamiento interrumpido.")
            self.panel_entrenamiento.boton_iniciar.setEnabled(True)
            self.panel_entrenamiento.boton_detener.setEnabled(False)
            self.panel_controles.setEnabled(True)

    def mostrar(self): self.show()

    def procesar_guardado_galeria(self, img, info):
        res = guardar_imagen_numpy(img, "galeria"); self.barra_estado.showMessage(f"Guardado: {res}")

    def procesar_evaluacion_tesis(self, s, c):
        m = self.panel_controles.selector_modelo.currentText()
        if guardar_evaluacion_estimulo(self.usuario_id, m, s, c):
            QMessageBox.information(self, "Tesis", "Registrada."); self.cambiar_pantalla(2, "Dashboard Principal")

    def closeEvent(self, event):
        if self.entrenamiento_activo:
            res = QMessageBox.question(self, "Confirmar Salida", "¿Deseas cerrar la App mientras la IA trabaja?", QMessageBox.Yes|QMessageBox.No)
            if res == QMessageBox.Yes:
                self.detener_entrenamiento(); event.accept()
            else: event.ignore()
        else: event.accept()
