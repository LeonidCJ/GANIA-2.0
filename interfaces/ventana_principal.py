# -*- coding: utf-8 -*-
from datetime import datetime
import os
from PIL import Image
import numpy as np
import traceback
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QStatusBar, QTextEdit, QMessageBox, QStackedWidget, QFrame,
                             QLabel, QPushButton, QScrollArea, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
import torch
import qtawesome as qta

from interfaces.estilos import ESTILO_OSCURO, ESTILO_CLARO
from interfaces.controles_personalizados import ToggleTemaNeumorfico
from interfaces.panel_entrenamiento import (PanelConfiguracionDataset, PanelControlFastGAN, PanelControlDCGAN,
                                         PanelAumentoDataset, PanelControlClasificador)
from interfaces.panel_resultados import (PanelVisualizacion, PanelEstadisticasFastGAN, PanelEstadisticasDCGAN,
                                         PanelEvaluacionClasificacion)
from interfaces.pantalla_login import PantallaLogin
from interfaces.pantalla_bienvenida import PantallaBienvenida

from utils.gestor_configuracion import cargar_config_usuario, guardar_config_usuario
from utils.gestor_logs import configurar_logging, obtener_logger
from utils.gestor_resultados import crear_estructura_resultados
from utils.tensor_utils import tensor_to_numpy

from modelos.fastgan import EntrenadorFastGAN
from aplicacion.sistema_gania import SistemaGania, WorkerFastGAN, WorkerDCGAN, WorkerGeneracionMasiva, WorkerClasificacion
from config import CONFIG

configurar_logging()
logger = obtener_logger("Interfaz")


class CardPipeline(QFrame):
    """Tarjeta interactiva para acceder a cada etapa del pipeline experimental."""
    def __init__(self, icon_name, titulo, descripcion, color_borde="#3B82F6", callback=None):
        super().__init__()
        self.callback = callback
        self.setObjectName("ModuloCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"QFrame#ModuloCard {{ border-left: 6px solid {color_borde}; }}")

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(12)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(4)
        self.shadow.setColor(QColor(0, 0, 0, 40))
        self.setGraphicsEffect(self.shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(8)

        header = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(qta.icon(icon_name, color=color_borde).pixmap(32, 32))
        icon_lbl.setStyleSheet("background: transparent;")

        title_lbl = QLabel(titulo)
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; background: transparent;")

        header.addWidget(icon_lbl)
        header.addWidget(title_lbl)
        header.addStretch()
        layout.addLayout(header)

        desc_lbl = QLabel(descripcion)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("font-size: 12px; color: #64748B; background: transparent;")
        layout.addWidget(desc_lbl)

        footer = QHBoxLayout()
        footer.addStretch()
        arrow = QLabel("Ir a Etapa →")
        arrow.setStyleSheet("font-weight: bold; color: #3B82F6; background: transparent; font-size: 12px;")
        footer.addWidget(arrow)
        layout.addLayout(footer)

    def mousePressEvent(self, event):
        if self.callback:
            self.callback()
        super().mousePressEvent(event)


class VentanaGAN(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GANIA 2.0 - Generación por Clases con FastGAN (ICLR 2021)")
        self.resize(1280, 850)

        # 1. Estado del Sistema
        self.sistema = SistemaGania()
        self.fastgan = None
        self.dcgan = None
        self.usuario_id = "Investigador"
        self._buffer_imgs_reales = []
        self._buffer_imgs_falsas = []
        self.ruta_dataset_ampliado = None

        # 2. Configurar Interfaz
        self.configurar_interfaz()

        # 3. Pantalla de Bienvenida / Sesión
        config = cargar_config_usuario()
        self.pantalla_login = PantallaLogin(config)
        self.pantalla_login.senal_login_exitoso.connect(self.finalizar_login)
        self.pantalla_bienvenida = PantallaBienvenida()

        self.pantallas_stack.insertWidget(0, self.pantalla_login)
        self.pantallas_stack.insertWidget(1, self.pantalla_bienvenida)
        self.pantallas_stack.setCurrentIndex(0)
        self.header.hide()

    def configurar_interfaz(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.layout_base = QVBoxLayout(central)
        self.layout_base.setContentsMargins(0, 0, 0, 0)
        self.layout_base.setSpacing(0)

        # --- HEADER PRINCIPAL ---
        self.header = QFrame()
        self.header.setFixedHeight(70)
        self.header.setObjectName("MainHeader")
        layout_h = QHBoxLayout(self.header)
        layout_h.setContentsMargins(20, 0, 20, 0)

        self.lbl_titulo_app = QLabel("GANIA 2.0")
        self.lbl_titulo_app.setStyleSheet("font-size: 20px; font-weight: 800; color: #3B82F6;")

        self.lbl_pantalla_actual = QLabel("|  PIPELINE EXPERIMENTAL")
        self.lbl_pantalla_actual.setStyleSheet("font-size: 13px; font-weight: 500; color: #94A3B8; margin-left: 10px;")

        # Menú Superior de Navegación
        self.btn_nav_home = QPushButton(" Inicio")
        self.btn_nav_home.setIcon(qta.icon("fa5s.home", color="white", color_active="white", color_disabled="white"))
        self.btn_nav_home.clicked.connect(lambda: self.cambiar_pantalla(2, "Dashboard Pipeline"))

        self.btn_nav_fastgan = QPushButton(" FastGAN")
        self.btn_nav_fastgan.setIcon(qta.icon("fa5s.brain", color="white", color_active="white", color_disabled="white"))
        self.btn_nav_fastgan.clicked.connect(lambda: self.cambiar_pantalla(3, "Entrenamiento FastGAN por Clase"))

        self.btn_nav_aumento = QPushButton(" Aumento")
        self.btn_nav_aumento.setIcon(qta.icon("fa5s.magic", color="white", color_active="white", color_disabled="white"))
        self.btn_nav_aumento.clicked.connect(lambda: self.cambiar_pantalla(4, "Dataset Ampliado por Porcentaje"))

        self.nav_buttons = [self.btn_nav_home, self.btn_nav_fastgan, self.btn_nav_aumento]

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

        # =========================================================================
        # 1. PANTALLA DASHBOARD (2) - Vista General del Pipeline
        # =========================================================================
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        dash_content = QWidget(); dash_content.setObjectName("DashboardContent"); layout_dash = QVBoxLayout(dash_content); layout_dash.setContentsMargins(35, 30, 35, 30); layout_dash.setSpacing(20)

        lbl_welcome = QLabel("Pipeline de Generación y Aumento con FastGAN (Few-Shot Synthesis)")
        lbl_welcome.setObjectName("WelcomeTitle")
        layout_dash.addWidget(lbl_welcome)

        layout_dash.addWidget(CardPipeline("fa5s.database", "1. Dataset de Entrada & Clases", "Detección automática de clases y conteo de imágenes originales.", "#3B82F6", lambda: self.cambiar_pantalla(3, "Entrenamiento FastGAN por Clase")))
        layout_dash.addWidget(CardPipeline("fa5s.brain", "2. Entrenamiento FastGAN por Clase", "Sintonización con conectividad SLE y decodificación perceptiva autosupervisada.", "#8B5CF6", lambda: self.cambiar_pantalla(3, "Entrenamiento FastGAN por Clase")))
        layout_dash.addWidget(CardPipeline("fa5s.magic", "3. Aumento por Porcentaje", "Selección de porcentaje (e.g. 25%), generación sintética y fusión del dataset ampliado.", "#10B981", lambda: self.cambiar_pantalla(4, "Dataset Ampliado por Porcentaje")))

        layout_dash.addStretch(); scroll.setWidget(dash_content); self.pantallas_stack.addWidget(scroll)

        # =========================================================================
        # 2. PANTALLA ENTRENAMIENTO FASTGAN (3)
        # =========================================================================
        self.pantalla_fastgan = QWidget(); layout_fastgan = QHBoxLayout(self.pantalla_fastgan)

        # Panel Izquierdo: Controles
        self.panel_dataset = PanelConfiguracionDataset()
        self.panel_control_fastgan = PanelControlFastGAN()
        self.panel_control_fastgan.senal_iniciar_fastgan.connect(self.iniciar_entrenamiento_fastgan)
        self.panel_control_fastgan.senal_detener_fastgan.connect(self.detener_fastgan)

        scroll_izq = QScrollArea(); scroll_izq.setFixedWidth(400); scroll_izq.setWidgetResizable(True)
        cont_izq = QWidget(); l_izq = QVBoxLayout(cont_izq)
        l_izq.addWidget(self.panel_dataset)
        l_izq.addWidget(self.panel_control_fastgan)
        l_izq.addStretch(); scroll_izq.setWidget(cont_izq)
        layout_fastgan.addWidget(scroll_izq)

        # Panel Derecho: Visualización y Gráficos (Rejilla de 3x3)
        der = QVBoxLayout()
        self.panel_vis_fastgan = PanelVisualizacion("Muestras Sintéticas FastGAN (Rejilla 3x3)")
        self.stats_fastgan = PanelEstadisticasFastGAN()

        self.panel_inferior = QTabWidget(); self.panel_inferior.setFixedHeight(270)
        self.log_eventos = QTextEdit(); self.log_eventos.setReadOnly(True); self.log_eventos.setObjectName("BitacoraLog")
        self.panel_inferior.addTab(self.stats_fastgan, "Gráficos de Pérdida & FID")
        self.panel_inferior.addTab(self.log_eventos, "Bitácora del Sistema")

        der.addWidget(self.panel_vis_fastgan, 1)
        der.addWidget(self.panel_inferior, 0)
        layout_fastgan.addLayout(der, 1)

        self.pantallas_stack.addWidget(self.pantalla_fastgan)

        # =========================================================================
        # 3. PANTALLA AUMENTO DE DATASET (4)
        # =========================================================================
        self.pantalla_aumento = QWidget(); layout_aumento = QHBoxLayout(self.pantalla_aumento)

        self.panel_control_aumento = PanelAumentoDataset()
        self.panel_control_aumento.senal_generar_aumento.connect(self.ejecutar_aumento_dataset)

        scroll_aum = QScrollArea(); scroll_aum.setFixedWidth(420); scroll_aum.setWidgetResizable(True)
        cont_aum = QWidget(); l_aum = QVBoxLayout(cont_aum)
        l_aum.addWidget(self.panel_control_aumento)
        l_aum.addStretch(); scroll_aum.setWidget(cont_aum)
        layout_aumento.addWidget(scroll_aum)

        # Vista derecha
        der_aum = QVBoxLayout()
        self.panel_vis_aumento = PanelVisualizacion("Vista del Dataset Ampliado (Rejilla 3x3 Sintéticas FastGAN)")
        der_aum.addWidget(self.panel_vis_aumento)
        layout_aumento.addLayout(der_aum, 1)

        self.pantallas_stack.addWidget(self.pantalla_aumento)

        self.panel_dataset.selector_clase.currentIndexChanged.connect(self.validar_inicio_entrenamiento)

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
        QTimer.singleShot(2000, lambda: self.cambiar_pantalla(2, "Dashboard Pipeline"))

    def validar_inicio_entrenamiento(self):
        ruta_ds = self.panel_dataset.obtener_ruta_dataset()
        clase_sel = self.panel_dataset.obtener_clase_seleccionada()
        cant_orig = self.panel_dataset.obtener_cantidad_original_clase()

        # Actualizar la UI del panel de aumento con el número real de imágenes de la clase
        self.panel_control_aumento.actualizar_conteo_original(cant_orig)

        habilitar = (ruta_ds is not None and os.path.exists(ruta_ds) and clase_sel is not None and clase_sel != "Sin clases detectadas")
        self.panel_control_fastgan.btn_iniciar.setEnabled(habilitar)

    # =========================================================================
    # EJECUCIÓN 1: ENTRENAMIENTO DE FASTGAN POR CLASE
    # =========================================================================
    def iniciar_entrenamiento_fastgan(self):
        crear_estructura_resultados()
        ruta_ds = self.panel_dataset.obtener_ruta_dataset()
        clase_sel = self.panel_dataset.obtener_clase_seleccionada()

        if not ruta_ds or not os.path.exists(ruta_ds):
            QMessageBox.warning(self, "Dataset Inválido", "Por favor selecciona o descarga una carpeta/ZIP válido con imágenes.")
            return

        if not clase_sel or clase_sel == "Sin clases detectadas":
            QMessageBox.warning(self, "Clase Inválida", "Por favor selecciona una clase válida para entrenar la red FastGAN.")
            return

        self.panel_control_fastgan.btn_iniciar.setEnabled(False)
        self.panel_control_fastgan.btn_detener.setEnabled(True)
        self.panel_control_fastgan.set_entrenando(True)
        self.panel_control_aumento.set_estado_modelo_listo(False)
        self.stats_fastgan.limpiar()
        self.log_eventos.clear()
        self._buffer_imgs_reales = []
        self._buffer_imgs_falsas = []

        lr = self.panel_control_fastgan.spin_lr.value()
        batch = self.panel_control_fastgan.spin_batch.value()
        epocas = self.panel_control_fastgan.spin_epocas.value()

        dispositivo = self.panel_control_fastgan.obtener_dispositivo()
        if dispositivo.type == "cuda":
            import torch
            self.escribir_log(f"Dispositivo de entrenamiento: GPU NVIDIA ({torch.cuda.get_device_name(0)}) con CUDA.")
        else:
            self.escribir_log("Dispositivo de entrenamiento: CPU (el entrenamiento será más lento).")

        self.escribir_log(f"Iniciando entrenamiento FastGAN exclusivo para la clase '{clase_sel}'...")
        from config import CONFIG
        cfg_gan = CONFIG["FASTGAN"]
        self.fastgan = EntrenadorFastGAN(
            dispositivo=dispositivo, lr=lr,
            politica_aug=cfg_gan.get("POLITICA_DIFFAUGMENT", "color,translation,cutout"),
            ema_beta=cfg_gan.get("EMA_BETA", 0.999)
        )
        self.dcgan = self.fastgan  # Alias para compatibilidad

        self.worker_fastgan = WorkerFastGAN(
            modelo_fastgan=self.fastgan,
            ruta_dataset=ruta_ds,
            clase_seleccionada=clase_sel,
            batch_size=batch,
            lr=lr,
            max_epocas=epocas
        )
        self.worker_fastgan.finalizado_epoca.connect(self.on_fastgan_epoca_terminada)
        self.worker_fastgan.entrenamiento_completado.connect(self.on_fastgan_completado)
        self.worker_fastgan.senal_log.connect(self.escribir_log)
        self.worker_fastgan.error.connect(self.on_fastgan_error)
        self.worker_fastgan.start()

    def on_fastgan_error(self, e):
        self.barra_estado.showMessage(f"Error: {e}")
        self.escribir_log(f"Error: {e}")
        self.panel_control_fastgan.btn_iniciar.setEnabled(True)
        self.panel_control_fastgan.btn_detener.setEnabled(False)
        self.panel_control_fastgan.set_entrenando(False)

    def on_fastgan_epoca_terminada(self, epoca, lg, ld, datos):
        img = datos.get('img') if isinstance(datos, dict) else datos
        metricas = datos.get('metricas') if isinstance(datos, dict) else None

        if img is not None:
            self.panel_vis_fastgan.mostrar_imagen(img)

        # Las métricas se calculan en el worker (cada N épocas) con suficientes muestras.
        fid_val = kid_val = div_val = None
        if metricas:
            fid_val, kid_val, div_val = metricas.get("fid"), metricas.get("kid"), metricas.get("diversidad")

        self.stats_fastgan.actualizar(epoca, lg, ld, fid=fid_val, diversidad=div_val, kid=kid_val)

        self.panel_control_fastgan.barra_progreso.setValue(int((epoca / self.panel_control_fastgan.spin_epocas.value()) * 100))

    def on_fastgan_completado(self, msj):
        self.escribir_log(msj)
        self.panel_control_fastgan.btn_iniciar.setEnabled(True)
        self.panel_control_fastgan.btn_detener.setEnabled(False)
        self.panel_control_fastgan.set_entrenando(False)
        self.panel_control_aumento.set_estado_modelo_listo(True)
        clase_sel = self.panel_dataset.obtener_clase_seleccionada()
        QMessageBox.information(self, "FastGAN Entrenada", f"FastGAN entrenada y guardada exitosamente para la clase '{clase_sel}'.\n\nEl modelo está listo para generar imágenes sintéticas de alta fidelidad.")

    def detener_fastgan(self):
        if hasattr(self, 'worker_fastgan') and self.worker_fastgan.isRunning():
            self.worker_fastgan.abortar()
            self.panel_control_fastgan.btn_iniciar.setEnabled(True)
            self.panel_control_fastgan.btn_detener.setEnabled(False)
            self.panel_control_fastgan.set_entrenando(False)
            self.escribir_log("Entrenamiento de FastGAN detenido por el usuario.")

    # =========================================================================
    # EJECUCIÓN 2: AUMENTO DE DATASET POR CLASE Y PORCENTAJE
    # =========================================================================
    def ejecutar_aumento_dataset(self, cantidad_sinteticas):
        if self.fastgan is None:
            QMessageBox.warning(self, "FastGAN Requerida", "Debes entrenar FastGAN antes de generar imágenes sintéticas.")
            return

        ruta_orig = self.panel_dataset.obtener_ruta_dataset()
        clase_sel = self.panel_dataset.obtener_clase_seleccionada()

        self.worker_aumento = WorkerGeneracionMasiva(
            entrenador_fastgan=self.fastgan,
            ruta_original=ruta_orig,
            clase_seleccionada=clase_sel,
            cantidad_sinteticas=cantidad_sinteticas
        )
        self.worker_aumento.senal_log.connect(self.escribir_log)
        self.worker_aumento.completado.connect(self.on_aumento_completado)
        self.worker_aumento.error.connect(lambda e: QMessageBox.critical(self, "Error Aumento", e))
        self.worker_aumento.start()

    def on_aumento_completado(self, ruta_ampliado, ruta_zip, img_grid_sinteticas=None):
        self.ruta_dataset_ampliado = ruta_ampliado
        clase_sel = self.panel_dataset.obtener_clase_seleccionada()
        cant_orig = self.panel_dataset.obtener_cantidad_original_clase()
        cant_sint = self.panel_control_aumento.cant_sintetica_calculada
        total = cant_orig + cant_sint

        if img_grid_sinteticas is not None:
            self.panel_vis_aumento.mostrar_imagen(img_grid_sinteticas)

        self.escribir_log(f"Dataset Ampliado creado exitosamente en {ruta_ampliado}")
        QMessageBox.information(self, "Dataset Ampliado Creado",
            f"Proceso de aumento finalizado sin re-entrenamiento:\n\n"
            f"• Clase: '{clase_sel}'\n"
            f"• Imágenes Originales: {cant_orig}\n"
            f"• Imágenes Sintéticas FastGAN: {cant_sint}\n"
            f"• Total en Dataset Ampliado: {total}\n\n"
            f"Ubicación: ./datos/datasets/augmented/{clase_sel}/\n"
            f"Paquete ZIP: {ruta_zip}")

    def mostrar(self):
        self.show()

    def closeEvent(self, event):
        event.accept()
