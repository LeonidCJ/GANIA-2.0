# -*- coding: utf-8 -*-
from datetime import datetime
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QStatusBar, QTextEdit, QMessageBox, QStackedWidget, QFrame,
                             QLabel, QPushButton, QScrollArea, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
import torch
import qtawesome as qta

from interfaces.estilos import ESTILO_OSCURO, ESTILO_CLARO
from interfaces.controles_personalizados import ToggleTemaNeumorfico
from interfaces.panel_entrenamiento import (PanelConfiguracionDataset, PanelControlFastGAN,
                                            PanelAumentoDataset)
from interfaces.panel_resultados import (PanelVisualizacion, PanelGaleriaSinteticas,
                                         PanelEstadisticasFastGAN)
from interfaces.pantalla_login import PantallaLogin
from interfaces.pantalla_bienvenida import PantallaBienvenida

from utils.gestor_configuracion import cargar_config_usuario, guardar_config_usuario
from utils.gestor_logs import configurar_logging, obtener_logger
from utils.gestor_resultados import crear_estructura_resultados
from utils.dispositivo import obtener_dispositivo
from utils.cargador_datos import guardar_imagenes_preprocesadas_clase

from modelos.fastgan import EntrenadorFastGAN
from aplicacion.sistema_gania import SistemaGania, WorkerFastGAN, WorkerGeneracionMasiva

configurar_logging()
logger = obtener_logger("Interfaz")


class CardPipeline(QFrame):
    """Tarjeta Claymórfica compacta y limpia para acceder a cada etapa del pipeline."""
    def __init__(self, icon_name, titulo, descripcion, color_acento="#3B82F6",
                 color_sombra="#1D4ED8", fondo_suave="rgba(59, 130, 246, 0.12)", callback=None):
        super().__init__()
        self.callback = callback
        self.setObjectName("ModuloCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(105)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(16)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(4)
        self.shadow.setColor(QColor(15, 23, 42, 25))
        self.setGraphicsEffect(self.shadow)

        layout_principal = QHBoxLayout(self)
        layout_principal.setContentsMargins(8, 6, 10, 6)
        layout_principal.setSpacing(16)

        # 1. Barra de acento interna redondeada (evita el recorte curvo en las esquinas)
        barra_acento = QFrame()
        barra_acento.setFixedWidth(5)
        barra_acento.setStyleSheet(f"background-color: {color_acento}; border-radius: 2px; border: none;")
        layout_principal.addWidget(barra_acento)

        # 2. Icono en contenedor suave Claymórfico
        badge_icono = QFrame()
        badge_icono.setFixedSize(48, 48)
        badge_icono.setStyleSheet(
            f"QFrame {{ background-color: {fondo_suave}; border-radius: 14px; border: none; }}"
        )
        l_badge = QVBoxLayout(badge_icono)
        l_badge.setContentsMargins(0, 0, 0, 0)
        icon_lbl = QLabel()
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setPixmap(qta.icon(icon_name, color=color_acento).pixmap(24, 24))
        icon_lbl.setStyleSheet("background: transparent; border: none;")
        l_badge.addWidget(icon_lbl)
        layout_principal.addWidget(badge_icono, 0, Qt.AlignVCenter)

        # 3. Título y descripción breve (sin chips sobrecargados)
        cuerpo = QVBoxLayout()
        cuerpo.setSpacing(6)
        cuerpo.setAlignment(Qt.AlignVCenter)
        title_lbl = QLabel(titulo)
        title_lbl.setStyleSheet("font-size: 14px; font-weight: bold; background: transparent;")
        desc_lbl = QLabel(descripcion)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("font-size: 12px; color: #64748B; background: transparent;")
        cuerpo.addWidget(title_lbl)
        cuerpo.addWidget(desc_lbl)
        layout_principal.addLayout(cuerpo, 1)

        # 4. Botón Claymórfico compacto a la derecha
        self.btn_ir = QPushButton(" Ir a Etapa")
        self.btn_ir.setCursor(Qt.PointingHandCursor)
        self.btn_ir.setIcon(qta.icon("fa5s.arrow-right", color="white"))
        self.btn_ir.setMinimumSize(142, 40)
        self.btn_ir.setStyleSheet(
            f"QPushButton {{ background-color: {color_acento}; color: white; border-radius: 12px; "
            f"padding: 6px 14px; border: none; border-bottom: 3px solid {color_sombra}; "
            f"font-weight: bold; font-size: 12px; }} "
            f"QPushButton:pressed {{ border-bottom: 0px; margin-top: 3px; }}"
        )
        if self.callback:
            self.btn_ir.clicked.connect(self.callback)
        layout_principal.addWidget(self.btn_ir, 0, Qt.AlignVCenter)

    def mousePressEvent(self, event):
        if self.callback:
            self.callback()
        super().mousePressEvent(event)


class CardResumenSesion(QFrame):
    """Mini-tarjeta Claymórfica para mostrar métricas y estado en vivo de la sesión en el Dashboard."""
    def __init__(self, icon_name, categoria, valor_inicial, detalle_inicial,
                 color_acento="#3B82F6", fondo_suave="rgba(59, 130, 246, 0.12)"):
        super().__init__()
        self.setObjectName("ModuloCard")
        self.setMinimumHeight(105)
        self.color_acento = color_acento
        self.fondo_suave = fondo_suave

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(14)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(3)
        self.shadow.setColor(QColor(15, 23, 42, 20))
        self.setGraphicsEffect(self.shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(6)

        fila_sup = QHBoxLayout()
        fila_sup.setSpacing(10)

        self.badge_ico = QFrame()
        self.badge_ico.setFixedSize(34, 34)
        self.badge_ico.setStyleSheet(
            f"QFrame {{ background-color: {fondo_suave}; border-radius: 10px; border: none; }}"
        )
        l_ico = QVBoxLayout(self.badge_ico)
        l_ico.setContentsMargins(0, 0, 0, 0)
        self.lbl_ico = QLabel()
        self.lbl_ico.setAlignment(Qt.AlignCenter)
        self.lbl_ico.setPixmap(qta.icon(icon_name, color=color_acento).pixmap(17, 17))
        self.lbl_ico.setStyleSheet("background: transparent; border: none;")
        l_ico.addWidget(self.lbl_ico)

        self.lbl_categoria = QLabel(categoria.upper())
        self.lbl_categoria.setStyleSheet(
            f"font-size: 10px; font-weight: 800; color: {color_acento}; background: transparent;"
        )

        fila_sup.addWidget(self.badge_ico)
        fila_sup.addWidget(self.lbl_categoria)
        fila_sup.addStretch()
        layout.addLayout(fila_sup)

        self.lbl_valor = QLabel(valor_inicial)
        self.lbl_valor.setStyleSheet("font-size: 15px; font-weight: bold; background: transparent;")
        layout.addWidget(self.lbl_valor)

        self.lbl_detalle = QLabel(detalle_inicial)
        self.lbl_detalle.setStyleSheet("font-size: 11px; color: #64748B; background: transparent;")
        layout.addWidget(self.lbl_detalle)

    def actualizar_datos(self, valor, detalle, icon_name=None):
        self.lbl_valor.setText(valor)
        self.lbl_detalle.setText(detalle)
        if icon_name:
            self.lbl_ico.setPixmap(qta.icon(icon_name, color=self.color_acento).pixmap(17, 17))


class VentanaGAN(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GANIA - Generación por Clases con FastGAN")
        self.resize(1280, 850)

        # 1. Estado del Sistema
        self.sistema = SistemaGania()
        self.fastgan = None
        self.usuario_id = "Investigador"
        self.ruta_dataset_ampliado = None
        self._tema_oscuro = False

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

        self._crear_header()

        self.pantallas_stack = QStackedWidget()
        self.layout_base.addWidget(self.pantallas_stack)

        self._crear_pantalla_dashboard()
        self._crear_pantalla_entrenamiento()
        self._crear_pantalla_aumento()

        self.panel_dataset.selector_clase.currentIndexChanged.connect(self.validar_inicio_entrenamiento)
        self.panel_control_fastgan.combo_resolucion.currentIndexChanged.connect(self.validar_inicio_entrenamiento)
        self._configurar_barra_estado()
        self.validar_inicio_entrenamiento()
        self.cambiar_tema(False)

    def _crear_header(self):
        self.header = QFrame()
        self.header.setFixedHeight(70)
        self.header.setObjectName("MainHeader")
        layout_h = QHBoxLayout(self.header)
        layout_h.setContentsMargins(20, 0, 20, 0)

        self.lbl_titulo_app = QLabel("GANIA 2.0")
        self.lbl_titulo_app.setStyleSheet("font-size: 20px; font-weight: 800; color: #3B82F6;")

        self.lbl_pantalla_actual = QLabel("|  PIPELINE EXPERIMENTAL")
        self.lbl_pantalla_actual.setStyleSheet("font-size: 13px; font-weight: 500; color: #94A3B8; margin-left: 10px;")

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
            btn.setMinimumWidth(135)
            layout_h.addWidget(btn)
        layout_h.addWidget(self.toggle_tema)
        self.layout_base.addWidget(self.header)

    def _crear_pantalla_dashboard(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        dash_content = QWidget()
        dash_content.setObjectName("DashboardContent")
        layout_exterior = QVBoxLayout(dash_content)
        layout_exterior.setContentsMargins(35, 30, 35, 30)

        # Contenedor central acotado (900 px) para que las tarjetas tengan proporción limpia sin estirarse
        contenedor_central = QWidget()
        contenedor_central.setFixedWidth(900)
        contenedor_central.setStyleSheet("background: transparent;")
        layout_dash = QVBoxLayout(contenedor_central)
        layout_dash.setContentsMargins(0, 0, 0, 0)
        layout_dash.setSpacing(18)

        lbl_welcome = QLabel("Pipeline de Generación y Aumento con FastGAN")
        lbl_welcome.setObjectName("WelcomeTitle")
        lbl_welcome.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout_dash.addWidget(lbl_welcome)

        layout_dash.addWidget(CardPipeline(
            icon_name="fa5s.brain",
            titulo="1. Configuración de Dataset y Entrenamiento FastGAN",
            descripcion="Carga de datos, detección de clases y entrenamiento exclusivo por clase.",
            color_acento="#3B82F6",
            color_sombra="#1D4ED8",
            fondo_suave="rgba(59, 130, 246, 0.12)",
            callback=lambda: self.cambiar_pantalla(3, "Entrenamiento FastGAN por Clase")
        ))
        layout_dash.addWidget(CardPipeline(
            icon_name="fa5s.magic",
            titulo="2. Aumento de Dataset por Porcentaje",
            descripcion="Generación sintética por porcentaje y exportación del dataset ampliado.",
            color_acento="#10B981",
            color_sombra="#059669",
            fondo_suave="rgba(16, 185, 129, 0.12)",
            callback=lambda: self.cambiar_pantalla(4, "Dataset Ampliado por Porcentaje")
        ))

        # Sección inferior: Resumen en vivo de la sesión (3 mini-tarjetas en fila)
        lbl_sec_resumen = QLabel("Estado de la Sesión Actual")
        lbl_sec_resumen.setStyleSheet("font-size: 13px; font-weight: bold; color: #64748B; margin-top: 8px;")
        layout_dash.addWidget(lbl_sec_resumen)

        fila_resumen = QHBoxLayout()
        fila_resumen.setSpacing(16)

        self.card_resumen_dataset = CardResumenSesion(
            icon_name="fa5s.database",
            categoria="Dataset Cargado",
            valor_inicial="Sin dataset",
            detalle_inicial="Esperando selección de clase",
            color_acento="#3B82F6",
            fondo_suave="rgba(59, 130, 246, 0.12)"
        )
        self.card_resumen_fid = CardResumenSesion(
            icon_name="fa5s.chart-line",
            categoria="Entrenamiento y FID",
            valor_inicial="En espera",
            detalle_inicial="FID: --  |  Diversidad: --",
            color_acento="#8B5CF6",
            fondo_suave="rgba(139, 92, 246, 0.12)"
        )
        self.card_resumen_sinteticas = CardResumenSesion(
            icon_name="fa5s.layer-group",
            categoria="Sintéticas Generadas",
            valor_inicial="0 imágenes",
            detalle_inicial="Dataset ampliado pendiente",
            color_acento="#10B981",
            fondo_suave="rgba(16, 185, 129, 0.12)"
        )

        fila_resumen.addWidget(self.card_resumen_dataset, 1)
        fila_resumen.addWidget(self.card_resumen_fid, 1)
        fila_resumen.addWidget(self.card_resumen_sinteticas, 1)
        layout_dash.addLayout(fila_resumen)

        layout_exterior.addStretch(1)
        layout_exterior.addWidget(contenedor_central, 0, Qt.AlignHCenter)
        layout_exterior.addStretch(2)
        scroll.setWidget(dash_content)
        self.pantallas_stack.addWidget(scroll)

    def _crear_pantalla_entrenamiento(self):
        self.pantalla_fastgan = QWidget()
        layout_fastgan = QHBoxLayout(self.pantalla_fastgan)

        self.panel_dataset = PanelConfiguracionDataset()
        self.panel_control_fastgan = PanelControlFastGAN()
        self.panel_control_fastgan.senal_iniciar_fastgan.connect(self.iniciar_entrenamiento_fastgan)
        self.panel_control_fastgan.senal_detener_fastgan.connect(self.detener_fastgan)

        scroll_izq = QScrollArea()
        scroll_izq.setFixedWidth(400)
        scroll_izq.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_izq.setWidgetResizable(True)
        cont_izq = QWidget()
        l_izq = QVBoxLayout(cont_izq)
        l_izq.setContentsMargins(6, 4, 12, 8)
        l_izq.addWidget(self.panel_dataset)
        l_izq.addWidget(self.panel_control_fastgan)
        l_izq.addStretch()
        scroll_izq.setWidget(cont_izq)
        layout_fastgan.addWidget(scroll_izq)

        der = QVBoxLayout()
        self.panel_vis_fastgan = PanelVisualizacion("Muestras Sintéticas FastGAN (Rejilla 3x3)")
        self.stats_fastgan = PanelEstadisticasFastGAN()

        self.panel_inferior = QTabWidget()
        self.panel_inferior.setFixedHeight(270)
        self.log_eventos = QTextEdit()
        self.log_eventos.setReadOnly(True)
        self.log_eventos.setObjectName("BitacoraLog")
        self.panel_inferior.addTab(self.stats_fastgan, "Gráficos de Pérdida y FID")
        self.panel_inferior.addTab(self.log_eventos, "Bitácora del Sistema")

        der.addWidget(self.panel_vis_fastgan, 1)
        der.addWidget(self.panel_inferior, 0)
        layout_fastgan.addLayout(der, 1)

        self.pantallas_stack.addWidget(self.pantalla_fastgan)

    def _crear_pantalla_aumento(self):
        self.pantalla_aumento = QWidget()
        layout_aumento = QHBoxLayout(self.pantalla_aumento)

        self.panel_control_aumento = PanelAumentoDataset()
        self.panel_control_aumento.senal_generar_aumento.connect(self.ejecutar_aumento_dataset)

        scroll_aum = QScrollArea()
        scroll_aum.setFixedWidth(420)
        scroll_aum.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_aum.setWidgetResizable(True)
        cont_aum = QWidget()
        l_aum = QVBoxLayout(cont_aum)
        l_aum.addWidget(self.panel_control_aumento)
        l_aum.addStretch()
        scroll_aum.setWidget(cont_aum)
        layout_aumento.addWidget(scroll_aum)

        der_aum = QVBoxLayout()
        self.panel_galeria_aumento = PanelGaleriaSinteticas("Galería de Muestras Sintéticas Generadas")
        der_aum.addWidget(self.panel_galeria_aumento)
        layout_aumento.addLayout(der_aum, 1)

        self.pantallas_stack.addWidget(self.pantalla_aumento)

    def _configurar_barra_estado(self):
        self.barra_estado = QStatusBar()
        self.setStatusBar(self.barra_estado)

        if torch.cuda.is_available():
            info_hw = f"Dispositivo activo: GPU ({torch.cuda.get_device_name(0)})"
        else:
            info_hw = "Dispositivo activo: CPU (No se detectó GPU CUDA)"
        self.barra_estado.showMessage(info_hw)

    def _actualizar_botones_nav(self):
        indice = self.pantallas_stack.currentIndex()
        definiciones = [
            (2, self.btn_nav_home, "fa5s.home"),
            (3, self.btn_nav_fastgan, "fa5s.brain"),
            (4, self.btn_nav_aumento, "fa5s.magic"),
        ]
        for idx_nav, btn, ico_name in definiciones:
            activo = (idx_nav == indice)
            btn.setObjectName("NavActivo" if activo else "NavInactivo")
            color_ico = "white" if activo else ("#CBD5E1" if self._tema_oscuro else "#475569")
            btn.setIcon(qta.icon(ico_name, color=color_ico, color_active="white"))
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def cambiar_pantalla(self, indice, titulo):
        self.pantallas_stack.setCurrentIndex(indice)
        self.lbl_pantalla_actual.setText(f"|  {titulo.upper()}")
        self.header.setVisible(indice >= 2)
        self._actualizar_botones_nav()

    def cambiar_tema(self, oscuro):
        self._tema_oscuro = bool(oscuro)
        self.setStyleSheet(ESTILO_OSCURO if oscuro else ESTILO_CLARO)
        if hasattr(self, "btn_nav_home"):
            self._actualizar_botones_nav()

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

        self.panel_control_aumento.actualizar_conteo_original(cant_orig, clase_sel)

        habilitar = (
            ruta_ds is not None and
            os.path.exists(ruta_ds) and
            clase_sel is not None and
            clase_sel != "Sin clases detectadas"
        )
        self.panel_control_fastgan.btn_iniciar.setEnabled(habilitar)

        if habilitar:
            res_actual = self.panel_control_fastgan.obtener_resolucion()
            guardar_imagenes_preprocesadas_clase(ruta_ds, clase_sel, tamano_imagen=res_actual)

        if hasattr(self, "card_resumen_dataset"):
            if habilitar:
                res_actual = self.panel_control_fastgan.obtener_resolucion()
                self.card_resumen_dataset.actualizar_datos(
                    f"Clase: {clase_sel}",
                    f"{cant_orig} imágenes ({res_actual}x{res_actual} px)"
                )
            else:
                self.card_resumen_dataset.actualizar_datos(
                    "Sin dataset",
                    "Esperando selección de clase"
                )

    def _set_estado_controles_entrenamiento(self, entrenando):
        """Activa o bloquea de forma coherente los selectores e hiperparámetros durante el entrenamiento."""
        self.panel_dataset._set_controles_carga_habilitados(not entrenando)
        self.panel_control_fastgan.set_controles_entrenamiento_habilitados(not entrenando)
        self.panel_control_fastgan.btn_iniciar.setEnabled(not entrenando)
        self.panel_control_fastgan.btn_detener.setEnabled(entrenando)

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

        lr = self.panel_control_fastgan.spin_lr.value()
        batch = self.panel_control_fastgan.spin_batch.value()
        epocas = self.panel_control_fastgan.spin_epocas.value()

        dispositivo_actual = obtener_dispositivo(usar_gpu=self.panel_control_fastgan.usar_gpu())
        resolucion = self.panel_control_fastgan.obtener_resolucion()

        if str(dispositivo_actual) == "cpu" and resolucion > 64:
            mult = 4 if resolucion == 128 else 16
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Aviso de Rendimiento en CPU")
            msg.setText(
                f"Entrenar a {resolucion}x{resolucion} px en CPU es ~{mult}x más lento "
                f"que en 64x64 px y usa más memoria RAM.\n\n"
                f"¿Deseas usar 64x64 px o continuar de todos modos?"
            )
            btn_cambiar = msg.addButton("Usar 64x64", QMessageBox.AcceptRole)
            btn_continuar = msg.addButton("Continuar", QMessageBox.DestructiveRole)
            btn_cancelar = msg.addButton("Cancelar", QMessageBox.RejectRole)
            msg.exec_()

            if msg.clickedButton() == btn_cancelar:
                return
            elif msg.clickedButton() == btn_cambiar:
                self.panel_control_fastgan.combo_resolucion.setCurrentIndex(0)
                resolucion = 64

        self._set_estado_controles_entrenamiento(True)
        self.panel_control_fastgan.iniciar_progreso_entrenamiento(epocas)
        self.panel_control_aumento.set_estado_modelo_listo(False)
        self.stats_fastgan.limpiar()
        self.log_eventos.clear()

        self.escribir_log(f"Dispositivo de cómputo detectado: {str(dispositivo_actual).upper()}")
        self.escribir_log(f"Iniciando entrenamiento FastGAN ({resolucion}x{resolucion} px) para la clase '{clase_sel}'...")
        self.fastgan = EntrenadorFastGAN(dispositivo=dispositivo_actual, lr=lr, tamano_img=resolucion)

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
        self.worker_fastgan.error.connect(self._on_error_fastgan)
        self.worker_fastgan.start()

    def _on_error_fastgan(self, msj_error):
        self._set_estado_controles_entrenamiento(False)
        self.panel_control_fastgan.finalizar_progreso_entrenamiento(exito=False)
        self.barra_estado.showMessage(f"Error: {msj_error}")
        self.escribir_log(f"Error en entrenamiento: {msj_error}")

    def on_fastgan_epoca_terminada(self, epoca, lg, ld, datos):
        img = datos.get('img') if isinstance(datos, dict) else datos
        metricas = datos.get('metricas') if isinstance(datos, dict) else None

        if img is not None:
            self.panel_vis_fastgan.mostrar_imagen(img)

        fid_val = None
        div_val = None
        if isinstance(metricas, dict):
            fid_raw = metricas.get("fid")
            div_raw = metricas.get("diversidad")
            if isinstance(fid_raw, (int, float)):
                fid_val = float(fid_raw)
            if isinstance(div_raw, str) and "%" in div_raw:
                try:
                    div_val = float(div_raw.replace("%", "").strip())
                except ValueError:
                    div_val = None
            elif isinstance(div_raw, (int, float)):
                div_val = float(div_raw)

        self.stats_fastgan.actualizar(epoca, lg, ld, fid=fid_val, diversidad=div_val)

        max_ep = max(1, self.panel_control_fastgan.spin_epocas.value())
        self.panel_control_fastgan.actualizar_progreso_entrenamiento(epoca, max_ep)

        if hasattr(self, "card_resumen_fid"):
            txt_fid = f"{fid_val:.1f}" if fid_val is not None else "--"
            txt_div = f"{div_val:.0f}%" if div_val is not None else "--"
            if fid_val is not None or div_val is not None:
                self._ultimo_resumen_metricas = f"FID: {txt_fid}  |  Diversidad: {txt_div}"
            det = getattr(self, "_ultimo_resumen_metricas", "Calculando métricas...")
            self.card_resumen_fid.actualizar_datos(f"Época {epoca} / {max_ep}", det)

    def on_fastgan_completado(self, msj):
        self.escribir_log(msj)
        self._set_estado_controles_entrenamiento(False)
        self.panel_control_fastgan.finalizar_progreso_entrenamiento(exito=True)
        self.panel_control_aumento.set_estado_modelo_listo(True)
        clase_sel = self.panel_dataset.obtener_clase_seleccionada()
        if hasattr(self, "card_resumen_fid"):
            det = getattr(self, "_ultimo_resumen_metricas", "Pesos EMA guardados")
            self.card_resumen_fid.actualizar_datos(f"Listo ({clase_sel})", det)

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("FastGAN Entrenada")
        msg.setText(
            f"FastGAN entrenada y guardada exitosamente para la clase '{clase_sel}'.\n\n"
            f"El modelo está listo para generar imágenes sintéticas de alta fidelidad."
        )
        btn_ir_aumento = msg.addButton("Ir a Aumento de Dataset", QMessageBox.AcceptRole)
        msg.addButton("Permanecer aquí", QMessageBox.RejectRole)
        msg.exec_()

        if msg.clickedButton() == btn_ir_aumento:
            self.cambiar_pantalla(4, "Dataset Ampliado por Porcentaje")

    def detener_fastgan(self):
        if hasattr(self, 'worker_fastgan') and self.worker_fastgan.isRunning():
            self.worker_fastgan.abortar()
            self._set_estado_controles_entrenamiento(False)
            self.panel_control_fastgan.finalizar_progreso_entrenamiento(exito=False, detenido=True)
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

        if not ruta_orig or not os.path.exists(ruta_orig):
            QMessageBox.warning(self, "Dataset Original Requerido", "Selecciona primero la carpeta del dataset original en la pestaña FastGAN.")
            return

        self.panel_control_aumento.btn_generar_aumento.setEnabled(False)
        self.panel_control_aumento.iniciar_progreso_aumento()

        self.worker_aumento = WorkerGeneracionMasiva(
            entrenador_fastgan=self.fastgan,
            ruta_original=ruta_orig,
            clase_seleccionada=clase_sel,
            cantidad_sinteticas=cantidad_sinteticas
        )
        self.worker_aumento.progreso.connect(self.panel_control_aumento.actualizar_progreso_aumento)
        self.worker_aumento.senal_log.connect(self.escribir_log)
        self.worker_aumento.completado.connect(self.on_aumento_completado)
        self.worker_aumento.error.connect(self._on_error_aumento)
        self.worker_aumento.start()

    def _on_error_aumento(self, e):
        self.panel_control_aumento.btn_generar_aumento.setEnabled(True)
        self.panel_control_aumento.finalizar_progreso_aumento(exito=False)
        QMessageBox.critical(self, "Error Aumento", e)

    def on_aumento_completado(self, ruta_ampliado, ruta_zip, dir_sinteticas_clase=None):
        self.ruta_dataset_ampliado = ruta_ampliado
        clase_sel = self.panel_dataset.obtener_clase_seleccionada()
        cant_orig = self.panel_dataset.obtener_cantidad_original_clase()
        cant_sint = self.panel_control_aumento.cant_sintetica_calculada
        total = cant_orig + cant_sint

        # Cargar galería de imágenes sintéticas individuales manteniendo sincronizada la barra de progreso
        if dir_sinteticas_clase and os.path.exists(dir_sinteticas_clase):
            pct_base = self.worker_aumento.porcentaje_inicio_galeria() if hasattr(self, 'worker_aumento') else 85
            def cb_galeria(actual, total_fotos):
                frac = actual / max(1, total_fotos)
                pct = int(pct_base + frac * (99 - pct_base))
                self.panel_control_aumento.actualizar_progreso_aumento(pct, f"Cargando vista previa ({actual}/{total_fotos})")

            self.panel_galeria_aumento.cargar_galeria_desde_carpeta(dir_sinteticas_clase, callback_progreso=cb_galeria)

        self.panel_control_aumento.finalizar_progreso_aumento(exito=True)
        self.panel_control_aumento.btn_generar_aumento.setEnabled(True)
        if hasattr(self, "card_resumen_sinteticas"):
            self.card_resumen_sinteticas.actualizar_datos(
                f"+{cant_sint} sintéticas",
                f"Total ampliado: {total} imágenes ({clase_sel})"
            )

        self.escribir_log(f"Dataset Ampliado creado exitosamente en {ruta_ampliado}")
        QMessageBox.information(
            self, "Dataset Ampliado Creado",
            f"Proceso de aumento finalizado sin re-entrenamiento:\n\n"
            f"- Clase: '{clase_sel}'\n"
            f"- Imágenes Originales: {cant_orig}\n"
            f"- Imágenes Sintéticas FastGAN: {cant_sint}\n"
            f"- Total en Dataset Ampliado: {total}\n\n"
            f"Ubicación: ./datos/datasets/augmented/{clase_sel}/\n"
            f"Paquete ZIP: {ruta_zip}"
        )

    def mostrar(self):
        self.show()

    def closeEvent(self, event):
        if hasattr(self, 'worker_fastgan') and self.worker_fastgan.isRunning():
            self.worker_fastgan.abortar()
            self.worker_fastgan.wait(1500)
        event.accept()
