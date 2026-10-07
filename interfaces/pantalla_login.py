# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QFrame, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
import qtawesome as qta
from utils.gestor_logs import obtener_logger

logger = obtener_logger("UI.Login")

class PantallaLogin(QWidget):
    """
    Pantalla de inicio de sesión simplificada para el estudio 'Capa de Diseño Humano'.
    Recuerda el último ID ingresado.
    """
    senal_login_exitoso = pyqtSignal(str, str) # Emite: usuario_id, ruta_vacia

    def __init__(self, config_previa=None):
        super().__init__()
        self.config_previa = config_previa
        self.init_ui()

    def init_ui(self):
        layout_principal = QVBoxLayout(self)
        layout_principal.setAlignment(Qt.AlignCenter)

        # Tarjeta central Neumórfica
        self.card = QFrame()
        self.card.setObjectName("ModuloCard")
        self.card.setFixedSize(450, 450)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(20)

        # Logo / Título
        h_logo = QHBoxLayout()
        h_logo.setAlignment(Qt.AlignCenter)
        lbl_icon = QLabel()
        lbl_icon.setPixmap(qta.icon("fa5s.dna", color="#3B82F6").pixmap(50, 50))
        lbl_logo = QLabel("GANIA 2.0")
        lbl_logo.setStyleSheet("font-size: 32px; font-weight: bold; color: #3B82F6;")
        h_logo.addWidget(lbl_icon); h_logo.addWidget(lbl_logo)
        card_layout.addLayout(h_logo)

        lbl_desc = QLabel("Capa de Diseño Humano")
        lbl_desc.setAlignment(Qt.AlignCenter)
        lbl_desc.setStyleSheet("color: #64748B; font-size: 16px; margin-bottom: 10px;")
        card_layout.addWidget(lbl_desc)

        # Campo ID
        card_layout.addWidget(QLabel("Nombre o Apodo:"))
        self.input_id = QLineEdit()
        self.input_id.setPlaceholderText("Ej: P001 o Tu Nombre")
        self.input_id.setFixedHeight(45)

        # Cargar memoria
        if self.config_previa:
            self.input_id.setText(self.config_previa.get("usuario_id", ""))
            logger.info(f"ID recuperado: {self.input_id.text()}")

        card_layout.addWidget(self.input_id)
        card_layout.addStretch()

        # Botón Acceder
        self.btn_acceder = QPushButton("Acceder al Entorno")
        self.btn_acceder.setIcon(qta.icon("fa5s.sign-in-alt", color="white"))
        self.btn_acceder.setFixedHeight(50)
        self.btn_acceder.clicked.connect(self.intentar_login)
        card_layout.addWidget(self.btn_acceder)

        layout_principal.addWidget(self.card)

    def intentar_login(self):
        usuario = self.input_id.text().strip()
        if usuario:
            logger.info(f"Login exitoso: {usuario}")
            self.senal_login_exitoso.emit(usuario, "")
        else:
            self.input_id.setStyleSheet("border: 2px solid #EF4444;")
