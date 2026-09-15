# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QProgressBar
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import qtawesome as qta


class PantallaBienvenida(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)

        # Contenedor central elegante
        self.card = QFrame()
        self.card.setObjectName("ModuloCard")
        self.card.setFixedSize(600, 350)

        card_layout = QVBoxLayout(self.card)
        card_layout.setAlignment(Qt.AlignCenter)
        card_layout.setSpacing(20)

        # Icono decorativo profesional
        self.lbl_icon = QLabel()
        self.lbl_icon.setPixmap(qta.icon("fa5s.magic", color="#3B82F6").pixmap(80, 80))
        self.lbl_icon.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.lbl_icon)

        # Texto de Saludo
        self.lbl_saludo = QLabel("¡BIENVENIDO!")
        self.lbl_saludo.setStyleSheet("""
            font-size: 36px;
            font-weight: bold;
            color: #3B82F6;
            letter-spacing: 2px;
        """)
        self.lbl_saludo.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.lbl_saludo)

        # Mensaje de carga
        self.lbl_status = QLabel("Preparando tu entorno creativo...")
        self.lbl_status.setStyleSheet("color: #94A3B8; font-size: 14px; font-style: italic;")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.lbl_status)

        # Barra de progreso sutil
        self.progress = QProgressBar()
        self.progress.setRange(0, 0) # Modo indeterminado (pulso)
        self.progress.setFixedSize(300, 4)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: none;
                background: rgba(0,0,0,0.1);
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #10B981;
                border-radius: 2px;
            }
        """)
        card_layout.addWidget(self.progress, 0, Qt.AlignCenter)

        layout.addWidget(self.card)

    def set_usuario(self, nombre):
        """Actualiza el texto con el nombre del usuario."""
        self.lbl_saludo.setText(f"¡BIENVENIDO, {nombre.upper()}!")
