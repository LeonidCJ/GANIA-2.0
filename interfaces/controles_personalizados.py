# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame, QLabel, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
import qtawesome as qta

class ToggleTemaNeumorfico(QFrame):
    tema_cambiado = pyqtSignal(bool) # True: Oscuro, False: Claro

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 36) 
        self.setCursor(Qt.PointingHandCursor)
        self._es_oscuro = False

        # Sombra neumórfica persistente
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(8)
        self.shadow.setXOffset(3)
        self.shadow.setYOffset(3)
        self.shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(self.shadow)

        # Layout principal
        self.layout_int = QHBoxLayout(self)
        self.layout_int.setContentsMargins(20, 0, 20, 0) 
        self.layout_int.setSpacing(0)

        # Componentes internos
        self.circulo = QFrame()
        self.circulo.setFixedSize(26, 26)
        
        circulo_layout = QVBoxLayout(self.circulo)
        circulo_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_icono = QLabel()
        self.lbl_icono.setAlignment(Qt.AlignCenter)
        self.lbl_icono.setStyleSheet("background: transparent; border: none;")
        circulo_layout.addWidget(self.lbl_icono)

        self.lbl_texto = QLabel()
        self.lbl_texto.setFixedWidth(100) # Ancho fijo para evitar solapamientos
        self.lbl_texto.setStyleSheet("font-weight: bold; font-size: 9px; background: transparent; border: none;")
        self.lbl_texto.setAlignment(Qt.AlignCenter)

        self.actualizar_ui()

    def actualizar_ui(self):
        # Limpiar layout
        while self.layout_int.count():
            item = self.layout_int.takeAt(0)
            if item.widget(): item.widget().setParent(None)

        if not self._es_oscuro:
            # Configuración MODO CLARO
            self.setStyleSheet("background-color: #FEF3C7; border-radius: 18px; border: none;")
            self.lbl_texto.setText("MODO CLARO")
            self.lbl_texto.setStyleSheet("color: #92400E; font-weight: bold; font-size: 9px; background: transparent;")

            icon = qta.icon("fa5s.sun", color="#F59E0B")
            self.lbl_icono.setPixmap(icon.pixmap(16, 16))

            self.circulo.setStyleSheet("background-color: white; border-radius: 13px; border: none;")
            
            self.layout_int.addWidget(self.circulo)
            self.layout_int.addStretch()
            self.layout_int.addWidget(self.lbl_texto)
            self.layout_int.addStretch()
            self.layout_int.addSpacing(26)
            self.shadow.setColor(QColor(0, 0, 0, 40))
        else:
            # Configuración MODO OSCURO
            self.setStyleSheet("background-color: #1E293B; border-radius: 18px; border: none;")
            self.lbl_texto.setText("MODO OSCURO")
            self.lbl_texto.setStyleSheet("color: #E2E8F0; font-weight: bold; font-size: 9px; background: transparent;")

            icon = qta.icon("fa5s.moon", color="#E2E8F0")
            self.lbl_icono.setPixmap(icon.pixmap(16, 16))

            self.circulo.setStyleSheet("background-color: #475569; border-radius: 13px; border: none;")
            
            self.layout_int.addSpacing(26)
            self.layout_int.addStretch()
            self.layout_int.addWidget(self.lbl_texto)
            self.layout_int.addWidget(self.circulo)
            self.shadow.setColor(QColor(0, 0, 0, 100))

    def mousePressEvent(self, event):
        self.shadow.setXOffset(1)
        self.shadow.setYOffset(1)
        self.shadow.setBlurRadius(5)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.shadow.setXOffset(3)
        self.shadow.setYOffset(3)
        self.shadow.setBlurRadius(10)
        
        self._es_oscuro = not self._es_oscuro
        self.actualizar_ui()
        self.tema_cambiado.emit(self._es_oscuro)
        super().mouseReleaseEvent(event)
