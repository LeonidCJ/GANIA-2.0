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

class NotaIcono(QFrame):
    """
    Etiqueta compacta compuesta por un icono vectorial (QtAwesome) + texto.
    Sustituye a los emojis en la interfaz para mantener un estilo visual consistente.

    Uso:
        nota = NotaIcono()
        nota.establecer("fa5s.bolt", "Texto", color="#059669", fondo="rgba(16,185,129,0.12)")
    """
    def __init__(self, parent=None, tam_icono=13, compacta=False):
        super().__init__(parent)
        self.tam_icono = tam_icono
        self.compacta = compacta
        self.setObjectName("NotaIcono")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.lbl_icono = QLabel()
        self.lbl_icono.setFixedSize(tam_icono + 2, tam_icono + 2)
        self.lbl_icono.setAlignment(Qt.AlignCenter)
        self.lbl_icono.setStyleSheet("background: transparent;")

        self.lbl_texto = QLabel()
        self.lbl_texto.setWordWrap(not compacta)
        self.lbl_texto.setStyleSheet("background: transparent;")

        layout.addWidget(self.lbl_icono, 0, Qt.AlignTop if not compacta else Qt.AlignVCenter)
        layout.addWidget(self.lbl_texto, 1)

    def establecer(self, icono, texto, color="#64748B", fondo=None, negrita=False, tam_fuente=11):
        self.lbl_icono.setPixmap(qta.icon(icono, color=color).pixmap(self.tam_icono, self.tam_icono))
        self.lbl_texto.setText(texto)
        peso = "bold" if negrita else "600" if fondo else "normal"
        self.lbl_texto.setStyleSheet(
            f"background: transparent; color: {color}; font-size: {tam_fuente}px; font-weight: {peso};"
        )
        if fondo:
            relleno = "4px 10px" if self.compacta else "6px 8px"
            radio = 10 if self.compacta else 8
            self.setStyleSheet(
                f"QFrame#NotaIcono {{ background: {fondo}; border-radius: {radio}px; padding: {relleno}; }}"
            )
            self.layout().setContentsMargins(8, 4, 8, 4) if self.compacta else self.layout().setContentsMargins(8, 6, 8, 6)
        else:
            self.setStyleSheet("QFrame#NotaIcono { background: transparent; }")
            self.layout().setContentsMargins(2, 2, 2, 2)

    def text(self):
        return self.lbl_texto.text()
