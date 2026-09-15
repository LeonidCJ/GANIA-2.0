# -*- coding: utf-8 -*-

# --- ESTILO CLAYMÓRFICO PROFESIONAL (DISEÑO ORIGINAL 2025) ---

ESTILO_OSCURO = """
    * {
        font-family: 'Segoe UI', sans-serif;
        outline: none;
    }
    QMainWindow, QStackedWidget, QWidget#DashboardContent, QWidget#pantalla_entrenamiento,
    QScrollArea, QScrollArea > QWidget > QWidget {
        background-color: #0F172A;
    }

    /* Grupos con títulos en el borde (Imagen 1) */
    QGroupBox {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 20px;
        margin-top: 15px;
        padding: 20px 15px 15px 15px;
        color: #F8FAFC;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 20px;
        padding: 0 8px;
        color: #3B82F6;
        font-weight: bold;
        font-size: 12px;
    }

    #ModuloCard {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 15px;
    }
    #ModuloCard:hover {
        border-color: #3B82F6;
    }

    QLabel {
        color: #CBD5E1;
    }

    #Header {
        background-color: #1E293B;
        border-bottom: 1px solid #334155;
    }

    /* Botones de acción principales (Azul original) */
    QPushButton {
        background-color: #3B82F6;
        color: white;
        padding: 10px 18px;
        border-radius: 14px;
        border: none;
        border-bottom: 3px solid #1D4ED8;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #60A5FA;
        border-bottom: 4px solid #1D4ED8;
    }
    QPushButton:pressed {
        background-color: #1D4ED8;
        border-bottom: 0px;
        margin-top: 3px;
    }

    /* Inputs y SpinBoxes Claymórficos */
    QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit {
        background-color: #1E293B; /* Fondo igual a la tarjeta */
        color: #F8FAFC;
        border: 2px solid #334155;
        border-radius: 10px;
        padding: 8px 12px;
        font-weight: bold;
        font-size: 14px;
    }
    QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {
        border-color: #3B82F6;
    }

    QTabWidget::pane {
        border: 1px solid #334155;
        border-radius: 15px;
        background-color: #1E293B;
    }
    QTabBar::tab {
        background-color: #0F172A;
        color: #94A3B8;
        padding: 10px 25px;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        margin-right: 4px;
    }
    QTabBar::tab:selected {
        background-color: #1E293B;
        color: #3B82F6;
        border: 1px solid #334155;
        border-bottom: none;
        font-weight: bold;
    }

    QProgressBar {
        border: 1px solid #334155;
        border-radius: 10px;
        text-align: center;
        background-color: #0F172A;
        height: 20px;
    }
    QProgressBar::chunk {
        background-color: #3B82F6;
        border-radius: 8px;
    }
"""

ESTILO_CLARO = """
    * {
        font-family: 'Segoe UI', sans-serif;
    }
    QMainWindow, QStackedWidget, QWidget#DashboardContent, QWidget#pantalla_entrenamiento,
    QScrollArea, QScrollArea > QWidget > QWidget {
        background-color: #F8FAFC;
    }

    QGroupBox {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        margin-top: 15px;
        padding: 20px 15px 15px 15px;
        color: #1E293B;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 20px;
        padding: 0 8px;
        color: #2563EB;
        font-weight: bold;
        font-size: 12px;
    }

    #ModuloCard {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 15px;
    }
    #ModuloCard:hover {
        border-color: #3B82F6;
    }

    QLabel {
        color: #475569;
    }

    #Header {
        background-color: #FFFFFF;
        border-bottom: 1px solid #E2E8F0;
    }

    QPushButton {
        background-color: #3B82F6;
        color: white;
        padding: 10px 18px;
        border-radius: 14px;
        border: none;
        border-bottom: 3px solid #2563EB;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #60A5FA;
        border-bottom: 4px solid #2563EB;
    }
    QPushButton:pressed {
        background-color: #1D4ED8;
        border-bottom: 0px;
        margin-top: 3px;
    }

    QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit {
        background-color: #FFFFFF; /* Fondo igual a la tarjeta */
        color: #1E293B;
        border: 2px solid #CBD5E1;
        border-radius: 10px;
        padding: 8px 12px;
        font-weight: bold;
        font-size: 14px;
    }

    QTabWidget::pane {
        border: 1px solid #E2E8F0;
        border-radius: 15px;
        background-color: #FFFFFF;
    }
    QTabBar::tab {
        background-color: #F1F5F9;
        color: #64748B;
        padding: 10px 25px;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        margin-right: 4px;
    }
    QTabBar::tab:selected {
        background-color: #FFFFFF;
        color: #3B82F6;
        border: 1px solid #E2E8F0;
        border-bottom: none;
        font-weight: bold;
    }

    QProgressBar {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        text-align: center;
        background-color: #F8FAFC;
        height: 20px;
    }
    QProgressBar::chunk {
        background-color: #3B82F6;
        border-radius: 8px;
    }
"""

def cargar_estilos(aplicacion):
    aplicacion.setStyleSheet(ESTILO_OSCURO)
