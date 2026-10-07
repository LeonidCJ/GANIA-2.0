# -*- coding: utf-8 -*-

# --- ESTILO CLAYMÓRFICO PROFESIONAL (GANIA 2.0 - TESIS) ---

ESTILO_OSCURO = """
    * {
        font-family: 'Segoe UI', sans-serif;
        outline: none;
    }
    QMainWindow, QStackedWidget, QWidget#DashboardContent, QWidget#pantalla_entrenamiento,
    QScrollArea, QScrollArea > QWidget > QWidget {
        background-color: #0F172A;
    }

    /* Título Dashboard */
    QLabel#WelcomeTitle {
        font-size: 32px;
        font-weight: bold;
        color: #FFFFFF;
        border-left: 6px solid #3B82F6;
        padding-left: 20px;
    }

    /* Grupos */
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
        color: #60A5FA;
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

    /* Opciones y Selectores */
    QRadioButton, QCheckBox {
        color: #F8FAFC;
        spacing: 12px;
        font-weight: 600;
        font-size: 13px;
    }
    QRadioButton::indicator, QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid #334155;
        background-color: #0F172A;
    }
    QRadioButton::indicator { border-radius: 11px; }

    QRadioButton::indicator:checked, QCheckBox::indicator:checked {
        background-color: #3B82F6;
        border: 4px solid #1E293B;
    }

    QLabel {
        color: #F8FAFC;
    }

    #Header {
        background-color: #1E293B;
        border-bottom: 1px solid #334155;
    }

    /* Botones */
    QPushButton {
        background-color: #3B82F6;
        color: white;
        padding: 10px 18px;
        border-radius: 14px;
        border: none;
        border-bottom: 3px solid #1D4ED8;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #60A5FA; }
    QPushButton:pressed { background-color: #1D4ED8; border-bottom: 0px; margin-top: 3px; }

    /* Inputs y SpinBoxes Integrados */
    QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit {
        background-color: #1E293B;
        color: #FFFFFF;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 8px 12px;
        font-weight: bold;
        font-size: 14px;
    }
    QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {
        border-color: #3B82F6;
    }

    /* Fix para Flechas */
    QComboBox::drop-down, QDoubleSpinBox::up-button, QSpinBox::up-button,
    QDoubleSpinBox::down-button, QSpinBox::down-button {
        border: none;
        background: transparent;
        width: 25px;
    }

    QComboBox::down-arrow, QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {
        image: none;
        width: 0; height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 7px solid #3B82F6;
    }

    QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {
        image: none;
        width: 0; height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-bottom: 7px solid #3B82F6;
    }

    /* Listas Desplegables */
    QComboBox QAbstractItemView {
        background-color: #1E293B;
        color: #FFFFFF;
        selection-background-color: #3B82F6;
        border: 1px solid #334155;
        outline: none;
    }

    /* Tablas de Métricas */
    QTableWidget {
        background-color: #0F172A;
        color: #F8FAFC;
        gridline-color: #334155;
        border: 1px solid #334155;
        border-radius: 10px;
    }
    QHeaderView::section {
        background-color: #1E293B;
        color: #60A5FA;
        font-weight: bold;
        border: 1px solid #334155;
        padding: 5px;
    }

    QTabWidget::pane { border: 1px solid #334155; border-radius: 15px; background-color: #1E293B; }
    QTabBar::tab { background-color: #0F172A; color: #94A3B8; padding: 10px 25px; border-top-left-radius: 10px; border-top-right-radius: 10px; }
    QTabBar::tab:selected { background-color: #1E293B; color: #3B82F6; border: 1px solid #334155; border-bottom: none; font-weight: bold; }

    QProgressBar { border: 1px solid #334155; border-radius: 10px; text-align: center; background-color: #0F172A; height: 20px; }
    QProgressBar::chunk { background-color: #3B82F6; border-radius: 8px; }

    QMessageBox { background-color: #1E293B; }
    QMessageBox QLabel { color: #F8FAFC; font-size: 15px; }
    QMessageBox QPushButton { background-color: #3B82F6; color: white; padding: 8px 25px; border-radius: 10px; min-width: 110px; }
"""

ESTILO_CLARO = """
    * {
        font-family: 'Segoe UI', sans-serif;
    }
    QMainWindow, QStackedWidget, QWidget#DashboardContent, QWidget#pantalla_entrenamiento,
    QScrollArea, QScrollArea > QWidget > QWidget {
        background-color: #F8FAFC;
    }

    QLabel#WelcomeTitle {
        font-size: 32px;
        font-weight: bold;
        color: #1E293B;
        border-left: 6px solid #3B82F6;
        padding-left: 20px;
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
    #ModuloCard:hover { border-color: #3B82F6; }

    /* Opciones y Selectores */
    QRadioButton, QCheckBox {
        color: #1E293B;
        spacing: 12px;
        font-weight: 600;
        font-size: 13px;
    }
    QRadioButton::indicator, QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid #CBD5E1;
        background-color: #F8FAFC;
    }
    QRadioButton::indicator { border-radius: 11px; }

    QRadioButton::indicator:checked, QCheckBox::indicator:checked {
        background-color: #3B82F6;
        border: 4px solid #FFFFFF;
    }

    QLabel { color: #475569; }
    #Header { background-color: #FFFFFF; border-bottom: 1px solid #E2E8F0; }

    QPushButton {
        background-color: #3B82F6;
        color: white;
        padding: 10px 18px;
        border-radius: 14px;
        border: none;
        border-bottom: 3px solid #2563EB;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #60A5FA; }
    QPushButton:pressed { background-color: #1D4ED8; border-bottom: 0px; margin-top: 3px; }

    /* Inputs y SpinBoxes */
    QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit {
        background-color: #FFFFFF;
        color: #1E293B;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 8px 12px;
        font-weight: bold;
        font-size: 14px;
    }

    /* Fix para Flechas */
    QComboBox::drop-down, QDoubleSpinBox::up-button, QSpinBox::up-button,
    QDoubleSpinBox::down-button, QSpinBox::down-button {
        border: none;
        background: transparent;
        width: 25px;
    }

    QComboBox::down-arrow, QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {
        image: none;
        width: 0; height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 7px solid #3B82F6;
    }

    QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {
        image: none;
        width: 0; height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-bottom: 7px solid #3B82F6;
    }

    /* Tablas de Métricas */
    QTableWidget {
        background-color: #FFFFFF;
        color: #1E293B;
        gridline-color: #E2E8F0;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
    }
    QHeaderView::section {
        background-color: #F1F5F9;
        color: #2563EB;
        font-weight: bold;
        border: 1px solid #E2E8F0;
        padding: 5px;
    }

    QTabWidget::pane { border: 1px solid #E2E8F0; border-radius: 15px; background-color: #FFFFFF; }
    QTabBar::tab { background-color: #F1F5F9; color: #64748B; padding: 10px 25px; border-top-left-radius: 10px; border-top-right-radius: 10px; }
    QTabBar::tab:selected { background-color: #FFFFFF; color: #3B82F6; border: 1px solid #E2E8F0; border-bottom: none; font-weight: bold; }

    QProgressBar { border: 1px solid #E2E8F0; border-radius: 10px; text-align: center; background-color: #F8FAFC; height: 20px; }
    QProgressBar::chunk { background-color: #3B82F6; border-radius: 8px; }

    QMessageBox { background-color: #FFFFFF; }
    QMessageBox QLabel { color: #1E293B; font-size: 15px; }
    QMessageBox QPushButton { background-color: #3B82F6; color: white; padding: 8px 20px; border-radius: 10px; min-width: 110px; }
"""

def cargar_estilos(aplicacion):
    aplicacion.setStyleSheet(ESTILO_OSCURO)
