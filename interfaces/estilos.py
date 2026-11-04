# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageDraw

# --- GENERACIÓN DE ICONOS VECTORIALES NÍTIDOS PARA STEPPERS Y COMBOBOX ---
_DIR_ICONOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recursos_ui")


def _asegurar_iconos_controles():
    os.makedirs(_DIR_ICONOS, exist_ok=True)
    variantes = {
        "chevron_up_blue.png": ("up", (59, 130, 246, 255), 11),
        "chevron_down_blue.png": ("down", (59, 130, 246, 255), 11),
        "chevron_up_white.png": ("up", (255, 255, 255, 255), 11),
        "chevron_down_white.png": ("down", (255, 255, 255, 255), 11),
        "chevron_up_muted.png": ("up", (148, 163, 184, 180), 11),
        "chevron_down_muted.png": ("down", (148, 163, 184, 180), 11),
    }
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for nombre, (direccion, color, tam) in variantes.items():
        ruta = os.path.join(_DIR_ICONOS, nombre)
        if not os.path.exists(ruta):
            escala = 4
            s = tam * escala
            img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            grosor = max(2, int(2.2 * escala))
            pad_x = int(s * 0.18)
            pad_y = int(s * 0.30)
            if direccion == "down":
                pts = [(pad_x, pad_y), (s // 2, s - pad_y), (s - pad_x, pad_y)]
            else:
                pts = [(pad_x, s - pad_y), (s // 2, pad_y), (s - pad_x, s - pad_y)]
            draw.line(pts, fill=color, width=grosor, joint="curve")
            r = grosor // 2
            for x, y in pts:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
            img.resize((tam, tam), resampling).save(ruta)


_asegurar_iconos_controles()
_RUTA_UP_BLUE = os.path.join(_DIR_ICONOS, "chevron_up_blue.png").replace("\\", "/")
_RUTA_DOWN_BLUE = os.path.join(_DIR_ICONOS, "chevron_down_blue.png").replace("\\", "/")
_RUTA_UP_WHITE = os.path.join(_DIR_ICONOS, "chevron_up_white.png").replace("\\", "/")
_RUTA_DOWN_WHITE = os.path.join(_DIR_ICONOS, "chevron_down_white.png").replace("\\", "/")
_RUTA_UP_MUTED = os.path.join(_DIR_ICONOS, "chevron_up_muted.png").replace("\\", "/")
_RUTA_DOWN_MUTED = os.path.join(_DIR_ICONOS, "chevron_down_muted.png").replace("\\", "/")

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

    #MainHeader {
        background-color: #1E293B;
        border-bottom: 1px solid #334155;
    }

    /* Botones Claymórficos */
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
    QPushButton:disabled {
        background-color: #334155;
        color: #64748B;
        border-bottom: 2px solid #1E293B;
    }

    QPushButton#BtnDetener {
        background-color: #EF4444;
        border-bottom: 3px solid #B91C1C;
    }
    QPushButton#BtnDetener:hover { background-color: #F87171; }
    QPushButton#BtnDetener:pressed { background-color: #B91C1C; border-bottom: 0px; margin-top: 3px; }
    QPushButton#BtnDetener:disabled {
        background-color: #334155;
        color: #64748B;
        border-bottom: 2px solid #1E293B;
    }

    QPushButton#NavInactivo {
        background-color: #334155;
        color: #CBD5E1;
        border-bottom: 3px solid #1E293B;
    }
    QPushButton#NavInactivo:hover {
        background-color: #475569;
        color: #FFFFFF;
    }
    QPushButton#NavActivo {
        background-color: #3B82F6;
        color: #FFFFFF;
        border-bottom: 3px solid #1D4ED8;
    }

    /* Inputs y SpinBoxes Integrados */
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QTextEdit {
        background-color: #1E293B;
        color: #FFFFFF;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 8px 12px;
        font-weight: bold;
        font-size: 14px;
    }
    QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
        border-color: #3B82F6;
    }
    QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled {
        background-color: #0F172A;
        color: #64748B;
        border-color: #1E293B;
    }

    /* Botones Desplegables y Steppers Claymórficos (Modo Oscuro) */
    QComboBox::drop-down {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 32px;
        border-left: 1px solid #334155;
        border-top-right-radius: 9px;
        border-bottom-right-radius: 9px;
        background-color: rgba(59, 130, 246, 0.10);
    }}
    QComboBox::drop-down:hover {{
        background-color: #3B82F6;
    }}
    QComboBox::down-arrow {{
        image: url("{_RUTA_DOWN_BLUE}");
        width: 11px;
        height: 11px;
    }}
    QComboBox::down-arrow:hover {{
        image: url("{_RUTA_DOWN_WHITE}");
    }}
    QComboBox::down-arrow:disabled {{
        image: url("{_RUTA_DOWN_MUTED}");
    }}

    QDoubleSpinBox::up-button, QSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 28px;
        border-left: 1px solid #334155;
        border-bottom: 1px solid #334155;
        border-top-right-radius: 9px;
        background-color: rgba(59, 130, 246, 0.10);
    }}
    QDoubleSpinBox::down-button, QSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 28px;
        border-left: 1px solid #334155;
        border-bottom-right-radius: 9px;
        background-color: rgba(59, 130, 246, 0.10);
    }}
    QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
    QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
        background-color: #3B82F6;
    }}
    QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed,
    QDoubleSpinBox::down-button:pressed, QSpinBox::down-button:pressed {{
        background-color: #1D4ED8;
    }}

    QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
        image: url("{_RUTA_UP_BLUE}");
        width: 10px;
        height: 10px;
    }}
    QDoubleSpinBox::up-arrow:hover, QSpinBox::up-arrow:hover {{
        image: url("{_RUTA_UP_WHITE}");
    }}
    QDoubleSpinBox::up-arrow:disabled, QSpinBox::up-arrow:disabled,
    QDoubleSpinBox::up-arrow:off, QSpinBox::up-arrow:off {{
        image: url("{_RUTA_UP_MUTED}");
    }}

    QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
        image: url("{_RUTA_DOWN_BLUE}");
        width: 10px;
        height: 10px;
    }}
    QDoubleSpinBox::down-arrow:hover, QSpinBox::down-arrow:hover {{
        image: url("{_RUTA_DOWN_WHITE}");
    }}
    QDoubleSpinBox::down-arrow:disabled, QSpinBox::down-arrow:disabled,
    QDoubleSpinBox::down-arrow:off, QSpinBox::down-arrow:off {{
        image: url("{_RUTA_DOWN_MUTED}");
    }}

    /* Listas Desplegables */
    QComboBox QAbstractItemView {{
        background-color: #1E293B;
        color: #FFFFFF;
        selection-background-color: #3B82F6;
        border: 1px solid #334155;
        outline: none;
    }}

    /* Tablas de Métricas */
    QTableWidget {{
        background-color: #0F172A;
        color: #F8FAFC;
        gridline-color: #334155;
        border: 1px solid #334155;
        border-radius: 10px;
    }}
    QHeaderView::section {{
        background-color: #1E293B;
        color: #60A5FA;
        font-weight: bold;
        border: 1px solid #334155;
        padding: 5px;
    }}

    QTabWidget::pane {{ border: 1px solid #334155; border-radius: 15px; background-color: #1E293B; }}
    QTabBar::tab {{ background-color: #0F172A; color: #94A3B8; padding: 10px 25px; border-top-left-radius: 10px; border-top-right-radius: 10px; }}
    QTabBar::tab:selected {{ background-color: #1E293B; color: #3B82F6; border: 1px solid #334155; border-bottom: none; font-weight: bold; }}

    QProgressBar {{ border: 1px solid #334155; border-radius: 10px; text-align: center; background-color: #0F172A; height: 20px; }}
    QProgressBar::chunk {{ background-color: #3B82F6; border-radius: 8px; }}

    QMessageBox {{ background-color: #1E293B; }}
    QMessageBox QLabel {{ color: #F8FAFC; font-size: 13px; }}
    QMessageBox QPushButton {{ background-color: #3B82F6; color: white; padding: 6px 12px; border-radius: 8px; font-size: 12px; min-width: 85px; }}
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
    #MainHeader { background-color: #FFFFFF; border-bottom: 1px solid #E2E8F0; }

    /* Botones Claymórficos */
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
    QPushButton:disabled {
        background-color: #E2E8F0;
        color: #94A3B8;
        border-bottom: 2px solid #CBD5E1;
    }

    QPushButton#BtnDetener {
        background-color: #EF4444;
        border-bottom: 3px solid #DC2626;
    }
    QPushButton#BtnDetener:hover { background-color: #F87171; }
    QPushButton#BtnDetener:pressed { background-color: #B91C1C; border-bottom: 0px; margin-top: 3px; }
    QPushButton#BtnDetener:disabled {
        background-color: #E2E8F0;
        color: #94A3B8;
        border-bottom: 2px solid #CBD5E1;
    }

    QPushButton#NavInactivo {
        background-color: #E2E8F0;
        color: #475569;
        border-bottom: 3px solid #CBD5E1;
    }
    QPushButton#NavInactivo:hover {
        background-color: #CBD5E1;
        color: #1E293B;
    }
    QPushButton#NavActivo {
        background-color: #3B82F6;
        color: #FFFFFF;
        border-bottom: 3px solid #1D4ED8;
    }

    /* Inputs y SpinBoxes */
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QTextEdit {
        background-color: #FFFFFF;
        color: #1E293B;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 8px 12px;
        font-weight: bold;
        font-size: 14px;
    }
    QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
        border-color: #3B82F6;
    }
    QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled {
        background-color: #F1F5F9;
        color: #94A3B8;
        border-color: #E2E8F0;
    }

    /* Botones Desplegables y Steppers Claymórficos (Modo Claro) */
    QComboBox::drop-down {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 32px;
        border-left: 1px solid #E2E8F0;
        border-top-right-radius: 9px;
        border-bottom-right-radius: 9px;
        background-color: #EFF6FF;
    }}
    QComboBox::drop-down:hover {{
        background-color: #3B82F6;
    }}
    QComboBox::down-arrow {{
        image: url("{_RUTA_DOWN_BLUE}");
        width: 11px;
        height: 11px;
    }}
    QComboBox::down-arrow:hover {{
        image: url("{_RUTA_DOWN_WHITE}");
    }}
    QComboBox::down-arrow:disabled {{
        image: url("{_RUTA_DOWN_MUTED}");
    }}

    QDoubleSpinBox::up-button, QSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 28px;
        border-left: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-top-right-radius: 9px;
        background-color: #EFF6FF;
    }}
    QDoubleSpinBox::down-button, QSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 28px;
        border-left: 1px solid #E2E8F0;
        border-bottom-right-radius: 9px;
        background-color: #EFF6FF;
    }}
    QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
    QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
        background-color: #3B82F6;
    }}
    QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed,
    QDoubleSpinBox::down-button:pressed, QSpinBox::down-button:pressed {{
        background-color: #1D4ED8;
    }}

    QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
        image: url("{_RUTA_UP_BLUE}");
        width: 10px;
        height: 10px;
    }}
    QDoubleSpinBox::up-arrow:hover, QSpinBox::up-arrow:hover {{
        image: url("{_RUTA_UP_WHITE}");
    }}
    QDoubleSpinBox::up-arrow:disabled, QSpinBox::up-arrow:disabled,
    QDoubleSpinBox::up-arrow:off, QSpinBox::up-arrow:off {{
        image: url("{_RUTA_UP_MUTED}");
    }}

    QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
        image: url("{_RUTA_DOWN_BLUE}");
        width: 10px;
        height: 10px;
    }}
    QDoubleSpinBox::down-arrow:hover, QSpinBox::down-arrow:hover {{
        image: url("{_RUTA_DOWN_WHITE}");
    }}
    QDoubleSpinBox::down-arrow:disabled, QSpinBox::down-arrow:disabled,
    QDoubleSpinBox::down-arrow:off, QSpinBox::down-arrow:off {{
        image: url("{_RUTA_DOWN_MUTED}");
    }}

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
    QMessageBox QLabel { color: #1E293B; font-size: 13px; }
    QMessageBox QPushButton { background-color: #3B82F6; color: white; padding: 6px 12px; border-radius: 8px; font-size: 12px; min-width: 85px; }
"""

def _inyectar_iconos(qss):
    return (
        qss.replace("{{", "{")
        .replace("}}", "}")
        .replace("{_RUTA_UP_BLUE}", _RUTA_UP_BLUE)
        .replace("{_RUTA_DOWN_BLUE}", _RUTA_DOWN_BLUE)
        .replace("{_RUTA_UP_WHITE}", _RUTA_UP_WHITE)
        .replace("{_RUTA_DOWN_WHITE}", _RUTA_DOWN_WHITE)
        .replace("{_RUTA_UP_MUTED}", _RUTA_UP_MUTED)
        .replace("{_RUTA_DOWN_MUTED}", _RUTA_DOWN_MUTED)
    )


ESTILO_OSCURO = _inyectar_iconos(ESTILO_OSCURO)
ESTILO_CLARO = _inyectar_iconos(ESTILO_CLARO)


def cargar_estilos(aplicacion):
    aplicacion.setStyleSheet(ESTILO_OSCURO)
