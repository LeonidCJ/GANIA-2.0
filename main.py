import sys
import os
import multiprocessing
from PyQt5.QtWidgets import QApplication
from interfaces.ventana_principal import VentanaGAN


def principal():
    # Asegurar que el directorio de trabajo es la carpeta del .exe (si está compilado) o la raíz del proyecto
    if getattr(sys, "frozen", False):
        ruta_base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        ruta_base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ruta_base)

    aplicacion = QApplication(sys.argv)

    # Cargar estilos desde el módulo de interfaces
    try:
        from interfaces.estilos import ESTILO_CLARO
        aplicacion.setStyleSheet(ESTILO_CLARO)
    except ImportError:
        print("Aviso: No se pudo cargar la hoja de estilos personalizada.")

    ventana = VentanaGAN()
    ventana.mostrar()

    sys.exit(aplicacion.exec_())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        principal()
    except Exception as e:
        import traceback
        print("¡Ocurrió un error crítico en la aplicación!")
        traceback.print_exc()
