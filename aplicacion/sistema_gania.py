# aplicacion/sistema_gania.py
"""
Núcleo de control del sistema GANIA 2.0.
Gestiona el worker de entrenamiento y actúa como puente de señales
entre la lógica de los modelos y la interfaz gráfica.
"""
import torch
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from modelos.dcgan import EntrenadorDCGAN
from modelos.cyclegan import EntrenadorCycleGAN
from utils.tensor_utils import tensor_to_numpy


class EntrenadorWorker(QThread):
    """
    Worker que ejecuta UNA época de entrenamiento en un hilo separado
    para no bloquear la interfaz gráfica (UI thread).

    Señales:
        finalizado(epoca, loss_G, loss_D, img_muestra): Emitida al completar la época.
        error(mensaje): Emitida si ocurre una excepción durante el entrenamiento.
    """
    finalizado = pyqtSignal(int, float, float, object)
    senal_log = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, modelo, cargadores, epoca, max_epocas):
        super().__init__()
        self.modelo = modelo
        self.cargadores = cargadores
        self.epoca = epoca
        self.max_epocas = max_epocas
        self._abortar = False

    def run(self):
        try:
            if self._abortar:
                return
            
            self.senal_log.emit(f">>> Iniciando Época {self.epoca + 1}...")

            img_muestra = None
            def check_abort(): return self._abortar

            if isinstance(self.modelo, EntrenadorDCGAN):
                lg, ld = self.modelo.entrenar_epoca(self.cargadores[0], self.epoca, self.max_epocas, abort_callback=check_abort)
                if self._abortar: return
                with torch.no_grad():
                    muestras = self.modelo.generador(self.modelo.ruido_fijo[:1]).cpu()
                    img_muestra = tensor_to_numpy(muestras[0])

            elif isinstance(self.modelo, EntrenadorCycleGAN):
                lg, ld = self.modelo.entrenar_epoca(self.cargadores[0], self.cargadores[1], self.epoca, self.max_epocas, abort_callback=check_abort)
                if self._abortar: return
                with torch.no_grad():
                    real_A, _ = next(iter(self.cargadores[0]))
                    transformada = self.modelo.G_AB(real_A[:1].to(self.modelo.dispositivo)).cpu()
                    img_muestra = tensor_to_numpy(transformada[0])

            else:
                raise ValueError(f"Tipo de modelo no soportado: {type(self.modelo).__name__}")

            if not self._abortar:
                self.finalizado.emit(self.epoca, lg, ld, img_muestra)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def abortar(self):
        self._abortar = True


class SistemaGania(QObject):
    """
    Puente de señales entre la lógica de los modelos y la ventana principal.
    La ventana instancia esta clase para conectar sus señales.
    El control del flujo de épocas lo realiza directamente VentanaGAN.
    """
    senal_epoca_terminada = pyqtSignal(int, float, float)
    senal_actualizar_vista = pyqtSignal(dict)
    senal_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
