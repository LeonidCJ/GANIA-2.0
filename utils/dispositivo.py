# utils/dispositivo.py
"""
Detección de GPU NVIDIA / CUDA para el entrenamiento.

Distingue tres casos para dar un mensaje claro al usuario:
  1. CUDA disponible en PyTorch            -> se puede entrenar en GPU.
  2. Hay GPU NVIDIA pero PyTorch es solo-CPU -> hay que instalar PyTorch con CUDA.
  3. No hay GPU NVIDIA                      -> solo CPU.
"""
import subprocess
import sys

import torch

_cache_info = None


def _consultar_nvidia_smi():
    """Devuelve el nombre de la GPU NVIDIA según nvidia-smi, o None si no hay driver/GPU."""
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # sin consola en el .exe
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, creationflags=flags
        )
        if salida.returncode == 0 and salida.stdout.strip():
            return salida.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None


def detectar_gpu(forzar=False):
    """
    Returns:
        dict con:
          - disponible (bool): True si PyTorch puede usar CUDA.
          - nombre (str|None): nombre de la GPU.
          - vram_gb (float|None): memoria de la GPU.
          - cuda_version (str|None): versión de CUDA con la que se compiló PyTorch.
          - mensaje (str): texto corto para mostrar en la interfaz.
          - detalle (str): explicación ampliada (tooltip).
    """
    global _cache_info
    if _cache_info is not None and not forzar:
        return _cache_info

    info = {"disponible": False, "nombre": None, "vram_gb": None,
            "cuda_version": torch.version.cuda, "mensaje": "", "detalle": ""}

    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(disponible=True, nombre=props.name,
                        vram_gb=round(props.total_memory / (1024 ** 3), 1))
            info["mensaje"] = f"GPU NVIDIA detectada: {props.name} ({info['vram_gb']} GB)"
            info["detalle"] = f"PyTorch {torch.__version__} con CUDA {torch.version.cuda}. El entrenamiento puede usar la GPU."
            _cache_info = info
            return info
    except Exception as e:
        info["detalle"] = f"Error al inicializar CUDA: {e}"

    nombre_smi = _consultar_nvidia_smi()
    if nombre_smi:
        info["nombre"] = nombre_smi
        info["mensaje"] = f"GPU {nombre_smi} encontrada, pero PyTorch no tiene soporte CUDA"
        info["detalle"] = (
            f"PyTorch {torch.__version__} está instalado solo para CPU.\n"
            "Instala la versión con CUDA, por ejemplo:\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )
    else:
        info["mensaje"] = "No se encontró GPU NVIDIA compatible: se usará CPU"
        info["detalle"] = info["detalle"] or "No se detectó ninguna GPU NVIDIA ni driver (nvidia-smi)."

    _cache_info = info
    return info


def obtener_dispositivo(usar_gpu=True):
    """Devuelve torch.device('cuda') si se pide y está disponible; si no, CPU."""
    if usar_gpu and detectar_gpu()["disponible"]:
        torch.backends.cudnn.benchmark = True  # acelera convoluciones con tamaños fijos
        return torch.device("cuda")
    return torch.device("cpu")
