# utils/tensor_utils.py
"""
Utilidades compartidas para conversión de tensores.
Centraliza operaciones repetidas en múltiples módulos del proyecto.
"""
import numpy as np


def tensor_to_numpy(tensor):
    """
    Convierte un tensor de imagen normalizado [-1, 1] a un array NumPy uint8 [0, 255].

    Args:
        tensor: Tensor PyTorch de forma (C, H, W) en rango [-1, 1].

    Returns:
        np.ndarray de forma (H, W, C) con dtype uint8, listo para QImage o PIL.
    """
    return (tensor.permute(1, 2, 0).numpy() * 127.5 + 127.5).clip(0, 255).astype("uint8")
