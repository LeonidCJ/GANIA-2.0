# utils/metricas_ia.py
"""
Módulo de métricas de calidad para imágenes generadas por GANs.
Implementa FID real usando ResNet18 como extractor de características,
con fallback a métrica simplificada si los recursos son insuficientes.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy import linalg


# ---------------------------------------------------------------------------
# Extractor de características (ResNet18 pre-entrenada, sin cabeza FC)
# ---------------------------------------------------------------------------

_extractor_cache = None

def _obtener_extractor(dispositivo="cpu"):
    """Carga ResNet18 pre-entrenada una sola vez y la cachea."""
    global _extractor_cache
    if _extractor_cache is None:
        try:
            from torchvision import models
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Eliminar la capa de clasificación final
            extractor = nn.Sequential(*list(resnet.children())[:-1])
            extractor.eval()
            _extractor_cache = extractor
            print("[Métricas] Extractor ResNet18 cargado correctamente para FID.")
        except Exception as e:
            print(f"[Métricas] No se pudo cargar ResNet18: {e}. Se usará métrica de píxeles.")
            _extractor_cache = None
    return _extractor_cache


def _extraer_caracteristicas(imagenes_np_lista, dispositivo="cpu"):
    """
    Convierte una lista de imágenes numpy (H,W,C) uint8 a características
    extraídas por ResNet18.

    Returns:
        np.ndarray de forma (N, 512)
    """
    extractor = _obtener_extractor(dispositivo)
    if extractor is None:
        # Fallback: aplanar y normalizar los píxeles directamente
        feats = [img.astype(np.float32).flatten() for img in imagenes_np_lista]
        return np.array(feats)

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tensores = torch.stack([preprocess(img) for img in imagenes_np_lista])
    extractor = extractor.to(dispositivo)
    with torch.no_grad():
        feats = extractor(tensores.to(dispositivo))  # (N, 512, 1, 1)
    return feats.squeeze(-1).squeeze(-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Cálculo de FID
# ---------------------------------------------------------------------------

def calcular_fid(caracteristicas_reales, caracteristicas_falsas):
    """
    Calcula el Fréchet Inception Distance entre dos conjuntos de características.
    Un FID bajo (~0-30) indica imágenes de buena calidad y diversidad.

    Args:
        caracteristicas_reales: np.ndarray (N, D)
        caracteristicas_falsas: np.ndarray (M, D)

    Returns:
        float: Valor FID (menor = mejor).
    """
    mu1, sigma1 = caracteristicas_reales.mean(0), np.cov(caracteristicas_reales, rowvar=False)
    mu2, sigma2 = caracteristicas_falsas.mean(0), np.cov(caracteristicas_falsas, rowvar=False)

    diff = mu1 - mu2

    # Raíz cuadrada de la covarianza producto (puede tener parte imaginaria pequeña)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def calcular_metricas_calidad(muestras_reales, muestras_falsas, dispositivo="cpu"):
    """
    Calcula métricas de calidad para las imágenes generadas.

    Args:
        muestras_reales: Lista de np.ndarray (H,W,C) uint8 de imágenes reales.
                         Puede ser None si no están disponibles (se omite FID).
        muestras_falsas: Lista de np.ndarray (H,W,C) uint8 de imágenes generadas.
        dispositivo: "cpu" o "cuda".

    Returns:
        dict con claves "fid", "fidelidad", "diversidad", "cantidad".
    """
    resultado = {
        "fid": "N/A",
        "fidelidad": "N/A",
        "diversidad": "N/A",
        "cantidad": len(muestras_falsas) if muestras_falsas else 0,
    }

    if not muestras_falsas or len(muestras_falsas) == 0:
        return resultado

    # --- Diversidad (detección de mode collapse) ---
    diversidad_valor, flag_collapse = estimar_diversidad(muestras_falsas)
    # Normalizar a porcentaje relativo para la UI
    pct_diversidad = min(100, int(diversidad_valor / 5.0))
    resultado["diversidad"] = f"{pct_diversidad}%"
    resultado["collapse"] = flag_collapse

    # --- Fidelidad Visual (varianza interna de cada imagen = nitidez) ---
    nitidez_media = np.mean([img.astype(float).var() for img in muestras_falsas])
    # Normalizar: imágenes ruidosas/muy uniformes → baja nitidez
    pct_fidelidad = min(100, int(nitidez_media / 50.0))
    resultado["fidelidad"] = f"{pct_fidelidad}%"

    # --- FID (requiere imágenes reales y suficientes muestras) ---
    if muestras_reales and len(muestras_reales) >= 10 and len(muestras_falsas) >= 10:
        try:
            feats_reales = _extraer_caracteristicas(muestras_reales, dispositivo)
            feats_falsas = _extraer_caracteristicas(muestras_falsas, dispositivo)
            fid_valor = calcular_fid(feats_reales, feats_falsas)
            resultado["fid"] = round(fid_valor, 2)
        except Exception as e:
            print(f"[Métricas] Error calculando FID: {e}")
            resultado["fid"] = "Error"
    else:
        resultado["fid"] = f"(min. 10 imgs)"

    return resultado


def estimar_diversidad(muestras_falsas, umbral_collapse=0.005):
    """
    Mide la diversidad (Variable Independiente) calculando la varianza media
    entre las imágenes generadas para detectar Mode Collapse.

    Args:
        muestras_falsas: Lista de np.ndarray o np.ndarray apilado.
        umbral_collapse: Límite por debajo del cual se considera colapso.

    Returns:
        tuple: (float diversidad, bool flag_collapse)
    """
    if len(muestras_falsas) < 2:
        return 0.0, False

    muestras = np.array(muestras_falsas, dtype=np.float32)
    # Normalizar para que la varianza sea comparable
    muestras_norm = muestras / 255.0
    varianza = np.var(muestras_norm, axis=0).mean()
    
    # Detección de Mode Collapse: si la varianza es extremadamente baja
    # el generador está produciendo imágenes casi idénticas
    flag_collapse = varianza < umbral_collapse
    
    return float(varianza * 1000), flag_collapse
