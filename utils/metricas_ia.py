# utils/metricas_ia.py
"""
Módulo de métricas de calidad (FID, KID, Diversidad) y de evaluación de Clasificación (Matriz de Confusión).
Optimizado para CPU/GPU y estabilizado para datasets few-shot (N < 512).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

_extractor_cache = None


def _obtener_extractor(dispositivo="cpu"):
    global _extractor_cache
    if _extractor_cache is None:
        try:
            from torchvision import models
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            extractor = nn.Sequential(*list(resnet.children())[:-1])
            extractor.eval()
            for p in extractor.parameters():
                p.requires_grad_(False)
            _extractor_cache = extractor
            print("[Métricas] Extractor ResNet18 cargado correctamente para FID/KID.")
        except Exception as e:
            print(f"[Métricas] No se pudo cargar ResNet18: {e}.")
            _extractor_cache = None
    return _extractor_cache


def _normalizar_a_tensor_batch(lista_imgs, tamano=128, dispositivo="cpu"):
    """
    Convierte una lista de imágenes numpy (ya sean float [0,1], float [-1,1] o uint8 [0,255])
    a un único tensor normalizado para ImageNet en un solo paso vectorizado (sin bucle PIL).
    """
    tensores = []
    for img in lista_imgs:
        arr = np.asarray(img, dtype=np.float32)
        # Si viene en formato (C, H, W), pasarlo a (H, W, C) temporalmente o manejar directo
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        # Unificar cualquier rango de entrada a [0.0, 1.0]
        min_v, max_v = arr.min(), arr.max()
        if max_v > 1.5:
            arr = arr / 255.0
        elif min_v < -0.05:
            arr = (arr + 1.0) / 2.0

        arr = np.clip(arr, 0.0, 1.0)
        t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        tensores.append(t)

    batch = torch.stack(tensores).to(dispositivo)
    if batch.shape[2] != tamano or batch.shape[3] != tamano:
        batch = F.interpolate(batch, size=(tamano, tamano), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=dispositivo).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dispositivo).view(1, 3, 1, 1)
    return (batch - mean) / std


def _extraer_caracteristicas(imagenes_np_lista, dispositivo="cpu", tamano=128, lote=32):
    extractor = _obtener_extractor(dispositivo)
    if extractor is None:
        feats = [np.asarray(img, dtype=np.float32).flatten() for img in imagenes_np_lista]
        return np.array(feats, dtype=np.float64)

    extractor = extractor.to(dispositivo)
    salida = []
    with torch.no_grad():
        for i in range(0, len(imagenes_np_lista), lote):
            sub_lista = imagenes_np_lista[i:i + lote]
            tensores = _normalizar_a_tensor_batch(sub_lista, tamano=tamano, dispositivo=dispositivo)
            feats = extractor(tensores)
            salida.append(feats.squeeze(-1).squeeze(-1).cpu().numpy())

    return np.concatenate(salida, axis=0).astype(np.float64)


def calcular_fid(caracteristicas_reales, caracteristicas_falsas, eps=1e-5):
    """
    Calcula Fréchet Inception/ResNet Distance con regularización diagonal (eps)
    para evitar explosión numérica cuando N_muestras < 512 (datasets few-shot).
    """
    f1 = caracteristicas_reales / (np.linalg.norm(caracteristicas_reales, axis=1, keepdims=True) + 1e-8) * 8.0
    f2 = caracteristicas_falsas / (np.linalg.norm(caracteristicas_falsas, axis=1, keepdims=True) + 1e-8) * 8.0
    mu1 = np.mean(f1, axis=0)
    mu2 = np.mean(f2, axis=0)
    sigma1 = np.cov(f1, rowvar=False)
    sigma2 = np.cov(f2, rowvar=False)
    n_min = max(2, min(len(f1), len(f2)))
    d = sigma1.shape[0]
    if n_min < d:
        alpha = min(0.85, d / (2.0 * n_min + d))
        sigma1 = (1.0 - alpha) * sigma1 + alpha * np.diag(np.diag(sigma1))
        sigma2 = (1.0 - alpha) * sigma2 + alpha * np.diag(np.diag(sigma2))

    diff = mu1 - mu2

    # Regularización en la diagonal para matrices singulares (N < D)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * (eps * 10)
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(max(0.0, fid))


def calcular_kid(caracteristicas_reales, caracteristicas_falsas, n_subconjuntos=50, tam_subconjunto=None, semilla=0):
    """
    Kernel Inception Distance (Bińkowski et al., 2018): MMD² no sesgado con kernel polinómico.
    Ideal para datasets pequeños (ej. 134 imágenes reales).
    """
    rng = np.random.default_rng(semilla)
    d = caracteristicas_reales.shape[1]
    m = tam_subconjunto or min(len(caracteristicas_reales), len(caracteristicas_falsas), 1000)
    if m < 2:
        return float("nan")

    valores = []
    for _ in range(n_subconjuntos):
        x = caracteristicas_reales[rng.choice(len(caracteristicas_reales), m, replace=False)]
        y = caracteristicas_falsas[rng.choice(len(caracteristicas_falsas), m, replace=False)]
        k_xx = (x @ x.T / d + 1.0) ** 3
        k_yy = (y @ y.T / d + 1.0) ** 3
        k_xy = (x @ y.T / d + 1.0) ** 3
        mmd = (
            (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
            + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
            - 2.0 * k_xy.mean()
        )
        valores.append(mmd)
    return float(max(0.0, np.mean(valores)))


def _diversidad_features(feats):
    """Distancia coseno media entre pares de features (0 = todas iguales)."""
    if len(feats) < 2:
        return 0.0
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sim = f @ f.T
    n = len(f)
    media_sim = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    return float(1.0 - media_sim)


def extraer_caracteristicas_reales(muestras_reales, dispositivo="cpu"):
    """Permite precalcular (y cachear) las features de las imágenes reales una sola vez."""
    if not muestras_reales:
        return None
    return _extraer_caracteristicas(muestras_reales, dispositivo)


def calcular_metricas_calidad(muestras_reales, muestras_falsas, dispositivo="cpu", feats_reales=None):
    """
    Calcula FID, KID (x1000), Diversidad relativa y Fidelidad (nitidez).
    """
    resultado = {
        "fid": "N/A",
        "kid": "N/A",
        "fidelidad": "N/A",
        "diversidad": "N/A",
        "cantidad": len(muestras_falsas) if muestras_falsas else 0,
    }

    if not muestras_falsas or len(muestras_falsas) == 0:
        return resultado

    nitidez_media = np.mean([np.asarray(img, dtype=float).var() for img in muestras_falsas])
    pct_fidelidad = min(100, int(nitidez_media / 50.0))
    resultado["fidelidad"] = f"{pct_fidelidad}%"

    try:
        feats_falsas = _extraer_caracteristicas(muestras_falsas, dispositivo)
    except Exception as e:
        print(f"[Métricas] Error extrayendo características: {e}")
        return resultado

    if feats_reales is None and muestras_reales and len(muestras_reales) >= 5:
        feats_reales = _extraer_caracteristicas(muestras_reales, dispositivo)

    div_falsas = _diversidad_features(feats_falsas)

    if feats_reales is not None and len(feats_reales) >= 5 and len(feats_falsas) >= 5:
        div_reales = _diversidad_features(feats_reales)
        pct_diversidad = int(round(100.0 * div_falsas / max(div_reales, 1e-8)))
        pct_diversidad = min(100, max(0, pct_diversidad))
        resultado["diversidad"] = f"{pct_diversidad}%"
        resultado["collapse"] = pct_diversidad < 30

        try:
            resultado["kid"] = round(calcular_kid(feats_reales, feats_falsas) * 1000.0, 2)
        except Exception as e:
            print(f"[Métricas] Error calculando KID: {e}")
            resultado["kid"] = "Error"

        try:
            resultado["fid"] = round(calcular_fid(feats_reales, feats_falsas), 2)
        except Exception as e:
            print(f"[Métricas] Error calculando FID: {e}")
            resultado["fid"] = "Error"
    else:
        diversidad_valor, flag_collapse = estimar_diversidad(muestras_falsas)
        resultado["diversidad"] = f"{min(100, int(diversidad_valor / 5.0))}%"
        resultado["collapse"] = flag_collapse
        resultado["fid"] = "(min. 5 imgs)"

    return resultado


def estimar_diversidad(muestras_falsas, umbral_collapse=0.005):
    if len(muestras_falsas) < 2:
        return 0.0, False
    muestras = np.array(muestras_falsas, dtype=np.float32)
    if muestras.max() > 1.5:
        muestras = muestras / 255.0
    varianza = np.var(muestras, axis=0).mean()
    flag_collapse = varianza < umbral_collapse
    return float(varianza * 1000), flag_collapse


def graficar_matriz_confusion_ax(ax, cm, clases_nombres=None, titulo="Matriz de Confusión", is_dark=True):
    """
    Renderiza la matriz de confusión en un eje de Matplotlib dado.
    """
    ax.clear()
    bg_color = '#0F172A' if is_dark else '#F1F5F9'
    text_color = '#F8FAFC' if is_dark else '#1E293B'

    ax.set_facecolor(bg_color)
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(titulo, color=text_color, fontsize=10, fontweight='bold', pad=10)

    if clases_nombres is None:
        clases_nombres = [f"C{i}" for i in range(cm.shape[0])]

    tick_marks = np.arange(len(clases_nombres))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(clases_nombres, color=text_color, fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(clases_nombres, color=text_color, fontsize=8)

    ax.set_ylabel('Real', color=text_color, fontsize=9)
    ax.set_xlabel('Predicción', color=text_color, fontsize=9)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            val_color = "white" if val > thresh else "black"
            if is_dark and val <= thresh:
                val_color = "#94A3B8"
            ax.text(j, i, format(val, 'd'),
                    ha="center", va="center",
                    color=val_color, fontsize=9, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_color('#334155' if is_dark else '#CBD5E1')