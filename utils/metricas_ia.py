# utils/metricas_ia.py
"""
Módulo de métricas de calidad (FID, Diversidad) y de evaluación de Clasificación (Matriz de Confusión).
"""
import numpy as np
import torch
import torch.nn as nn
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
            _extractor_cache = extractor
            print("[Métricas] Extractor ResNet18 cargado correctamente para FID.")
        except Exception as e:
            print(f"[Métricas] No se pudo cargar ResNet18: {e}.")
            _extractor_cache = None
    return _extractor_cache

def _extraer_caracteristicas(imagenes_np_lista, dispositivo="cpu", tamano=128, lote=64):
    extractor = _obtener_extractor(dispositivo)
    if extractor is None:
        feats = [img.astype(np.float32).flatten() for img in imagenes_np_lista]
        return np.array(feats)

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((tamano, tamano)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    extractor = extractor.to(dispositivo)
    salida = []
    with torch.no_grad():
        for i in range(0, len(imagenes_np_lista), lote):
            tensores = torch.stack([preprocess(img) for img in imagenes_np_lista[i:i + lote]])
            feats = extractor(tensores.to(dispositivo))
            salida.append(feats.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(salida, axis=0).astype(np.float64)

def calcular_fid(caracteristicas_reales, caracteristicas_falsas):
    mu1, sigma1 = caracteristicas_reales.mean(0), np.cov(caracteristicas_reales, rowvar=False)
    mu2, sigma2 = caracteristicas_falsas.mean(0), np.cov(caracteristicas_falsas, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def calcular_kid(caracteristicas_reales, caracteristicas_falsas, n_subconjuntos=50, tam_subconjunto=None, semilla=0):
    """
    Kernel Inception Distance (Bińkowski et al., 2018): MMD² no sesgado con kernel polinómico.
    A diferencia del FID, no es sesgado con pocas muestras, por lo que es más fiable
    con datasets pequeños (p. ej. 80 imágenes reales). Menor es mejor; 0 = distribuciones iguales.
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
        k_xx = (x @ x.T / d + 1) ** 3
        k_yy = (y @ y.T / d + 1) ** 3
        k_xy = (x @ y.T / d + 1) ** 3
        mmd = ((k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
               + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
               - 2 * k_xy.mean())
        valores.append(mmd)
    return float(np.mean(valores))

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

    Recomendado: >= 50 reales y >= 200 falsas. Con menos muestras el FID es muy sesgado
    (por eso se reporta también KID, que es fiable con pocas muestras).
    `feats_reales` permite reutilizar las features reales ya calculadas.
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

    nitidez_media = np.mean([img.astype(float).var() for img in muestras_falsas])
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
        # Diversidad relativa: 100% = tan diversas como las imágenes reales.
        div_reales = _diversidad_features(feats_reales)
        pct_diversidad = int(round(100.0 * div_falsas / max(div_reales, 1e-8)))
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
    muestras_norm = muestras / 255.0
    varianza = np.var(muestras_norm, axis=0).mean()
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
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
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

    # Anotar valores numéricos dentro de las celdas
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
