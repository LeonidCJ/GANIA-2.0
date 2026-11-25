# -*- coding: utf-8 -*-
"""
Suite Automatizada de Pruebas Funcionales (Caja Negra / Integración) - Sistema GANIA.
Ejecuta los 7 casos de prueba funcionales (CP-01 a CP-07) en un entorno aislado
sin alterar los datasets ni checkpoints reales del usuario, e imprime una tabla
resumen lista para documentación de tesis.
"""
import os
import sys
import time
import math
import zipfile
import tempfile
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMessageBox

from interfaces.ventana_principal import VentanaGAN
from modelos.fastgan import EntrenadorFastGAN
from aplicacion.gestor_aumento import GestorAumentoDataset
from utils.cargador_datos import (
    preparar_y_analizar_dataset,
    guardar_imagenes_preprocesadas_clase,
    crear_cargador_fastgan_clase,
    cargar_imagenes_reales_clase,
)
from utils.metricas_ia import calcular_metricas_calidad


def _crear_dataset_prueba(dir_base):
    """Crea un dataset sintético de prueba con 2 clases (8 y 6 imágenes RGB)."""
    rng = np.random.default_rng(42)
    clases = {"felinos_prueba": 8, "caninos_prueba": 6}
    for nombre_clase, cantidad in clases.items():
        ruta_c = os.path.join(dir_base, nombre_clase)
        os.makedirs(ruta_c, exist_ok=True)
        for i in range(1, cantidad + 1):
            arr = rng.integers(30, 235, size=(160, 140, 3), dtype=np.uint8)
            Image.fromarray(arr, "RGB").save(os.path.join(ruta_c, f"img_{i:03d}.png"))
    return clases


def ejecutar_suite_funcional():
    app = QApplication.instance() or QApplication(sys.argv)
    # Evitar bloqueos de diálogos modales durante las pruebas automatizadas
    QMessageBox.information = lambda *a, **kw: None
    QMessageBox.warning = lambda *a, **kw: None
    QMessageBox.critical = lambda *a, **kw: None

    resultados = []
    ventana = VentanaGAN()

    with tempfile.TemporaryDirectory(prefix="gania_test_") as tmp_dir:
        dir_ds = os.path.join(tmp_dir, "dataset_origen")
        dir_preproc = os.path.join(tmp_dir, "preprocesadas")
        dir_ckpt = os.path.join(tmp_dir, "checkpoints")
        dir_orig_aum = os.path.join(tmp_dir, "original")
        dir_sint_aum = os.path.join(tmp_dir, "synthetic")
        dir_amp_aum = os.path.join(tmp_dir, "augmented")
        ruta_zip_aum = os.path.join(tmp_dir, "dataset_ampliado_test.zip")

        _crear_dataset_prueba(dir_ds)

        # =====================================================================
        # CP-01: Control de acceso e identificación de sesión (PantallaLogin)
        # =====================================================================
        t0 = time.perf_counter()
        from utils.gestor_configuracion import cargar_config_usuario, guardar_config_usuario
        cfg_previa = cargar_config_usuario()

        login = ventana.pantalla_login
        login.input_id.setText("")
        login.intentar_login()
        bloqueo_vacio_ok = not login.lbl_aviso_login.isHidden()

        login.input_id.setText("Investigador_Tesis")
        login.intentar_login()
        acceso_valido_ok = (ventana.usuario_id == "Investigador_Tesis") and login.lbl_aviso_login.isHidden()

        if cfg_previa and cfg_previa.get("usuario_id"):
            guardar_config_usuario(cfg_previa.get("usuario_id"), cfg_previa.get("ruta_drive", ""))

        paso_cp01 = bloqueo_vacio_ok and acceso_valido_ok
        resultados.append((
            "CP-01",
            "Control de acceso y validación de sesión",
            "Bloquea ID vacío y autentica 'Investigador_Tesis'",
            paso_cp01,
            time.perf_counter() - t0,
        ))

        # =====================================================================
        # CP-02: Carga, validación y detección de clases del Dataset
        # =====================================================================
        t0 = time.perf_counter()
        ruta_prep, resumen = preparar_y_analizar_dataset(dir_ds)
        clases_detectadas = resumen.get("train") or resumen.get("total_por_clase") or {}
        paso_cp02 = (
            clases_detectadas.get("felinos_prueba") == 8
            and clases_detectadas.get("caninos_prueba") == 6
        )
        resultados.append((
            "CP-02",
            "Carga y detección multiclase de Dataset",
            f"2 clases detectadas ({sum(clases_detectadas.values())} imágenes válidas)",
            paso_cp02,
            time.perf_counter() - t0,
        ))

        # =====================================================================
        # CP-03: Preprocesamiento y redimensionamiento automático (64 -> 128 px)
        # =====================================================================
        t0 = time.perf_counter()
        dir_64 = guardar_imagenes_preprocesadas_clase(
            ruta_prep, "felinos_prueba", tamano_imagen=64, ruta_base=dir_preproc
        )
        existe_64 = os.path.isdir(dir_64) and len(os.listdir(dir_64)) == 8

        # Cambiar a 128x128 y verificar que 64x64 se limpie automáticamente
        dir_128 = guardar_imagenes_preprocesadas_clase(
            ruta_prep, "felinos_prueba", tamano_imagen=128, ruta_base=dir_preproc
        )
        archivos_128 = os.listdir(dir_128) if os.path.isdir(dir_128) else []
        with Image.open(os.path.join(dir_128, archivos_128[0])) as img_verif:
            dim_ok = (img_verif.size == (128, 128))
        limpieza_ok = not os.path.exists(dir_64)
        paso_cp03 = existe_64 and len(archivos_128) == 8 and dim_ok and limpieza_ok
        resultados.append((
            "CP-03",
            "Redimensionamiento y limpieza de preprocesadas",
            "8 imgs a 128x128 px y eliminación de carpeta previa",
            paso_cp03,
            time.perf_counter() - t0,
        ))

        # =====================================================================
        # CP-04: Entrenamiento FastGAN y cálculo de métricas (FID / Diversidad)
        # =====================================================================
        t0 = time.perf_counter()
        cargador = crear_cargador_fastgan_clase(
            ruta_prep,
            clase_seleccionada="felinos_prueba",
            batch_size=4,
            tamano_imagen=64,
            iteraciones_por_epoca=2,
        )
        entrenador_64 = EntrenadorFastGAN(dispositivo="cpu", tamano_img=64)
        loss_g, loss_d = entrenador_64.entrenar_epoca(cargador, 0, 1)
        reales_np = cargar_imagenes_reales_clase(ruta_prep, "felinos_prueba", tamano_imagen=64)
        falsas_t = entrenador_64.generar_filtrado(6, factor_candidatos=2).cpu()
        falsas_np = [
            ((t.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            for t in falsas_t
        ]
        metricas = calcular_metricas_calidad(reales_np, falsas_np, dispositivo="cpu")
        paso_cp04 = (
            math.isfinite(loss_g)
            and math.isfinite(loss_d)
            and isinstance(metricas.get("fid"), (int, float))
            and "%" in str(metricas.get("diversidad", ""))
        )
        resultados.append((
            "CP-04",
            "Entrenamiento FastGAN y métricas FID/Diversidad",
            f"LossG={loss_g:.2f}, LossD={loss_d:.2f}, FID={metricas.get('fid')}, Div={metricas.get('diversidad')}",
            paso_cp04,
            time.perf_counter() - t0,
        ))

        # =====================================================================
        # CP-05: Exportación y guardado de modelo entrenado (.pth)
        # =====================================================================
        t0 = time.perf_counter()
        entrenador_128 = EntrenadorFastGAN(dispositivo="cpu", tamano_img=128)
        ruta_pth = entrenador_128.guardar_checkpoint(
            epoca=350, ruta_dir=dir_ckpt, clase_nombre="felinos_prueba"
        )
        ckpt_data = torch.load(ruta_pth, map_location="cpu", weights_only=False)
        paso_cp05 = (
            os.path.isfile(ruta_pth)
            and ckpt_data.get("epoca") == 350
            and ckpt_data.get("tamano_img") == 128
            and ckpt_data.get("clase_nombre") == "felinos_prueba"
            and "gen_ema_state_dict" in ckpt_data
        )
        resultados.append((
            "CP-05",
            "Guardado de checkpoint entrenado (.pth)",
            f"Archivo '{os.path.basename(ruta_pth)}' (128x128 px, Época 350)",
            paso_cp05,
            time.perf_counter() - t0,
        ))

        # =====================================================================
        # CP-06: Carga de modelo (.pth / Colab) con autodetección de resolución
        # =====================================================================
        t0 = time.perf_counter()
        ventana.cargar_modelo_fastgan_archivo(ruta_pth)
        paso_cp06 = (
            ventana.fastgan is not None
            and ventana.fastgan.tamano_img == 128
            and ventana.fastgan.epoca_checkpoint == 350
            and ventana.panel_control_aumento.btn_generar_aumento.isEnabled()
        )
        resultados.append((
            "CP-06",
            "Carga de modelo (.pth / Colab) sin reentrenar",
            f"Autodetectado {ventana.fastgan.tamano_img}x{ventana.fastgan.tamano_img} px · Botón Generar habilitado",
            paso_cp06,
            time.perf_counter() - t0,
        ))

        # =====================================================================
        # CP-07: Aumento de Dataset por Porcentaje (25%), filtro 4x y ZIP
        # =====================================================================
        t0 = time.perf_counter()
        ventana.panel_control_aumento.actualizar_conteo_original(8, "felinos_prueba")
        ventana.panel_control_aumento.combo_porcentaje.setCurrentText("25%")
        cant_calc = ventana.panel_control_aumento.cant_sintetica_calculada  # ceil(8 * 0.25) = 2

        gestor_aum = GestorAumentoDataset(
            ruta_original=ruta_prep,
            ruta_sinteticas=dir_sint_aum,
            ruta_ampliado=dir_amp_aum,
        )
        gestor_aum.copiar_dataset_original()
        generadas = gestor_aum.integrar_imagenes_sinteticas_clase(
            entrenador_fastgan=ventana.fastgan,
            cantidad=cant_calc,
            clase_nombre="felinos_prueba",
        )
        gestor_aum.exportar_zip_dataset_ampliado(ruta_zip_destino=ruta_zip_aum)
        total_ampliado = len(os.listdir(os.path.join(dir_amp_aum, "felinos_prueba")))
        zip_valido = os.path.isfile(ruta_zip_aum) and zipfile.is_zipfile(ruta_zip_aum)
        paso_cp07 = (cant_calc == 2 and generadas == 2 and total_ampliado == 10 and zip_valido)
        resultados.append((
            "CP-07",
            "Aumento por porcentaje (25%), filtro 4x y ZIP",
            f"8 orig + {generadas} sintéticas = {total_ampliado} total (ZIP exportado)",
            paso_cp07,
            time.perf_counter() - t0,
        ))

    # Limpiar posible carpeta preprocesadas por defecto creada durante CP-04
    from utils.cargador_datos import limpiar_imagenes_preprocesadas
    limpiar_imagenes_preprocesadas()

    # =========================================================================
    # Impresión de Reporte Académico en Consola y Guardado en Disco
    # =========================================================================
    fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lineas = [
        "=" * 124,
        f"  REPORTE DE PRUEBAS FUNCIONALES AUTOMATIZADAS - SISTEMA GANIA  ({fecha_str})",
        "=" * 124,
        f"{'ID':<6} | {'CASO DE PRUEBA FUNCIONAL':<46} | {'EVIDENCIA / RESULTADO OBTENIDO':<52} | {'TIEMPO':<7} | {'ESTADO'}",
        "-" * 124,
    ]
    exitos = 0
    tiempo_total = 0.0
    for cp_id, nombre, evidencia, ok, dur in resultados:
        estado = "PASÓ" if ok else "FALLÓ"
        if ok:
            exitos += 1
        tiempo_total += dur
        ev_corta = evidencia if len(evidencia) <= 52 else evidencia[:49] + "..."
        lineas.append(f"{cp_id:<6} | {nombre:<46} | {ev_corta:<52} | {dur:>5.2f}s  | [{estado}]")

    lineas.append("-" * 124)
    lineas.append(
        f"  RESUMEN FINAL: {exitos}/{len(resultados)} casos de prueba aprobados "
        f"({(exitos / len(resultados)) * 100:.0f}% de cobertura funcional exitosa) en {tiempo_total:.2f} s."
    )
    lineas.append("=" * 124)

    reporte_texto = "\n".join(lineas)
    print("\n" + reporte_texto + "\n")

    os.makedirs("./datos/resultados", exist_ok=True)
    ruta_reporte = "./datos/resultados/reporte_pruebas_funcionales.txt"
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write(reporte_texto + "\n")
    print(f"Reporte guardado en: {ruta_reporte}")
    return exitos == len(resultados)


if __name__ == "__main__":
    exito = ejecutar_suite_funcional()
    sys.exit(0 if exito else 1)
