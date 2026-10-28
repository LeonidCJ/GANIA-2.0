# utils/adaptadores_dataset.py
import os
import shutil
import json
import csv
import numpy as np
from pathlib import Path

EXTENSIONES_VALIDAS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')


def validar_imagen(ruta):
    """
    Verificación ultrarrápida de archivo de imagen válida sin bloquear el hilo de E/S.
    Comprueba existencia, extensión y tamaño mínimo (> 64 bytes).
    """
    try:
        return (
            os.path.isfile(ruta)
            and ruta.lower().endswith(EXTENSIONES_VALIDAS)
            and os.path.getsize(ruta) > 64
        )
    except OSError:
        return False


def buscar_archivos_por_extension(dir_base, extensiones=('.mat',), max_profundidad=3):
    """Busca recursivamente archivos con ciertas extensiones hasta max_profundidad niveles."""
    encontrados = []
    dir_base_str = str(dir_base)
    prof_base = dir_base_str.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(dir_base_str):
        prof_actual = root.rstrip(os.sep).count(os.sep) - prof_base
        if prof_actual >= max_profundidad:
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            if f.lower().endswith(extensiones):
                encontrados.append(Path(root) / f)
    return encontrados


class AdaptadorBase:
    """Clase base abstracta para adaptadores de datasets."""
    def __init__(self, ruta_origen, ruta_destino="./datos/datasets/dataset_prepared"):
        self.ruta_origen = Path(ruta_origen)
        self.ruta_destino = Path(ruta_destino)

    def puede_procesar(self):
        raise NotImplementedError

    def analizar(self):
        raise NotImplementedError

    def preparar(self):
        raise NotImplementedError


class AdaptadorOxford17(AdaptadorBase):
    """
    Maneja específicamente el dataset Oxford Flowers 17
    (17 clases x 80 imágenes = 1360 imágenes image_0001.jpg .. image_1360.jpg).
    """
    CLASS_NAMES = [
        "daffodil", "snowdrop", "lily_valley", "bluebell", "crocus",
        "iris", "tigerlily", "tulip", "fritillary", "sunflower",
        "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
        "windflower", "pansy"
    ]

    def puede_procesar(self):
        if not self.ruta_origen.exists():
            return False
        nombre_ruta = str(self.ruta_origen).lower()
        dir_imgs, archivos = self._obtener_archivos_imagenes()
        if "17flowers" in nombre_ruta and len(archivos) >= 1360:
            return True
        # Solo activar si es una carpeta plana de exactamente 1360 imágenes con prefijo image_
        if len(archivos) == 1360 and archivos[0].lower().startswith("image_"):
            return True
        return False

    def _obtener_archivos_imagenes(self):
        for root, dirs, files in os.walk(self.ruta_origen):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            validos = [f for f in files if f.lower().endswith(EXTENSIONES_VALIDAS)]
            if len(validos) >= 1360:
                validos.sort()
                return Path(root), validos

        try:
            validos = [
                f for f in os.listdir(self.ruta_origen)
                if (self.ruta_origen / f).is_file() and f.lower().endswith(EXTENSIONES_VALIDAS)
            ]
            validos.sort()
            return self.ruta_origen, validos
        except OSError:
            return self.ruta_origen, []

    def analizar(self):
        resumen = {
            "train": {}, "val": {}, "test": {}, "total_por_clase": {},
            "invalidas": 0,
            "formato": "Caso D: Oxford Flowers 17 (17 Clases x 80 imgs)"
        }
        dir_imgs, archivos = self._obtener_archivos_imagenes()

        imgs_per_class = 80
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            start = class_idx * imgs_per_class
            end = (class_idx + 1) * imgs_per_class
            class_files = archivos[start:end]

            valid_cnt = 0
            for f in class_files:
                if validar_imagen(str(dir_imgs / f)):
                    valid_cnt += 1
                else:
                    resumen["invalidas"] += 1

            resumen["train"][class_name] = valid_cnt
            resumen["total_por_clase"][class_name] = valid_cnt

        return resumen

    def preparar(self):
        resumen = {
            "train": {}, "val": {}, "test": {}, "total_por_clase": {},
            "invalidas": 0,
            "formato": "Caso D: Oxford Flowers 17 (17 Clases x 80 imgs)"
        }
        dir_imgs, archivos = self._obtener_archivos_imagenes()

        if self.ruta_destino.exists():
            shutil.rmtree(self.ruta_destino)

        imgs_per_class = 80
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            dest_dir = self.ruta_destino / "train" / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            start = class_idx * imgs_per_class
            end = (class_idx + 1) * imgs_per_class
            class_files = archivos[start:end]

            valid_cnt = 0
            for f in class_files:
                src_path = dir_imgs / f
                if validar_imagen(str(src_path)):
                    shutil.copy2(src_path, dest_dir / f)
                    valid_cnt += 1
                else:
                    resumen["invalidas"] += 1

            resumen["train"][class_name] = valid_cnt
            resumen["total_por_clase"][class_name] = valid_cnt

        return str(self.ruta_destino), resumen


class AdaptadorEstructuraCarpeta(AdaptadorBase):
    """
    Maneja Caso A (carpetas directas por clase: dataset/gatos/)
    y Caso B (particiones train/val/test: dataset/train/gatos/).
    """
    def puede_procesar(self):
        if not self.ruta_origen.exists():
            return False
        subdirs = [d for d in os.listdir(self.ruta_origen) if (self.ruta_origen / d).is_dir() and not d.startswith('.')]
        return len(subdirs) > 0

    def analizar(self):
        resumen = {"train": {}, "val": {}, "test": {}, "total_por_clase": {}, "invalidas": 0, "formato": "A/B (Carpetas)"}

        # Si la carpeta raíz tiene una única subcarpeta contenedora, descender a ella
        ruta_efectiva = self.ruta_origen
        subdirs_raw = [d for d in os.listdir(ruta_efectiva) if (ruta_efectiva / d).is_dir() and not d.startswith('.')]
        if len(subdirs_raw) == 1 and subdirs_raw[0].lower() not in ("train", "val", "test", "valid"):
            sub_unica = ruta_efectiva / subdirs_raw[0]
            sub_internas = [d for d in os.listdir(sub_unica) if (sub_unica / d).is_dir() and not d.startswith('.')]
            if len(sub_internas) >= 1:
                ruta_efectiva = sub_unica
                self.ruta_origen = ruta_efectiva

        mapa_subdirs = {d.lower(): d for d in os.listdir(ruta_efectiva) if (ruta_efectiva / d).is_dir() and not d.startswith('.')}
        particiones_encontradas = [p for p in ["train", "val", "test"] if p in mapa_subdirs]

        if particiones_encontradas:
            resumen["formato"] = "Caso B: Particiones (train/val/test)"
            for p in particiones_encontradas:
                dir_particion = ruta_efectiva / mapa_subdirs[p]
                for c in sorted(os.listdir(dir_particion)):
                    dir_clase = dir_particion / c
                    if dir_clase.is_dir() and not c.startswith('.'):
                        validas = 0
                        with os.scandir(dir_clase) as entries:
                            for entry in entries:
                                if entry.is_file():
                                    if entry.name.lower().endswith(EXTENSIONES_VALIDAS):
                                        validas += 1
                                    else:
                                        resumen["invalidas"] += 1
                        if validas > 0:
                            resumen[p][c] = validas
                            resumen["total_por_clase"][c] = resumen["total_por_clase"].get(c, 0) + validas
        else:
            resumen["formato"] = "Caso A: Carpetas directas por clase"
            for c in sorted(os.listdir(ruta_efectiva)):
                dir_clase = ruta_efectiva / c
                if dir_clase.is_dir() and not c.startswith('.'):
                    validas = 0
                    with os.scandir(dir_clase) as entries:
                        for entry in entries:
                            if entry.is_file():
                                if entry.name.lower().endswith(EXTENSIONES_VALIDAS):
                                    validas += 1
                                else:
                                    resumen["invalidas"] += 1
                    if validas > 0:
                        resumen["train"][c] = validas
                        resumen["total_por_clase"][c] = validas

        return resumen

    def preparar(self):
        resumen = self.analizar()
        return str(self.ruta_origen), resumen


class AdaptadorCSV_JSON(AdaptadorBase):
    """
    Maneja Caso C (imágenes + labels.csv / labels.json / labels.txt).
    """
    def puede_procesar(self):
        if not self.ruta_origen.exists():
            return False
        archivos = [f.name.lower() for f in buscar_archivos_por_extension(self.ruta_origen, ('.csv', '.json', '.txt'))]
        return any('label' in a or 'class' in a or 'target' in a for a in archivos)

    def analizar(self):
        resumen = {"train": {}, "val": {}, "test": {}, "total_por_clase": {}, "invalidas": 0, "formato": "Caso C: Etiquetas CSV/JSON"}
        mapa_labels = self._leer_etiquetas()
        dir_images = self._encontrar_directorio_imagenes()

        for img_name, label in mapa_labels.items():
            img_path = dir_images / img_name
            c_name = f"clase_{label}"
            if validar_imagen(str(img_path)):
                resumen["train"][c_name] = resumen["train"].get(c_name, 0) + 1
                resumen["total_por_clase"][c_name] = resumen["total_por_clase"].get(c_name, 0) + 1
            else:
                resumen["invalidas"] += 1

        return resumen

    def _encontrar_directorio_imagenes(self):
        for sub in ["images", "img", "jpg", "png", "data"]:
            if (self.ruta_origen / sub).exists() and (self.ruta_origen / sub).is_dir():
                return self.ruta_origen / sub
        return self.ruta_origen

    def _leer_etiquetas(self):
        mapa = {}
        archivos_meta = buscar_archivos_por_extension(self.ruta_origen, ('.csv', '.json', '.txt'))

        for f_path in archivos_meta:
            f_name = f_path.name.lower()
            if f_name.endswith('.csv') and ('label' in f_name or 'class' in f_name or 'target' in f_name):
                with open(f_path, 'r', encoding='utf-8', errors='ignore') as file:
                    reader = csv.reader(file)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            mapa[row[0].strip()] = row[1].strip()
            elif f_name.endswith('.json') and ('label' in f_name or 'class' in f_name or 'target' in f_name):
                with open(f_path, 'r', encoding='utf-8', errors='ignore') as file:
                    data = json.load(file)
                    if isinstance(data, dict):
                        mapa = {k: str(v) for k, v in data.items()}
        return mapa

    def preparar(self):
        resumen = {"train": {}, "val": {}, "test": {}, "total_por_clase": {}, "invalidas": 0, "formato": "Caso C: Etiquetas CSV/JSON"}
        mapa = self._leer_etiquetas()
        dir_images = self._encontrar_directorio_imagenes()

        if self.ruta_destino.exists():
            shutil.rmtree(self.ruta_destino)

        dir_train = self.ruta_destino / "train"
        dir_train.mkdir(parents=True, exist_ok=True)

        for img_name, label in mapa.items():
            src_path = dir_images / img_name
            c_name = f"clase_{label}"
            if validar_imagen(str(src_path)):
                dest_dir = dir_train / c_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_dir / img_name)
                resumen["train"][c_name] = resumen["train"].get(c_name, 0) + 1
                resumen["total_por_clase"][c_name] = resumen["total_por_clase"].get(c_name, 0) + 1
            else:
                resumen["invalidas"] += 1

        return str(self.ruta_destino), resumen


class AdaptadorMatGeneral(AdaptadorBase):
    """
    Maneja Caso D: Datasets con metadatos .mat (e.g. Oxford Flowers 102).
    """
    def puede_procesar(self):
        if not self.ruta_origen.exists():
            return False
        archivos_mat = buscar_archivos_por_extension(self.ruta_origen, ('.mat',))
        if not archivos_mat:
            return False

        labels, _ = self._leer_mat_labels_y_setid()
        return labels is not None and len(labels) > 1

    def _encontrar_directorio_imagenes(self):
        for sub in ["jpg", "jpeg", "png", "images", "img", "102flowers"]:
            d = self.ruta_origen / sub
            if d.exists() and d.is_dir():
                return d

        max_imgs = 0
        mejor_dir = self.ruta_origen
        for root, dirs, files in os.walk(self.ruta_origen):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            count = sum(1 for f in files if f.lower().endswith(EXTENSIONES_VALIDAS))
            if count > max_imgs:
                max_imgs = count
                mejor_dir = Path(root)

        return mejor_dir

    def _obtener_lista_imagenes_ordenada(self):
        dir_img = self._encontrar_directorio_imagenes()
        archivos = [f for f in os.listdir(dir_img) if f.lower().endswith(EXTENSIONES_VALIDAS)]
        archivos.sort()
        return dir_img, archivos

    def _leer_mat_labels_y_setid(self):
        labels = None
        setids = None

        archivos_mat = buscar_archivos_por_extension(self.ruta_origen, ('.mat',))
        if not archivos_mat:
            return None, None

        try:
            from scipy import io

            mat_labels_file = None
            mat_setid_file = None

            for f in archivos_mat:
                fname = f.name.lower()
                if any(k in fname for k in ('imagelabels', 'labels', 'cls', 'category', 'target')):
                    mat_labels_file = f
                elif 'setid' in fname or 'split' in fname:
                    mat_setid_file = f

            if not mat_labels_file and archivos_mat:
                mat_labels_file = archivos_mat[0]

            if mat_labels_file:
                data = io.loadmat(str(mat_labels_file))
                posibles_keys = ['labels', 'label', 'class', 'target', 'category', 'y', 'cls']

                labels_arr = None
                for k in posibles_keys:
                    if k in data:
                        labels_arr = data[k]
                        break

                if labels_arr is None:
                    for k, v in data.items():
                        if not k.startswith('__') and isinstance(v, np.ndarray) and v.size > 1:
                            labels_arr = v
                            break

                if labels_arr is not None:
                    labels = labels_arr.flatten()

            if mat_setid_file:
                data_s = io.loadmat(str(mat_setid_file))
                setids = {}
                for k_tr in ['trnid', 'train', 'tr']:
                    if k_tr in data_s:
                        setids['tr'] = set(int(x) for x in data_s[k_tr].flatten())
                for k_val in ['valid', 'val', 'v']:
                    if k_val in data_s:
                        setids['val'] = set(int(x) for x in data_s[k_val].flatten())
                for k_tst in ['tstid', 'test', 'tst', 'ts']:
                    if k_tst in data_s:
                        setids['tst'] = set(int(x) for x in data_s[k_tst].flatten())

        except Exception as e:
            print(f"[MatAdapter] Error procesando archivos .mat: {e}")

        return labels, setids

    def analizar(self):
        resumen = {"train": {}, "val": {}, "test": {}, "total_por_clase": {}, "invalidas": 0, "formato": "Caso D: Metadatos en .MAT"}
        labels, setids = self._leer_mat_labels_y_setid()
        dir_img, lista_imgs = self._obtener_lista_imagenes_ordenada()

        if labels is not None:
            tr_set = setids.get("tr", set()) if setids else set()
            val_set = setids.get("val", set()) if setids else set()
            tst_set = setids.get("tst", set()) if setids else set()

            for idx, label_id in enumerate(labels, start=1):
                part = "train"
                if setids:
                    if idx in tr_set:
                        part = "train"
                    elif idx in val_set:
                        part = "val"
                    elif idx in tst_set:
                        part = "test"

                c_name = f"clase_{int(label_id):03d}"
                img_name_oxford = f"image_{idx:05d}.jpg"
                img_path = dir_img / img_name_oxford

                if not img_path.exists() and (idx - 1) < len(lista_imgs):
                    img_path = dir_img / lista_imgs[idx - 1]

                if validar_imagen(str(img_path)):
                    resumen[part][c_name] = resumen[part].get(c_name, 0) + 1
                    resumen["total_por_clase"][c_name] = resumen["total_por_clase"].get(c_name, 0) + 1
                else:
                    resumen["invalidas"] += 1

        return resumen

    def preparar(self):
        resumen = {"train": {}, "val": {}, "test": {}, "total_por_clase": {}, "invalidas": 0, "formato": "Caso D: Metadatos en .MAT"}
        labels, setids = self._leer_mat_labels_y_setid()
        dir_img, lista_imgs = self._obtener_lista_imagenes_ordenada()

        if self.ruta_destino.exists():
            shutil.rmtree(self.ruta_destino)

        if labels is not None:
            tr_set = setids.get("tr", set()) if setids else set()
            val_set = setids.get("val", set()) if setids else set()
            tst_set = setids.get("tst", set()) if setids else set()

            for idx, label_id in enumerate(labels, start=1):
                part = "train"
                if setids:
                    if idx in tr_set:
                        part = "train"
                    elif idx in val_set:
                        part = "val"
                    elif idx in tst_set:
                        part = "test"

                c_name = f"clase_{int(label_id):03d}"
                img_name_oxford = f"image_{idx:05d}.jpg"
                src_path = dir_img / img_name_oxford

                if not src_path.exists() and (idx - 1) < len(lista_imgs):
                    src_path = dir_img / lista_imgs[idx - 1]

                if validar_imagen(str(src_path)):
                    dest_dir = self.ruta_destino / part / c_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dest_dir / src_path.name)
                    resumen[part][c_name] = resumen[part].get(c_name, 0) + 1
                    resumen["total_por_clase"][c_name] = resumen["total_por_clase"].get(c_name, 0) + 1
                else:
                    resumen["invalidas"] += 1

        return str(self.ruta_destino), resumen


class GestorPreparacionDataset:
    """
    Analizador y Gestor principal de Adaptación de Datasets para GANIA 2.0.
    """
    def __init__(self, ruta_origen, ruta_destino="./datos/datasets/dataset_prepared"):
        self.ruta_origen = Path(ruta_origen)
        self.ruta_destino = Path(ruta_destino)
        self.adaptadores = [
            AdaptadorOxford17(ruta_origen, ruta_destino),
            AdaptadorMatGeneral(ruta_origen, ruta_destino),
            AdaptadorCSV_JSON(ruta_origen, ruta_destino),
            AdaptadorEstructuraCarpeta(ruta_origen, ruta_destino)
        ]

    def seleccionar_adaptador(self):
        for ad in self.adaptadores:
            if ad.puede_procesar():
                return ad
        return AdaptadorEstructuraCarpeta(self.ruta_origen, self.ruta_destino)

    def analizar_y_preparar(self):
        adaptador = self.seleccionar_adaptador()
        print(f"[GestorPreparacion] Adaptador seleccionado: {adaptador.__class__.__name__}")
        ruta_preparada, resumen = adaptador.preparar()

        totales = resumen.get("total_por_clase", {})
        if totales:
            promedio_imgs = sum(totales.values()) / max(1, len(totales))
            if promedio_imgs <= 1.0 and not isinstance(adaptador, AdaptadorEstructuraCarpeta):
                fb = AdaptadorEstructuraCarpeta(self.ruta_origen, self.ruta_destino)
                if fb.puede_procesar():
                    print("[GestorPreparacion] Fallback a AdaptadorEstructuraCarpeta por conteo nulo en metadatos.")
                    return fb.preparar()

        return ruta_preparada, resumen
