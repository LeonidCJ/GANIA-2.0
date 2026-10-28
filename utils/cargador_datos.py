# utils/cargador_datos.py
import os
import shutil
import zipfile
import urllib.request
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset, RandomSampler, random_split
from torchvision import datasets, transforms
from PIL import Image

from utils.adaptadores_dataset import GestorPreparacionDataset, validar_imagen

EXTENSIONES_VALIDAS = ('.png', '.jpg', '.jpeg', '.webp')


def validar_imagen_valida(ruta_archivo):
    """Valida la extensión y la integridad del archivo de imagen."""
    return validar_imagen(ruta_archivo)


def resolver_subcarpeta_raiz(ruta_dir):
    """Si la carpeta contiene una sola subcarpeta contenedora raíz, la resuelve."""
    if not os.path.exists(ruta_dir):
        return ruta_dir
    elementos = [e for e in os.listdir(ruta_dir) if not e.startswith('.')]
    if len(elementos) == 1 and os.path.isdir(os.path.join(ruta_dir, elementos[0])):
        return os.path.join(ruta_dir, elementos[0])
    return ruta_dir


def resolver_carpeta_train_o_base(ruta_dataset):
    """
    Detecta si existe la carpeta 'train/' dentro del dataset.
    Si existe, devuelve la ruta a 'train/'; de lo contrario, la ruta resuelta.
    """
    ruta_base = resolver_subcarpeta_raiz(ruta_dataset)
    ruta_train = os.path.join(ruta_base, "train")
    if os.path.exists(ruta_train) and os.path.isdir(ruta_train):
        return ruta_train
    return ruta_base


def preparar_y_analizar_dataset(ruta_dataset):
    """
    Ejecuta el gestor de preparación de dataset (Caso A, B, C o D),
    retornando la ruta estandarizada lista y un resumen completo.
    """
    if not os.path.exists(ruta_dataset):
        return ruta_dataset, {"formato": "Inexistente", "clases": {}, "train": {}, "invalidas": 0}

    gestor = GestorPreparacionDataset(ruta_dataset)
    ruta_preparada, resumen = gestor.analizar_y_preparar()
    return ruta_preparada, resumen


def obtener_clases_disponibles(ruta_dataset):
    """
    Detecta automáticamente las clases disponibles utilizando el gestor de preparación.
    """
    ruta_prep, resumen = preparar_y_analizar_dataset(ruta_dataset)

    clases_dict = resumen.get("train", {})
    if not clases_dict or max(clases_dict.values(), default=0) <= 1:
        clases_dict = resumen.get("total_por_clase", {})
    if not clases_dict and "clases" in resumen:
        clases_dict = resumen["clases"]

    return clases_dict


def extraer_archivo_comprimido(ruta_archivo, ruta_destino="./datos/datasets/zip_extraido"):
    """
    Descomprime un archivo comprimido (.zip, .tgz, .tar.gz, .tar) en la ruta de destino.
    """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo comprimido en: {ruta_archivo}")

    if os.path.exists(ruta_destino):
        shutil.rmtree(ruta_destino)

    os.makedirs(ruta_destino, exist_ok=True)

    ruta_lower = ruta_archivo.lower()
    if ruta_lower.endswith(('.tgz', '.tar.gz', '.tar')):
        import tarfile
        with tarfile.open(ruta_archivo, 'r:*') as tar_ref:
            tar_ref.extractall(path=ruta_destino)
        print(f"[Cargador] Archivo TAR/TGZ extraído en: {ruta_destino}")
    else:
        with zipfile.ZipFile(ruta_archivo, 'r') as zip_ref:
            zip_ref.extractall(ruta_destino)
        print(f"[Cargador] Archivo ZIP extraído en: {ruta_destino}")

    return resolver_subcarpeta_raiz(ruta_destino)


def extraer_zip_dataset(ruta_zip, ruta_destino="./datos/datasets/zip_extraido"):
    return extraer_archivo_comprimido(ruta_zip, ruta_destino)


def descargar_extraer_url(url, ruta_destino="./datos/datasets/url_extraido", callback_log=None):
    """
    Descarga un archivo comprimido (.zip, .tgz, .tar.gz) desde una URL y lo extrae.
    """
    if os.path.exists(ruta_destino):
        shutil.rmtree(ruta_destino)

    os.makedirs(ruta_destino, exist_ok=True)

    url_lower = url.lower()
    ext = ".tgz" if url_lower.endswith(('.tgz', '.tar.gz')) else ".zip"
    tmp_file = os.path.join(ruta_destino, f"temp_download{ext}")

    if callback_log:
        callback_log(f"Iniciando descarga desde URL: {url}...")

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(tmp_file, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    if callback_log:
        callback_log("Descarga completada. Extrayendo archivos comprimidos...")

    extraer_archivo_comprimido(tmp_file, ruta_destino)

    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    if callback_log:
        callback_log(f"Dataset descargado y extraído en: {ruta_destino}")

    return resolver_subcarpeta_raiz(ruta_destino)


def obtener_transformaciones(tamano_imagen=64, normalizar_fastgan=True, para_entrenamiento=False):
    norm_mean = (0.5, 0.5, 0.5) if normalizar_fastgan else (0.485, 0.456, 0.406)
    norm_std = (0.5, 0.5, 0.5) if normalizar_fastgan else (0.229, 0.224, 0.225)

    lista_tf = [
        transforms.Resize((tamano_imagen, tamano_imagen))
    ]

    if para_entrenamiento and normalizar_fastgan:
        # Nota: No usamos RandomRotation ni ColorJitter aquí porque FastGAN ya aplica
        # DiffAugment de forma diferenciable en D. Poner rotación aquí distorsiona los bordes en G.
        lista_tf.append(transforms.RandomHorizontalFlip(p=0.5))
    elif para_entrenamiento and not normalizar_fastgan:
        lista_tf.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])

    lista_tf.extend([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    return transforms.Compose(lista_tf)


class DatasetClaseEspecifica(Dataset):
    """
    Dataset optimizado con Caché en RAM:
    Lee y redimensiona las imágenes del disco UNA SOLA VEZ al iniciar el entrenamiento,
    eliminando el cuello de botella de I/O de disco durante las 500 épocas en CPU.
    """
    def __init__(self, folder, tamano_imagen=64, para_entrenamiento=True, max_cache_ram=500):
        self.folder = folder
        self.tamano_imagen = tamano_imagen
        self.para_entrenamiento = para_entrenamiento
        self.rutas = []
        self.cache_tensors = []

        if os.path.exists(folder):
            for file in sorted(os.listdir(folder)):
                ruta_completa = os.path.join(folder, file)
                if os.path.isfile(ruta_completa) and validar_imagen_valida(ruta_completa):
                    self.rutas.append(ruta_completa)

        # Transformación base que se ejecuta una sola vez al cargar en RAM
        # Transformación base que se ejecuta una sola vez al cargar en RAM (preservando proporción)
        self.imgs = self.rutas
        self.tam_cache = int(round(tamano_imagen * 1.125)) if para_entrenamiento else tamano_imagen
        tf_base = transforms.Compose([
            transforms.Resize(self.tam_cache, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.tam_cache),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Si el dataset cabe holgadamente en RAM (<= 2000 imágenes), cacheamos los tensores
        self.usar_cache = len(self.rutas) <= max_cache_ram
        if self.usar_cache:
            for r in self.rutas:
                try:
                    with Image.open(r) as img:
                        self.cache_tensors.append(tf_base(img.convert('RGB')))
                except Exception:
                    pass
        else:
            self.tf_fallback = obtener_transformaciones(tamano_imagen, normalizar_fastgan=True, para_entrenamiento=para_entrenamiento)

    def __len__(self):
        return len(self.cache_tensors) if self.usar_cache else len(self.rutas)

    def __getitem__(self, idx):
        if self.usar_cache:
            tensor = self.cache_tensors[idx]
            # Flip horizontal ultrarrápido sobre tensor en RAM
            if self.para_entrenamiento:
                extra = tensor.size(1) - self.tamano_imagen
                if extra > 0:
                    y0 = int(torch.randint(0, extra + 1, (1,)).item())
                    x0 = int(torch.randint(0, extra + 1, (1,)).item())
                    tensor = tensor[:, y0:y0 + self.tamano_imagen, x0:x0 + self.tamano_imagen]
            if self.para_entrenamiento and torch.rand(1).item() < 0.5:
                tensor = torch.flip(tensor, dims=[2])
            return tensor, 0
        else:
            img_path = self.rutas[idx]
            with Image.open(img_path) as img:
                tensor = self.tf_fallback(img.convert('RGB'))
            return tensor, 0


def _resolver_directorio_clase(ruta_dataset, clase_seleccionada=None):
    ruta_base = resolver_carpeta_train_o_base(ruta_dataset)
    if clase_seleccionada and clase_seleccionada not in ("clase_unica", "Todas", "Base"):
        dir_clase = os.path.join(ruta_base, clase_seleccionada)
        if not os.path.exists(dir_clase):
            dir_clase = os.path.join(resolver_subcarpeta_raiz(ruta_dataset), clase_seleccionada)
    else:
        dir_clase = ruta_base
    return dir_clase


_resolver_dir_clase = _resolver_directorio_clase


def cargar_imagenes_reales_clase(ruta_dataset, clase_seleccionada=None, tamano_imagen=64, max_imagenes=256):
    """
    Carga las imágenes reales de la clase seleccionada como lista de arrays numpy uint8 (RGB)
    para el cálculo de métricas FID / KID en sistema_gania.py.
    """
    dir_clase = _resolver_directorio_clase(ruta_dataset, clase_seleccionada)
    imagenes_np = []
    if not os.path.exists(dir_clase):
        return imagenes_np

    archivos = [
        os.path.join(dir_clase, f) for f in sorted(os.listdir(dir_clase))
        if os.path.isfile(os.path.join(dir_clase, f)) and validar_imagen_valida(os.path.join(dir_clase, f))
    ]

    for ruta in archivos[:max_imagenes]:
        try:
            with Image.open(ruta) as img:
                tf_real = transforms.Compose([
                    transforms.Resize(tamano_imagen, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(tamano_imagen)
                ])
                img_resized = tf_real(img.convert('RGB'))
                imagenes_np.append(np.asarray(img_resized, dtype=np.uint8))
        except Exception:
            continue

    return imagenes_np


def crear_cargador_fastgan_clase(
    ruta_dataset,
    clase_seleccionada=None,
    batch_size=32,
    tamano_imagen=64,
    limite_imagenes=None,
    iteraciones_por_epoca=None
):
    """
    Crea un DataLoader optimizado en RAM para entrenar FastGAN con imágenes de UNA sola clase.
    Soporta `iteraciones_por_epoca` mediante RandomSampler con reemplazo para datasets few-shot.
    """
    if not os.path.exists(ruta_dataset):
        print(f"[Cargador] Ruta {ruta_dataset} no encontrada. Usando datos fallback.")
        return crear_dataloader_fallback(batch_size, tamano_imagen)

    try:
        dir_clase = _resolver_directorio_clase(ruta_dataset, clase_seleccionada)
        dataset = DatasetClaseEspecifica(dir_clase, tamano_imagen=tamano_imagen, para_entrenamiento=True)

        if len(dataset) == 0:
            print(f"[Cargador] No se encontraron imágenes válidas en {dir_clase}. Usando fallback.")
            return crear_dataloader_fallback(batch_size, tamano_imagen)

        if limite_imagenes and len(dataset) > limite_imagenes:
            dataset = Subset(dataset, list(range(limite_imagenes)))

        batch_efectivo = min(batch_size, len(dataset))

        if iteraciones_por_epoca and iteraciones_por_epoca > 0:
            num_muestras_epoca = iteraciones_por_epoca * batch_efectivo
            sampler = RandomSampler(dataset, replacement=True, num_samples=num_muestras_epoca)
            return DataLoader(dataset, batch_size=batch_efectivo, sampler=sampler, drop_last=True)

        return DataLoader(dataset, batch_size=batch_efectivo, shuffle=True, drop_last=False)

    except Exception as e:
        print(f"[Cargador] Error cargando clase '{clase_seleccionada}' ({e}). Usando fallback.")
        return crear_dataloader_fallback(batch_size, tamano_imagen)


def crear_cargador_fastgan(ruta_dataset, batch_size=32, tamano_imagen=64, limite_imagenes=None):
    return crear_cargador_fastgan_clase(ruta_dataset, None, batch_size, tamano_imagen, limite_imagenes)


def crear_cargadores_clasificacion(ruta_dataset, batch_size=32, tamano_imagen=64, val_split=0.2):
    """
    Crea los DataLoaders (train_loader, test_loader, num_clases) para evaluar el clasificador
    en el dataset Original vs Ampliado.
    """
    tf_train = obtener_transformaciones(tamano_imagen, normalizar_fastgan=False, para_entrenamiento=True)
    tf_test = obtener_transformaciones(tamano_imagen, normalizar_fastgan=False, para_entrenamiento=False)

    ruta_base = resolver_subcarpeta_raiz(ruta_dataset)
    ruta_train = os.path.join(ruta_base, "train")
    ruta_val = os.path.join(ruta_base, "val")
    if not os.path.exists(ruta_val):
        ruta_val = os.path.join(ruta_base, "test")

    if os.path.exists(ruta_train) and os.path.exists(ruta_val):
        ds_train = datasets.ImageFolder(ruta_train, transform=tf_train)
        ds_test = datasets.ImageFolder(ruta_val, transform=tf_test)
        n_clases = len(ds_train.classes)
    else:
        carpeta_objetivo = ruta_train if os.path.exists(ruta_train) else ruta_base
        ds_completo = datasets.ImageFolder(carpeta_objetivo, transform=tf_train)
        n_clases = len(ds_completo.classes)
        n_test = max(1, int(len(ds_completo) * val_split))
        n_train = max(1, len(ds_completo) - n_test)
        ds_train, ds_test = random_split(
            ds_completo, [n_train, n_test],
            generator=torch.Generator().manual_seed(42)
        )

    loader_tr = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_te = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    return loader_tr, loader_te, n_clases


# Aliases de compatibilidad
_resolver_dir_clase = _resolver_directorio_clase
crear_cargador_dcgan_clase = crear_cargador_fastgan_clase
crear_cargador_dcgan = crear_cargador_fastgan


def crear_dataloader_fallback(batch_size, size=64):
    class DummyDataset(Dataset):
        def __init__(self):
            self.data = torch.randn(100, 3, size, size)
            self.labels = torch.randint(0, 2, (100,))
        def __len__(self): return 100
        def __getitem__(self, idx): return self.data[idx], self.labels[idx]

    return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True)