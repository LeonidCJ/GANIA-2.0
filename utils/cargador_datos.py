# utils/cargador_datos.py
import os
import shutil
import zipfile
import urllib.request
import torch
from torch.utils.data import DataLoader, Subset, random_split, Dataset, RandomSampler
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
    Detecta si existe la carpeta 'train/' dentro del dataset (e.g., Oxford Flowers 102/17).
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

    Returns:
        dict: {"nombre_clase": cantidad_imagenes_validas, ...}
    """
    ruta_prep, resumen = preparar_y_analizar_dataset(ruta_dataset)

    # Extraer conteo por clase (priorizando train, pero con fallback a total_por_clase)
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

    if normalizar_fastgan:
        # GAN: se preserva la proporción (Resize del lado corto + recorte) para no deformar
        # las flores. Solo se usa flip horizontal: rotaciones (esquinas negras) o ColorJitter
        # se "filtrarían" a las imágenes generadas. El resto de aumentos lo hace DiffAugment.
        tamano_resize = int(round(tamano_imagen * 1.125))
        lista_tf = [transforms.Resize(tamano_resize)]
        if para_entrenamiento:
            lista_tf.extend([
                transforms.RandomCrop(tamano_imagen),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            lista_tf.append(transforms.CenterCrop(tamano_imagen))
    else:
        lista_tf = [
            transforms.Resize((tamano_imagen, tamano_imagen))
        ]

    lista_tf.extend([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    return transforms.Compose(lista_tf)

class DatasetClaseEspecifica(Dataset):
    """Dataset que carga EXCLUSIVAMENTE imágenes de una sola clase especificada."""
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.imgs = []

        if os.path.exists(folder):
            for file in os.listdir(folder):
                ruta_completa = os.path.join(folder, file)
                if os.path.isfile(ruta_completa) and validar_imagen_valida(ruta_completa):
                    self.imgs.append(ruta_completa)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0

def _resolver_dir_clase(ruta_dataset, clase_seleccionada=None):
    """Resuelve la carpeta física de una clase dentro del dataset."""
    ruta_base = resolver_carpeta_train_o_base(ruta_dataset)
    if clase_seleccionada and clase_seleccionada != "clase_unica":
        dir_clase = os.path.join(ruta_base, clase_seleccionada)
        if not os.path.exists(dir_clase):
            dir_clase = os.path.join(resolver_subcarpeta_raiz(ruta_dataset), clase_seleccionada)
        return dir_clase
    return ruta_base

def crear_cargador_fastgan_clase(ruta_dataset, clase_seleccionada=None, batch_size=32, tamano_imagen=64,
                                 limite_imagenes=None, iteraciones_por_epoca=None):
    """
    Crea un DataLoader optimizado para entrenar FastGAN con imágenes de UNA sola clase.

    Si se indica `iteraciones_por_epoca`, cada época tendrá exactamente ese número de lotes
    (muestreo aleatorio con reemplazo). Con datasets pequeños (p. ej. 80 imágenes) esto evita
    que una "época" sean solo 2-3 iteraciones.
    """
    transform = obtener_transformaciones(tamano_imagen, normalizar_fastgan=True, para_entrenamiento=True)

    if not os.path.exists(ruta_dataset):
        print(f"[Cargador] Ruta {ruta_dataset} no encontrada. Usando datos fallback.")
        return crear_dataloader_fallback(batch_size, tamano_imagen)

    try:
        dir_clase = _resolver_dir_clase(ruta_dataset, clase_seleccionada)
        dataset = DatasetClaseEspecifica(dir_clase, transform=transform)

        if len(dataset) == 0:
            print(f"[Cargador] No se encontraron imágenes válidas en {dir_clase}. Usando fallback.")
            return crear_dataloader_fallback(batch_size, tamano_imagen)

        if limite_imagenes and len(dataset) > limite_imagenes:
            dataset = Subset(dataset, list(range(limite_imagenes)))

        if iteraciones_por_epoca:
            sampler = RandomSampler(dataset, replacement=True, num_samples=iteraciones_por_epoca * batch_size)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    except Exception as e:
        print(f"[Cargador] Error cargando clase '{clase_seleccionada}' ({e}). Usando fallback.")
        return crear_dataloader_fallback(batch_size, tamano_imagen)

def cargar_imagenes_reales_clase(ruta_dataset, clase_seleccionada=None, tamano_imagen=64, maximo=500):
    """
    Carga TODAS las imágenes reales de la clase (sin aumentos aleatorios, recorte central)
    como arrays uint8 (H, W, C). Se usan como referencia para FID/KID/Diversidad.
    """
    from utils.tensor_utils import tensor_to_numpy
    transform = obtener_transformaciones(tamano_imagen, normalizar_fastgan=True, para_entrenamiento=False)
    try:
        dataset = DatasetClaseEspecifica(_resolver_dir_clase(ruta_dataset, clase_seleccionada), transform=transform)
        n = min(len(dataset), maximo)
        return [tensor_to_numpy(dataset[i][0]) for i in range(n)]
    except Exception as e:
        print(f"[Cargador] No se pudieron cargar imágenes reales para métricas: {e}")
        return []

def crear_cargador_fastgan(ruta_dataset, batch_size=32, tamano_imagen=64, limite_imagenes=None):
    return crear_cargador_fastgan_clase(ruta_dataset, None, batch_size, tamano_imagen, limite_imagenes)

# Aliases para retrocompatibilidad interna
crear_cargador_dcgan_clase = crear_cargador_fastgan_clase
crear_cargador_dcgan = crear_cargador_fastgan

def crear_cargadores_clasificacion(ruta_dataset, batch_size=32, tamano_imagen=64, pct_test=0.2):
    transform = obtener_transformaciones(tamano_imagen, normalizar_fastgan=False)

    if not os.path.exists(ruta_dataset):
        return crear_dataloader_fallback(batch_size, tamano_imagen), crear_dataloader_fallback(batch_size, tamano_imagen), 2

    try:
        ruta_resuelta = resolver_carpeta_train_o_base(ruta_dataset)
        subcarpetas = [d for d in os.listdir(ruta_resuelta)
                       if os.path.isdir(os.path.join(ruta_resuelta, d)) and not d.startswith('.')]

        if len(subcarpetas) > 0:
            dataset = datasets.ImageFolder(ruta_resuelta, transform=transform)
        else:
            dataset = DatasetClaseEspecifica(ruta_resuelta, transform=transform)

        n_total = len(dataset)
        if n_total < 2:
            fb = crear_dataloader_fallback(batch_size, tamano_imagen)
            return fb, fb, 2

        n_test = max(1, int(n_total * pct_test))
        n_train = n_total - n_test

        train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

        loader_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        loader_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return loader_train, loader_test, len(subcarpetas) if len(subcarpetas) > 0 else 2
    except Exception as e:
        print(f"[Cargador] Error en cargador de clasificación: {e}")
        fb = crear_dataloader_fallback(batch_size, tamano_imagen)
        return fb, fb, 2

def crear_dataloader_fallback(batch_size, size=64):
    class DummyDataset(Dataset):
        def __init__(self):
            self.data = torch.randn(100, 3, size, size)
            self.labels = torch.randint(0, 2, (100,))
        def __len__(self): return 100
        def __getitem__(self, idx): return self.data[idx], self.labels[idx]

    return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True)
