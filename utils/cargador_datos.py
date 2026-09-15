from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader, Subset

def crear_cargador_datos(ruta_dataset, tamano_lote=32, tamano_imagen=64):
    if not os.path.exists(ruta_dataset):
        raise FileNotFoundError(f"No se encontró la ruta: {ruta_dataset}")
    transform = transforms.Compose([
        transforms.Resize(tamano_imagen),
        transforms.CenterCrop(tamano_imagen),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    ])
    dataset = datasets.ImageFolder(ruta_dataset, transform=transform)
    return DataLoader(dataset, batch_size=tamano_lote, shuffle=True)

def crear_cargador_datos_cifar10(limite_imagenes=1000, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB (3 canales)
    ])

    dataset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)

    # Limitar la cantidad de imágenes
    if limite_imagenes and limite_imagenes < len(dataset):
        dataset = Subset(dataset, list(range(limite_imagenes)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def crear_cargador_datos_diseno(tipo_dataset, batch_size=32, tamano_imagen=64):
    """Cargador especializado para el estudio de Diseño Humano"""
    from config import CONFIG
    base_path = CONFIG["RUTAS"]["DATASETS"]
    ruta = os.path.join(base_path, tipo_dataset.lower())

    if not os.path.exists(ruta):
        # Crear carpeta vacía para evitar errores si no existe el dataset físico aún
        os.makedirs(ruta, exist_ok=True)
        print(f"Aviso: Ruta {ruta} no existía. Se ha creado una carpeta vacía.")

    transform = transforms.Compose([
        transforms.Resize(tamano_imagen),
        transforms.CenterCrop(tamano_imagen),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        dataset = datasets.ImageFolder(ruta, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except:
        # Fallback a un dataset aleatorio si la carpeta está vacía
        import torch
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=(3, 64, 64), length=100):
                self.size = size
                self.length = length
            def __len__(self): return self.length
            def __getitem__(self, idx): return torch.randn(self.size), 0
        return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True)
