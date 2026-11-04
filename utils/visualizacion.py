import matplotlib.pyplot as plt
from torchvision.utils import save_image


def mostrar_imagen_tensor(imagen_tensor, titulo="Imagen generada"):
    """
    Muestra una imagen en tensor (C, H, W) usando matplotlib.
    """
    imagen_np = imagen_tensor.detach().cpu().permute(1, 2, 0).numpy()
    imagen_np = (imagen_np * 0.5 + 0.5).clip(0, 1)

    plt.imshow(imagen_np)
    plt.title(titulo)
    plt.axis("off")
    plt.show()


def guardar_imagenes_batch(batch, ruta, normalizar=True, nrow=8):
    """
    Guarda un lote de imágenes (tensor) como una cuadrícula.
    """
    save_image(batch, ruta, normalize=normalizar, nrow=nrow)


def mostrar_evolucion_perdidas(lista_perdidas, nombres, titulo="Pérdidas durante entrenamiento"):
    """
    Dibuja la evolución de las pérdidas durante el entrenamiento.
    """
    plt.figure(figsize=(10, 5))
    for perdidas, nombre in zip(lista_perdidas, nombres):
        plt.plot(perdidas, label=nombre)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
