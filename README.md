# GANIA 2.0 - Entorno de Análisis Generativo con IA

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PyQt5](https://img.shields.io/badge/PyQt5-41CD52?style=for-the-badge&logo=qt&logoColor=white)

GANIA 2.0 es una plataforma avanzada desarrollada en Python para el entrenamiento, visualización y análisis de Redes Neuronales Generativas (GANs). Este proyecto forma parte de una investigación de tesis y busca proporcionar una interfaz intuitiva para experimentar con arquitecturas de vanguardia en el campo del aprendizaje profundo generativo.

## 🚀 Características Principales

- **Arquitecturas Implementadas:**
    - **DCGAN (Deep Convolutional GAN):** Generación de imágenes desde ruido aleatorio.
    - **CycleGAN:** Traducción de imagen a imagen (estilo/dominio) sin necesidad de pares emparejados.
    - **Generador Combinado:** Modelo híbrido para experimentación avanzada.
- **Interfaz Gráfica Neumórfica:** UI moderna construida con PyQt5 con soporte para temas claro y oscuro.
- **Entrenamiento Asíncrono:** Procesamiento en hilos separados para mantener la fluidez de la interfaz.
- **Monitoreo en Tiempo Real:** Visualización de muestras generadas y métricas de pérdida durante el entrenamiento.
- **Sistema de Checkpoints:** Guardado automático del estado de los modelos para reanudación y evaluación posterior.
- **Gestión de Datos:** Cargadores optimizados para CIFAR-10 y datasets personalizados (ej. Orange dataset).

## 🛠️ Tecnologías

- **Lenguaje:** Python 3.8+
- **Framework de IA:** PyTorch / Torchvision
- **Interfaz Gráfica:** PyQt5
- **Procesamiento de Imágenes:** Pillow, OpenCV, NumPy
- **Visualización:** Matplotlib

## 📂 Estructura del Proyecto

```text
GANIA 2.0/
├── aplicacion/         # Lógica de control y gestión de hilos
├── datos/              # Almacenamiento de datasets
├── interfaces/         # Componentes de la interfaz de usuario (PyQt5)
├── modelos/            # Implementaciones de DCGAN, CycleGAN, etc.
├── resultados/         # Muestras generadas y checkpoints (.pth)
├── utils/              # Herramientas de carga de datos y gestión de archivos
├── config.py           # Configuración centralizada del sistema
├── main.py             # Punto de entrada de la aplicación
└── requirements.txt    # Dependencias del proyecto
```

## ⚙️ Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/GANIA-2.0.git
   cd GANIA-2.0
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv venv
   # En Windows:
   .\venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Uso

Para iniciar la aplicación, simplemente ejecuta el archivo principal:

```bash
python main.py
```

## 📈 Mejoras en Desarrollo (Roadmap)

- [ ] Implementación de métricas cuantitativas (FID, Inception Score).
- [ ] Gráficas de pérdida (Loss) integradas en tiempo real en la UI.
- [ ] Exportación de reportes de entrenamiento en PDF.
- [ ] Soporte para arquitecturas adicionales (StyleGAN, WGAN-GP).

---
**Nota:** Este proyecto está en desarrollo activo como parte de un trabajo de tesis académica.
