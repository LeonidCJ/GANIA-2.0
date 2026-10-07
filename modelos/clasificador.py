# modelos/clasificador.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class CNN_Simple(nn.Module):
    """Red Neuronal Convolucional ligera para clasificación."""
    def __init__(self, num_clases=2, canales_img=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(canales_img, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_clases)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def crear_modelo_clasificador(arquitectura="ResNet18", num_clases=2, pretrained=True):
    if arquitectura == "ResNet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            modelo = models.resnet18(weights=weights)
            num_ftrs = modelo.fc.in_features
            modelo.fc = nn.Linear(num_ftrs, num_clases)
            return modelo
        except Exception as e:
            print(f"[Clasificador] Error cargando ResNet18 ({e}). Usando CNN_Simple fallback.")
            return CNN_Simple(num_clases=num_clases)
    else:
        return CNN_Simple(num_clases=num_clases)

class EntrenadorClasificador:
    def __init__(self, dispositivo, arquitectura="ResNet18", num_clases=2, lr=0.001):
        self.dispositivo = dispositivo
        self.arquitectura = arquitectura
        self.num_clases = num_clases
        self.modelo = crear_modelo_clasificador(arquitectura, num_clases).to(dispositivo)
        self.criterio = nn.CrossEntropyLoss()
        self.optimizador = optim.Adam(self.modelo.parameters(), lr=lr)

    def entrenar_epoca(self, loader_train):
        self.modelo.train()
        total_loss = 0.0
        correctos = 0
        total = 0

        for imgs, labels in loader_train:
            imgs, labels = imgs.to(self.dispositivo), labels.to(self.dispositivo)
            self.optimizador.zero_grad()
            salidas = self.modelo(imgs)
            loss = self.criterio(salidas, labels)
            loss.backward()
            self.optimizador.step()

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(salidas, 1)
            correctos += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        loss_avg = total_loss / max(1, total)
        acc_avg = correctos / max(1, total)
        return loss_avg, acc_avg

    def evaluar(self, loader_test):
        """
        Evalúa el clasificador y devuelve predicciones, etiquetas verdaderas y métricas completas.
        """
        self.modelo.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for imgs, labels in loader_test:
                imgs = imgs.to(self.dispositivo)
                salidas = self.modelo(imgs)
                _, preds = torch.max(salidas, 1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        metricas = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "matriz_confusion": cm,
            "y_true": y_true,
            "y_pred": y_pred
        }
        return metricas
