import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import traceback
from torchvision.utils import save_image


class SLE(nn.Module):
    """
    Skip-Layer Excitation Module (FastGAN ICLR 2021).
    Realiza excitación por canal entre características de baja y alta resolución.
    """
    def __init__(self, in_channels_low, in_channels_high, out_channels_high):
        super().__init__()
        self.map = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(in_channels_low, in_channels_low, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels_low, out_channels_high, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_low, x_high):
        weight = self.map(x_low)
        return x_high * weight


class GeneradorFastGAN(nn.Module):
    """
    Generador FastGAN con Upsampling por Vecino Más Cercano + Convolución 3x3
    para eliminar artefactos de damero/rejilla sin difuminar la imagen.
    """
    def __init__(self, dim_latente=100, canales_img=3):
        super().__init__()
        self.dim_latente = dim_latente

        # Bloque Inicial: Latente 1x1 -> 4x4
        self.init = nn.Sequential(
            nn.ConvTranspose2d(dim_latente, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        # Bloque 4x4 -> 8x8
        self.feat_8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # Bloque 8x8 -> 16x16
        self.feat_16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        # Bloque 16x16 -> 32x32
        self.feat_32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        # Bloque 32x32 -> 64x64
        self.feat_64 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        # Módulo SLE (conecta características 8x8 con 64x64)
        self.sle = SLE(in_channels_low=256, in_channels_high=32, out_channels_high=32)

        # Capa Final de Salida (RGB, Tanh)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(32, canales_img, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        out_4 = self.init(z)
        out_8 = self.feat_8(out_4)
        out_16 = self.feat_16(out_8)
        out_32 = self.feat_32(out_16)
        out_64 = self.feat_64(out_32)

        # Aplicar excitación Skip-Layer desde 8x8 a 64x64
        out_64 = self.sle(out_8, out_64)

        return self.to_rgb(out_64)


class DecodificadorD(nn.Module):
    """
    Decodificador Auxiliar del Discriminador para Pérdida de Reconstrucción Perceptiva (Self-Supervised).
    """
    def __init__(self, in_channels=128, out_channels=3):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(64, out_channels, 3, 1, 1, bias=False)),
            nn.Tanh()
        )

    def forward(self, feat):
        return self.dec(feat)


class DiscriminadorFastGAN(nn.Module):
    """
    Discriminador FastGAN con Normalización Espectral.
    """
    def __init__(self, canales_img=3):
        super().__init__()

        # Capa Entrada 64x64 -> 32x32
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(canales_img, 32, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 32x32 -> 16x16
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(32, 64, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 16x16 -> 8x8
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 8x8 -> 4x4
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Capa de Logit Adversarial (Scoring)
        self.logits = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 1, 4, 1, 0, bias=False))
        )

        # Decodificador Auxiliar Autosupervisado
        self.decoder = DecodificadorD(in_channels=128, out_channels=canales_img)

    def forward(self, img, rec=False):
        f1 = self.conv1(img)  # 32x32
        f2 = self.conv2(f1)   # 16x16
        f3 = self.conv3(f2)   # 8x8
        f4 = self.conv4(f3)   # 4x4

        out_logit = self.logits(f4).view(img.size(0), -1).mean(1)

        if rec:
            rec_img = self.decoder(f3)
            img_patch = F.interpolate(img, size=(32, 32), mode='nearest')
            return out_logit, rec_img, img_patch

        return out_logit


class EntrenadorFastGAN:
    """
    Entrenador de FastGAN con Hinge Loss + Normalización Espectral + Pérdida Perceptiva (0.2).
    """
    def __init__(self, dispositivo, dim_latente=100, canales_img=3, lr=0.0002):
        self.dispositivo = dispositivo
        self.dim_latente = dim_latente
        self.generador = GeneradorFastGAN(dim_latente, canales_img).to(dispositivo)
        self.discriminador = DiscriminadorFastGAN(canales_img).to(dispositivo)

        self.opt_gen = torch.optim.Adam(self.generador.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_dis = torch.optim.Adam(self.discriminador.parameters(), lr=lr, betas=(0.5, 0.999))

        self.ruido_fijo = torch.randn(64, dim_latente, 1, 1, device=dispositivo)

    def guardar_checkpoint(self, epoca, ruta_dir="./datos/resultados/checkpoints_dcgan"):
        os.makedirs(ruta_dir, exist_ok=True)
        ruta = os.path.join(ruta_dir, f"fastgan_epoca_{epoca}.pth")
        torch.save({
            'epoca': epoca,
            'gen_state_dict': self.generador.state_dict(),
            'dis_state_dict': self.discriminador.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'opt_dis_state_dict': self.opt_dis.state_dict(),
        }, ruta)
        print(f"[FastGAN] Checkpoint guardado en: {ruta}")
        return ruta

    def cargar_checkpoint(self, ruta):
        checkpoint = torch.load(ruta, map_location=self.dispositivo)
        self.generador.load_state_dict(checkpoint['gen_state_dict'])
        self.discriminador.load_state_dict(checkpoint['dis_state_dict'])
        self.opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        self.opt_dis.load_state_dict(checkpoint['opt_dis_state_dict'])
        print(f"[FastGAN] Checkpoint cargado desde: {ruta}")
        return checkpoint['epoca']

    def entrenar_epoca(self, cargador_datos, epoca, num_epocas, abort_callback=None):
        try:
            self.generador.train()
            self.discriminador.train()

            total_loss_G = 0.0
            total_loss_D = 0.0
            batches = 0

            for i, (imgs_reales, _) in enumerate(cargador_datos):
                if abort_callback and abort_callback():
                    return 0, 0

                imgs_reales = imgs_reales.to(self.dispositivo)
                batch_size = imgs_reales.size(0)

                # === 1. Entrenamiento Discriminador (Hinge Loss + Self-Supervised Rec Loss 0.2) ===
                self.opt_dis.zero_grad()

                real_logits, rec_img, real_patch = self.discriminador(imgs_reales, rec=True)
                loss_d_real = F.relu(1.0 - real_logits).mean()
                loss_rec = F.mse_loss(rec_img, real_patch)

                ruido = torch.randn(batch_size, self.dim_latente, 1, 1, device=self.dispositivo)
                imgs_falsas = self.generador(ruido)
                fake_logits = self.discriminador(imgs_falsas.detach())
                loss_d_fake = F.relu(1.0 + fake_logits).mean()

                loss_D = loss_d_real + loss_d_fake + (0.2 * loss_rec)
                loss_D.backward()
                self.opt_dis.step()

                # === 2. Entrenamiento Generador (Hinge Loss) ===
                pasos_G = 2
                loss_G_acum = 0.0
                for _ in range(pasos_G):
                    self.opt_gen.zero_grad()
                    ruido_G = torch.randn(batch_size, self.dim_latente, 1, 1, device=self.dispositivo)
                    falsas_G = self.generador(ruido_G)
                    g_logits = self.discriminador(falsas_G)
                    loss_G = -g_logits.mean()
                    loss_G.backward()
                    self.opt_gen.step()
                    loss_G_acum += loss_G.item()

                total_loss_D += loss_D.item()
                total_loss_G += (loss_G_acum / pasos_G)
                batches += 1

            loss_D_avg = total_loss_D / max(1, batches)
            loss_G_avg = total_loss_G / max(1, batches)

            return loss_G_avg, loss_D_avg

        except Exception as e:
            print("[FastGAN] Error en entrenamiento:", e)
            traceback.print_exc()
            return 0, 0

    def generar_lote_sintetico(self, cantidad, ruta_destino, clase_nombre=None, batch_size=32, callback_progreso=None):
        if clase_nombre:
            ruta_destino = os.path.join(ruta_destino, clase_nombre)

        os.makedirs(ruta_destino, exist_ok=True)
        self.generador.eval()
        rutas_guardadas = []

        generadas = 0
        with torch.no_grad():
            while generadas < cantidad:
                tamano_batch_actual = min(batch_size, cantidad - generadas)
                z_ruido = torch.randn(tamano_batch_actual, self.dim_latente, 1, 1, device=self.dispositivo)
                imgs_falsas = self.generador(z_ruido)

                for idx in range(tamano_batch_actual):
                    img_tensor = imgs_falsas[idx]
                    nombre_archivo = f"synthetic_{generadas + 1:03d}.png"
                    ruta_img = os.path.join(ruta_destino, nombre_archivo)

                    img_normalizada = (img_tensor.clamp(-1.0, 1.0) + 1.0) / 2.0
                    save_image(img_normalizada, ruta_img)

                    rutas_guardadas.append(ruta_img)
                    generadas += 1

                    if callback_progreso:
                        callback_progreso(generadas, cantidad)

        print(f"[FastGAN] Generadas {len(rutas_guardadas)} imágenes sintéticas únicas en: {ruta_destino}")
        return rutas_guardadas
