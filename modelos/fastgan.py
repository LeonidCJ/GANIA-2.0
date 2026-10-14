import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import traceback
from torchvision.utils import save_image


# =============================================================================
# DiffAugment (Zhao et al., NeurIPS 2020) - usado por FastGAN para few-shot.
# Se aplica a imágenes REALES y FALSAS antes del discriminador, de forma
# diferenciable, para que D no memorice el dataset y los aumentos no se
# "filtren" a las imágenes generadas. Asume tensores en rango [-1, 1].
# =============================================================================
def _rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def _rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean


def _rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean


def _rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)


def _rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


_AUGMENT_FNS = {
    'color': [_rand_brightness, _rand_saturation, _rand_contrast],
    'translation': [_rand_translation],
    'cutout': [_rand_cutout],
}


def DiffAugment(x, policy='color,translation,cutout'):
    if policy:
        for p in policy.split(','):
            for f in _AUGMENT_FNS[p]:
                x = f(x)
        x = x.contiguous()
    return x


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
    Entrenador de FastGAN con Hinge Loss + Normalización Espectral + Pérdida de Reconstrucción (0.2)
    + DiffAugment (few-shot) + Generador EMA (promedio móvil de pesos para inferencia).
    """
    def __init__(self, dispositivo, dim_latente=100, canales_img=3, lr=0.0002,
                 politica_aug='color,translation,cutout', ema_beta=0.999):
        self.dispositivo = dispositivo
        self.dim_latente = dim_latente
        self.politica_aug = politica_aug
        self.ema_beta = ema_beta
        self.pasos_totales = 0

        self.generador = GeneradorFastGAN(dim_latente, canales_img).to(dispositivo)
        self.discriminador = DiscriminadorFastGAN(canales_img).to(dispositivo)

        # Copia EMA del generador: es la que se usa para muestrear/generar (más nítida y estable).
        self.generador_ema = copy.deepcopy(self.generador).eval()
        for p in self.generador_ema.parameters():
            p.requires_grad_(False)

        self.opt_gen = torch.optim.Adam(self.generador.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_dis = torch.optim.Adam(self.discriminador.parameters(), lr=lr, betas=(0.5, 0.999))

        self.ruido_fijo = torch.randn(64, dim_latente, 1, 1, device=dispositivo)

    @torch.no_grad()
    def _actualizar_ema(self):
        # Warm-up: al inicio el EMA sigue de cerca a G, luego se estabiliza en ema_beta.
        beta = min(self.ema_beta, (1.0 + self.pasos_totales) / (10.0 + self.pasos_totales))
        for p_ema, p in zip(self.generador_ema.parameters(), self.generador.parameters()):
            p_ema.lerp_(p, 1.0 - beta)
        for b_ema, b in zip(self.generador_ema.buffers(), self.generador.buffers()):
            b_ema.copy_(b)

    @torch.no_grad()
    def generar(self, z):
        """Genera imágenes con el generador EMA en modo evaluación. Devuelve tensores en [-1, 1]."""
        self.generador_ema.eval()
        return self.generador_ema(z.to(self.dispositivo))

    def guardar_checkpoint(self, epoca, ruta_dir="./datos/resultados/checkpoints_fastgan"):
        os.makedirs(ruta_dir, exist_ok=True)
        ruta = os.path.join(ruta_dir, f"fastgan_epoca_{epoca}.pth")
        torch.save({
            'epoca': epoca,
            'pasos_totales': self.pasos_totales,
            'gen_state_dict': self.generador.state_dict(),
            'gen_ema_state_dict': self.generador_ema.state_dict(),
            'dis_state_dict': self.discriminador.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'opt_dis_state_dict': self.opt_dis.state_dict(),
        }, ruta)
        print(f"[FastGAN] Checkpoint guardado en: {ruta}")
        return ruta

    def cargar_checkpoint(self, ruta):
        checkpoint = torch.load(ruta, map_location=self.dispositivo)
        self.generador.load_state_dict(checkpoint['gen_state_dict'])
        # Compatibilidad con checkpoints antiguos sin EMA
        self.generador_ema.load_state_dict(checkpoint.get('gen_ema_state_dict', checkpoint['gen_state_dict']))
        self.discriminador.load_state_dict(checkpoint['dis_state_dict'])
        self.opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        self.opt_dis.load_state_dict(checkpoint['opt_dis_state_dict'])
        self.pasos_totales = checkpoint.get('pasos_totales', 0)
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
                # DiffAugment se aplica a reales y falsas: D nunca ve la imagen "cruda",
                # lo que evita que memorice las pocas imágenes del dataset.
                self.opt_dis.zero_grad()

                reales_aug = DiffAugment(imgs_reales, self.politica_aug)
                real_logits, rec_img, real_patch = self.discriminador(reales_aug, rec=True)
                loss_d_real = F.relu(1.0 - real_logits).mean()
                loss_rec = F.mse_loss(rec_img, real_patch)

                ruido = torch.randn(batch_size, self.dim_latente, 1, 1, device=self.dispositivo)
                imgs_falsas = self.generador(ruido)
                fake_logits = self.discriminador(DiffAugment(imgs_falsas.detach(), self.politica_aug))
                loss_d_fake = F.relu(1.0 + fake_logits).mean()

                loss_D = loss_d_real + loss_d_fake + (0.2 * loss_rec)
                loss_D.backward()
                self.opt_dis.step()

                # === 2. Entrenamiento Generador (Hinge Loss), 1 paso por paso de D ===
                self.opt_gen.zero_grad()
                g_logits = self.discriminador(DiffAugment(imgs_falsas, self.politica_aug))
                loss_G = -g_logits.mean()
                loss_G.backward()
                self.opt_gen.step()

                self.pasos_totales += 1
                self._actualizar_ema()

                total_loss_D += loss_D.item()
                total_loss_G += loss_G.item()
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
        rutas_guardadas = []

        generadas = 0
        with torch.no_grad():
            while generadas < cantidad:
                tamano_batch_actual = min(batch_size, cantidad - generadas)
                z_ruido = torch.randn(tamano_batch_actual, self.dim_latente, 1, 1, device=self.dispositivo)
                imgs_falsas = self.generar(z_ruido)

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
