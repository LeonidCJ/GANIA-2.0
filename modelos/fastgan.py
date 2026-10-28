import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import traceback
from torchvision.utils import save_image


# =============================================================================
# DiffAugment Calibrado para 64x64 / 128x128 / 256x256 Few-Shot (CPU/GPU)
# =============================================================================
def _rand_brightness(x):
    # Variación suave de brillo (+-0.15) para no quemar detalles claros ni fondos oscuros
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * 0.30


def _rand_saturation(x):
    # Variación moderada de saturación (0.75x a 1.25x)
    x_mean = x.mean(dim=1, keepdim=True)
    factor = 0.75 + torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 0.50
    return (x - x_mean) * factor + x_mean


def _rand_contrast(x):
    # Variación moderada de contraste (0.80x a 1.20x)
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = 0.80 + torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 0.40
    return (x - x_mean) * factor + x_mean


def _rand_translation(x, ratio=0.0625):
    B, C, H, W = x.shape
    shift_x = max(1, int(H * ratio + 0.5))
    shift_y = max(1, int(W * ratio + 0.5))

    if x.device.type == 'cpu':
        x_pad = F.pad(x, [shift_y, shift_y, shift_x, shift_x], mode='reflect')
        tx = torch.randint(0, 2 * shift_x + 1, (B,))
        ty = torch.randint(0, 2 * shift_y + 1, (B,))
        return torch.stack([
            x_pad[b, :, tx[b]:tx[b] + H, ty[b]:ty[b] + W] for b in range(B)
        ], dim=0)
    else:
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[B, 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[B, 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(B, dtype=torch.long, device=x.device),
            torch.arange(H, dtype=torch.long, device=x.device),
            torch.arange(W, dtype=torch.long, device=x.device),
            indexing='ij'
        )
        grid_x = torch.clamp(grid_x + translation_x, 0, H - 1)
        grid_y = torch.clamp(grid_y + translation_y, 0, W - 1)
        return x.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)


def _rand_cutout(x, ratio=0.25):
    # Parche del 25% del lado aplicado con probabilidad 50% por imagen
    B, C, H, W = x.shape
    cut_h, cut_w = int(H * ratio + 0.5), int(W * ratio + 0.5)
    aplicar = torch.rand(B, device=x.device) < 0.5
    if not aplicar.any():
        return x

    mask = torch.ones((B, 1, H, W), dtype=x.dtype, device=x.device)
    cy = torch.randint(0, H, (B,), device=x.device)
    cx = torch.randint(0, W, (B,), device=x.device)
    for b in range(B):
        if aplicar[b]:
            y1 = max(0, int(cy[b]) - cut_h // 2)
            y2 = min(H, int(cy[b]) + cut_h // 2)
            x1 = max(0, int(cx[b]) - cut_w // 2)
            x2 = min(W, int(cx[b]) + cut_w // 2)
            mask[b, :, y1:y2, x1:x2] = 0
    return x * mask


_AUGMENT_FNS = {
    'color': [_rand_brightness, _rand_saturation, _rand_contrast],
    'translation': [_rand_translation],
    'cutout': [_rand_cutout],
}


def DiffAugment(x, policy='color,translation,cutout'):
    if policy:
        for p in policy.split(','):
            p = p.strip()
            if p in _AUGMENT_FNS:
                for f in _AUGMENT_FNS[p]:
                    x = f(x)
    return x


# =============================================================================
# Arquitectura FastGAN Original (Nearest + SLE Estándar)
# =============================================================================
class SLE(nn.Module):
    """
    Skip-Layer Excitation Module (FastGAN ICLR 2021).
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
        return x_high * self.map(x_low)


class GeneradorFastGAN(nn.Module):
    """
    Generador FastGAN multirresolución original (Nearest neighbor upsampling).
    """
    def __init__(self, dim_latente=100, canales_img=3, tamano_img=64):
        super().__init__()
        self.dim_latente = dim_latente
        self.tamano_img = int(tamano_img)

        self.init = nn.Sequential(
            nn.ConvTranspose2d(dim_latente, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.feat_8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.feat_16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.feat_32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.feat_64 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        # SLE base (4x4 -> 32x32 y 8x8 -> 64x64)
        self.sle_32 = SLE(in_channels_low=512, in_channels_high=64, out_channels_high=64)
        self.sle_64 = SLE(in_channels_low=256, in_channels_high=32, out_channels_high=32)

        canales_final = 32

        if self.tamano_img >= 128:
            self.feat_128 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(32, 16, 3, 1, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            )
            self.sle_128 = SLE(in_channels_low=128, in_channels_high=16, out_channels_high=16)
            canales_final = 16

        if self.tamano_img >= 256:
            self.feat_256 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            )
            self.sle_256 = SLE(in_channels_low=64, in_channels_high=16, out_channels_high=16)
            canales_final = 16

        self.to_rgb = nn.Sequential(
            nn.Conv2d(canales_final, canales_img, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        out_4 = self.init(z)
        out_8 = self.feat_8(out_4)
        out_16 = self.feat_16(out_8)
        out_32 = self.sle_32(out_4, self.feat_32(out_16))
        out_64 = self.sle_64(out_8, self.feat_64(out_32))
        out = out_64

        if self.tamano_img >= 128:
            out_128 = self.sle_128(out_16, self.feat_128(out_64))
            out = out_128
        if self.tamano_img >= 256:
            out_256 = self.sle_256(out_32, self.feat_256(out))
            out = out_256

        return self.to_rgb(out)


class DecodificadorD(nn.Module):
    """
    Decodificador Auxiliar original del Discriminador:
    Reconstruye a 32x32 para la pérdida de consistencia espacial con el patch real.
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
    Discriminador FastGAN multirresolución con Normalización Espectral.
    """
    def __init__(self, canales_img=3, tamano_img=64):
        super().__init__()
        self.tamano_img = int(tamano_img)

        canales_entrada_64 = canales_img
        if self.tamano_img == 128:
            self.down_pre = nn.Sequential(
                spectral_norm(nn.Conv2d(canales_img, 16, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True)
            )
            canales_entrada_64 = 16
        elif self.tamano_img >= 256:
            self.down_pre = nn.Sequential(
                spectral_norm(nn.Conv2d(canales_img, 16, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(16, 16, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=True)
            )
            canales_entrada_64 = 16

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(canales_entrada_64, 32, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(32) if self.tamano_img >= 128 else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(32, 64, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.logits = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 1, 4, 1, 0, bias=False))
        )

        self.decoder = DecodificadorD(in_channels=128, out_channels=canales_img)

    def forward(self, img, rec=False, img_objetivo=None):
        x = self.down_pre(img) if hasattr(self, 'down_pre') else img
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)

        out_logit = self.logits(f4).view(img.size(0), -1).mean(1)

        if rec:
            rec_img = self.decoder(f3)
            fuente = img_objetivo if img_objetivo is not None else img.clamp(-1.0, 1.0)
            factor = max(1, fuente.size(2) // 32)
            img_patch = F.avg_pool2d(fuente, kernel_size=factor, stride=factor)
            return out_logit, rec_img, img_patch

        return out_logit


def _adaptar_state_dict_generador(state_dict):
    """
    Limpia checkpoints con capas de ruido residuales si existieran.
    """
    if not any('.2.peso' in k for k in state_dict.keys()):
        return state_dict
    nuevo = {}
    for k, v in state_dict.items():
        if '.2.peso' in k:
            continue
        if any(k.startswith(p + '.3.') for p in ('feat_8', 'feat_16', 'feat_32', 'feat_64')):
            k = k.replace('.3.', '.2.')
        nuevo[k] = v
    return nuevo


# =============================================================================
# Clase Entrenador
# =============================================================================
class EntrenadorFastGAN:
    """
    Entrenador de FastGAN optimizado para CPU y GPU con soporte para 64x64, 128x128 y 256x256.
    """
    def __init__(self, dispositivo, dim_latente=100, canales_img=3, lr=0.0002,
                 tamano_img=64, politica_aug='color,translation,cutout', ema_beta=None):
        self.dispositivo = dispositivo
        self.dim_latente = dim_latente
        self.tamano_img = int(tamano_img)
        self.politica_aug = politica_aug
        if ema_beta is None:
            self.ema_beta = 0.99 if str(dispositivo) == 'cpu' else 0.995
        else:
            self.ema_beta = ema_beta
        self.pasos_totales = 0
        self._ema_bn_calibrado = False

        if str(dispositivo) == 'cpu':
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(max(1, num_cores - 1))

        self.generador = GeneradorFastGAN(dim_latente, canales_img, tamano_img=self.tamano_img).to(dispositivo)
        self.discriminador = DiscriminadorFastGAN(canales_img, tamano_img=self.tamano_img).to(dispositivo)

        self.generador_ema = copy.deepcopy(self.generador).eval()
        for p in self.generador_ema.parameters():
            p.requires_grad_(False)

        self.opt_gen = torch.optim.Adam(self.generador.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_dis = torch.optim.Adam(self.discriminador.parameters(), lr=lr, betas=(0.5, 0.999))

        self.ruido_fijo = torch.randn(64, dim_latente, 1, 1, device=dispositivo).clamp(-2.0, 2.0) * 0.90

    @torch.no_grad()
    def _actualizar_ema(self):
        beta = min(self.ema_beta, (1.0 + self.pasos_totales) / (10.0 + self.pasos_totales))
        for p_ema, p in zip(self.generador_ema.parameters(), self.generador.parameters()):
            p_ema.lerp_(p, 1.0 - beta)
        self._ema_bn_calibrado = False

    @torch.no_grad()
    def _calibrar_bn_ema(self, truncation_psi=0.90):
        self.generador_ema.train()
        for _ in range(10):
            z_cal = torch.randn(16, self.dim_latente, 1, 1, device=self.dispositivo)
            if truncation_psi is not None and truncation_psi < 1.0:
                z_cal = z_cal.clamp(-2.0, 2.0) * truncation_psi
            self.generador_ema(z_cal)
        self.generador_ema.eval()
        self._ema_bn_calibrado = True

    @torch.no_grad()
    def generar(self, z, truncation_psi=0.90):
        if not self._ema_bn_calibrado:
            self._calibrar_bn_ema(truncation_psi=truncation_psi)
        self.generador_ema.eval()
        z_in = z.to(self.dispositivo)
        if truncation_psi is not None and truncation_psi < 1.0:
            z_in = z_in.clamp(-2.0, 2.0) * truncation_psi
        return self.generador_ema(z_in)

    def guardar_checkpoint(self, epoca, ruta_dir="./datos/resultados/checkpoints_fastgan"):
        if not self._ema_bn_calibrado:
            self._calibrar_bn_ema()
        os.makedirs(ruta_dir, exist_ok=True)
        ruta = os.path.join(ruta_dir, f"fastgan_epoca_{epoca}.pth")
        torch.save({
            'epoca': epoca,
            'tamano_img': self.tamano_img,
            'pasos_totales': self.pasos_totales,
            'gen_state_dict': self.generador.state_dict(),
            'gen_ema_state_dict': self.generador_ema.state_dict(),
            'dis_state_dict': self.discriminador.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'opt_dis_state_dict': self.opt_dis.state_dict(),
        }, ruta)
        print(f"[FastGAN] Checkpoint guardado en: {ruta}")

        try:
            ckpts = [
                os.path.join(ruta_dir, f) for f in os.listdir(ruta_dir)
                if f.startswith("fastgan_epoca_") and f.endswith(".pth")
            ]
            ckpts.sort(key=lambda p: os.path.getmtime(p))
            for viejo in ckpts[:-5]:
                os.remove(viejo)
        except Exception:
            pass
        return ruta

    def cargar_checkpoint(self, ruta):
        checkpoint = torch.load(ruta, map_location=self.dispositivo, weights_only=False)
        gen_sd = _adaptar_state_dict_generador(checkpoint['gen_state_dict'])
        ema_sd = _adaptar_state_dict_generador(checkpoint.get('gen_ema_state_dict', checkpoint['gen_state_dict']))
        try:
            self.generador.load_state_dict(gen_sd)
            self.generador_ema.load_state_dict(ema_sd)
            self.discriminador.load_state_dict(checkpoint['dis_state_dict'])
            self.opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
            self.opt_dis.load_state_dict(checkpoint['opt_dis_state_dict'])
        except (RuntimeError, ValueError):
            self.generador.load_state_dict(gen_sd, strict=False)
            self.generador_ema.load_state_dict(ema_sd, strict=False)
            self.discriminador.load_state_dict(checkpoint['dis_state_dict'], strict=False)
        self.pasos_totales = checkpoint.get('pasos_totales', 0)
        self._calibrar_bn_ema()
        print(f"[FastGAN] Checkpoint cargado desde: {ruta}")
        return checkpoint['epoca']

    def entrenar_epoca(self, cargador_datos, epoca_actual, total_epocas, abort_callback=None):
        self.generador.train()
        self.discriminador.train()

        total_loss_D = 0.0
        total_loss_G = 0.0
        batches = 0

        for i, (imgs_reales, _) in enumerate(cargador_datos):
            if abort_callback and abort_callback():
                return 0, 0

            imgs_reales = imgs_reales.to(self.dispositivo)
            batch_size = imgs_reales.size(0)

            # === 1. Entrenamiento Discriminador (Hinge Loss + Self-Supervised Rec Loss 0.2) ===
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

    def generar_lote_sintetico(self, cantidad=500, ruta_destino="./datos/datasets/augmented",
                               clase_nombre="clase_01", batch_size=32, callback_progreso=None):
        """
        Genera N imágenes sintéticas de alta fidelidad con el generador EMA y las guarda en disco.
        """
        if clase_nombre:
            ruta_destino = os.path.join(ruta_destino, clase_nombre)

        os.makedirs(ruta_destino, exist_ok=True)
        rutas_guardadas = []

        generadas = 0
        with torch.no_grad():
            while generadas < cantidad:
                tamano_batch_actual = min(batch_size, cantidad - generadas)
                z_ruido = torch.randn(tamano_batch_actual, self.dim_latente, 1, 1, device=self.dispositivo)
                imgs_falsas = self.generar(z_ruido, truncation_psi=0.90)

                for idx in range(tamano_batch_actual):
                    img_tensor = imgs_falsas[idx]
                    idx_global = generadas + idx + 1
                    nombre_archivo = f"sintetica_{clase_nombre}_{idx_global:05d}.png"
                    ruta_img = os.path.join(ruta_destino, nombre_archivo)
                    save_image(img_tensor, ruta_img, normalize=True, value_range=(-1, 1))
                    rutas_guardadas.append(ruta_img)

                    if callback_progreso:
                        callback_progreso(idx_global, cantidad)

                generadas += tamano_batch_actual

        print(f"[FastGAN] Generadas {len(rutas_guardadas)} imágenes sintéticas únicas en: {ruta_destino}")
        return rutas_guardadas

    # Alias para compatibilidad
    generar_imagenes_aumento = generar_lote_sintetico
