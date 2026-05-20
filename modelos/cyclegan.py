import torch
import torch.nn as nn
import itertools
from modelos.generador_resnet import GeneradorResNet

class DiscriminadorCycle(nn.Module):
    def __init__(self, canales_img=3):
        super().__init__()
        def bloque_d(in_c, out_c, norm=True):
            capas = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
            if norm: capas.append(nn.InstanceNorm2d(out_c))
            capas.append(nn.LeakyReLU(0.2, inplace=True))
            return capas

        self.modelo = nn.Sequential(
            *bloque_d(canales_img, 64, norm=False),
            *bloque_d(64, 128),
            *bloque_d(128, 256),
            *bloque_d(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.modelo(x)

class EntrenadorCycleGAN:
    def __init__(self, dispositivo, canales_img=3, lr=0.0002):
        self.dispositivo = dispositivo
        self.G_AB = GeneradorResNet(canales_img, canales_img).to(dispositivo)
        self.G_BA = GeneradorResNet(canales_img, canales_img).to(dispositivo)
        self.D_A = DiscriminadorCycle(canales_img).to(dispositivo)
        self.D_B = DiscriminadorCycle(canales_img).to(dispositivo)
        
        # Optimizador que agrupa ambos generadores
        self.opt_gen = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=lr, betas=(0.5, 0.999)
        )
        self.opt_dis = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=lr, betas=(0.5, 0.999)
        )

        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        self.G_AB.apply(self._init_pesos)
        self.G_BA.apply(self._init_pesos)
        self.D_A.apply(self._init_pesos)
        self.D_B.apply(self._init_pesos)

    def _init_pesos(self, m):
        
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def transformar(self, imagen_tensor):
        imagen_tensor = imagen_tensor.to(self.dispositivo)
        if imagen_tensor.ndim == 3:
            imagen_tensor = imagen_tensor.unsqueeze(0)
        with torch.no_grad():
            salida = self.G_AB(imagen_tensor)
        return salida

    def entrenar_epoca(self, cargador_a, cargador_b, epoca, max_epocas):
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()

        total_loss_G, total_loss_D = 0.0, 0.0
        
        for i, ((real_A, _), (real_B, _)) in enumerate(zip(cargador_a, cargador_b)):
            real_A, real_B = real_A.to(self.dispositivo), real_B.to(self.dispositivo)
            batch_size = real_A.size(0)
            
            # Etiquetas (PatchGAN output size suele ser 3x3 o 4x4 para inputs de 64x64)
            valid = torch.ones((batch_size, 1, 3, 3), device=self.dispositivo, requires_grad=False)
            fake = torch.zeros((batch_size, 1, 3, 3), device=self.dispositivo, requires_grad=False)

            # --- Entrenamiento Generadores ---
            self.opt_gen.zero_grad()

            # Identidad
            loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
            loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = self.G_AB(real_A)
            loss_GAN_AB = self.criterion_gan(self.D_B(fake_B), valid)
            fake_A = self.G_BA(real_B)
            loss_GAN_BA = self.criterion_gan(self.D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = self.G_BA(fake_B)
            loss_cycle_A = self.criterion_cycle(recov_A, real_A)
            recov_B = self.G_AB(fake_A)
            loss_cycle_B = self.criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total G
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity
            loss_G.backward()
            self.opt_gen.step()

            # --- Entrenamiento Discriminadores ---
            self.opt_dis.zero_grad()

            # D_A
            loss_real_A = self.criterion_gan(self.D_A(real_A), valid)
            loss_fake_A = self.criterion_gan(self.D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2

            # D_B
            loss_real_B = self.criterion_gan(self.D_B(real_B), valid)
            loss_fake_B = self.criterion_gan(self.D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real_B + loss_fake_B) / 2

            loss_D = (loss_D_A + loss_D_B) / 2
            loss_D.backward()
            self.opt_dis.step()

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()

        n_batches = min(len(cargador_a), len(cargador_b))
        return total_loss_G / n_batches, total_loss_D / n_batches
