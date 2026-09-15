import torch
import torch.nn as nn
import traceback

class Generador(nn.Module):
    def __init__(self, dim_latente=100, canales_img=3):
        
        super().__init__()
        
        self.modelo = nn.Sequential(
            
            # Entrada: [B, 100, 1, 1]
            nn.ConvTranspose2d(dim_latente, 512, 4, 1, 0, bias=False),  # [B, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, canales_img, 4, 2, 1, bias=False),  # [B, 3, 64, 64]
            nn.Tanh()
        )

    def forward(self, z):
        return self.modelo(z)
    
class Discriminador(nn.Module):
    def __init__(self, canales_img=3):
        super().__init__()
        self.modelo = nn.Sequential(
            # Entrada: [B, 3, 64, 64]
            nn.Conv2d(canales_img, 64, 4, 2, 1, bias=False),  # [B, 64, 32, 32]
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # [B, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # [B, 1, 1, 1]
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.modelo(img)
        return out.view(out.size(0), -1).mean(1) # Salida escalar por imagen

class EntrenadorDCGAN:
    def __init__(self, dispositivo, dim_latente=100, canales_img=3, lr=0.0002):
        self.dispositivo = dispositivo
        self.generador = Generador(dim_latente, canales_img).to(dispositivo)
        self.discriminador = Discriminador(canales_img).to(dispositivo)
        
        self.dim_latente = dim_latente
        self.opt_gen = torch.optim.Adam(self.generador.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_dis = torch.optim.Adam(self.discriminador.parameters(), lr=lr, betas=(0.5, 0.999))
        
        self.criterio = nn.BCELoss()
        self.ruido_fijo = torch.randn(64, dim_latente, 1, 1, device=dispositivo)

    def guardar_checkpoint(self, epoca, ruta_dir="resultados/checkpoints"):
        import os
        os.makedirs(ruta_dir, exist_ok=True)
        ruta = os.path.join(ruta_dir, f"dcgan_epoca_{epoca}.pth")
        torch.save({
            'epoca': epoca,
            'gen_state_dict': self.generador.state_dict(),
            'dis_state_dict': self.discriminador.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'opt_dis_state_dict': self.opt_dis.state_dict(),
        }, ruta)
        print(f"Checkpoint guardado en: {ruta}")

    def cargar_checkpoint(self, ruta):
        checkpoint = torch.load(ruta, map_location=self.dispositivo)
        self.generador.load_state_dict(checkpoint['gen_state_dict'])
        self.discriminador.load_state_dict(checkpoint['dis_state_dict'])
        self.opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        self.opt_dis.load_state_dict(checkpoint['opt_dis_state_dict'])
        return checkpoint['epoca']

    def entrenar_epoca(self, cargador_datos, epoca, num_epocas, abort_callback=None):
        try:
            self.generador.train()
            self.discriminador.train()

            total_loss_G = 0
            total_loss_D = 0
            batches = 0

            for i, (imgs_reales, _) in enumerate(cargador_datos):
                if abort_callback and abort_callback():
                    return 0, 0

                imgs_reales = imgs_reales.to(self.dispositivo)
                batch_size = imgs_reales.size(0)

                # Label Smoothing: Usar 0.9 para reales en lugar de 1.0
                reales = torch.full((batch_size,), 0.9, device=self.dispositivo)
                falsas = torch.zeros(batch_size, device=self.dispositivo)

                # === Entrenamiento del discriminador ===
                self.opt_dis.zero_grad()

                salida_real = self.discriminador(imgs_reales)
                perdida_real = self.criterio(salida_real, reales)

                ruido = torch.randn(batch_size, self.dim_latente, 1, 1, device=self.dispositivo)
                imgs_falsas = self.generador(ruido)

                salida_falsa = self.discriminador(imgs_falsas.detach())
                perdida_falsa = self.criterio(salida_falsa, falsas)

                perdida_dis = perdida_real + perdida_falsa
                perdida_dis.backward()
                self.opt_dis.step()

                # === Entrenamiento del generador ===
                self.opt_gen.zero_grad()
                # Para el generador, queremos que el discriminador piense que son reales (1.0)
                reales_gen = torch.ones(batch_size, device=self.dispositivo)
                salida = self.discriminador(imgs_falsas)
                perdida_gen = self.criterio(salida, reales_gen)
                perdida_gen.backward()
                self.opt_gen.step()

                total_loss_D += perdida_dis.item()
                total_loss_G += perdida_gen.item()
                batches += 1

            loss_G_avg = total_loss_G / batches
            loss_D_avg = total_loss_D / batches

            return loss_G_avg, loss_D_avg

        except Exception as e:
            print("Error durante entrenamiento de DCGAN:", e)
            traceback.print_exc()
            return 0, 0
