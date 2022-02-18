from __future__ import print_function
import torch
from torch import nn, normal
from torch.nn import functional as F
from torch.autograd import Variable
from torch import Tensor
from typing import List
from utils import *

# Simple Implementation
class StandardVAE(nn.Module):

    def __init__(self, input_shape = (3, 32, 32)):
        super().__init__()
        self.input_shape = input_shape
        img_in_channels = input_shape[0]

        if input_shape[1] == 32: 
            hid_dims = [32, 64, 128, 256]
            latent_dim = 64
            self.start_width =2
        
        elif input_shape[1] == 64:
            hid_dims = [32, 64, 128, 256, 512]
            latent_dim = 128
            self.start_width =2

        else :
            raise NotImplementedError("input_shape parameter only 32, 64")

        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_modules = []

        in_channels = img_in_channels
        
        for h_dim in hid_dims:
            self.encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder_modules = nn.Sequential(*self.encoder_modules)
        self.encode_mu = nn.Linear(hid_dims[-1]*self.start_width*self.start_width, latent_dim)
        self.encode_logvar = nn.Linear(hid_dims[-1]*self.start_width*self.start_width, latent_dim)


        # Decoder
        self.proj_decode = nn.Linear(latent_dim, hid_dims[-1]*self.start_width*self.start_width)

        self.decoder_modules = []
        hid_dims.reverse()

        for i in range(len(hid_dims)-1):
            self.decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hid_dims[i],
                                       hid_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hid_dims[i + 1]),
                    nn.ReLU())
            )

        self.decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hid_dims[-1],
                                               hid_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hid_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hid_dims[-1], out_channels= img_in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid()
            )
        )
        self.decoder_modules = nn.Sequential(*self.decoder_modules)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)


    def encoder(self, x:Tensor) -> List[Tensor]:
        x = self.encoder_modules(x)
        # Flattening the encoder output
        x = x.view(x.size(0), -1)
        mu = self.encode_mu(x)
        log_var = self.encode_logvar(x)
        return [mu, log_var]

    def decoder(self, z:Tensor) -> Tensor:
        z = self.proj_decode(z)
        z = z.view(z.size(0), -1, self.start_width, self.start_width)
        return self.decoder_modules(z)
        

    def reparameterize_trick(self, mu : Tensor, log_var : Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return std*eps + mu

    def generate(self, x:Tensor) -> Tensor:
        return self.forward(x)[0]

    def sample(self, samples_num : input, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> List[Tensor]:
        latent = torch.randn(samples_num, self.latent_dim).to(device)
        return self.decoder(latent)


    def forward(self, x : Tensor) -> List[Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize_trick(mu, log_var)
        output = self.decoder(z)
        return output, mu, log_var

    def loss_function(self, pred : Tensor, real_img : Tensor, mu : Tensor, log_var : Tensor):
        recons_loss = nn.BCELoss(size_average=False)(pred, real_img) / pred.size(0)
        kld_loss = ((mu ** 2 + log_var.exp() - 1 - log_var) / 2).mean()
        loss = recons_loss + kld_loss
        return loss, recons_loss, kld_loss