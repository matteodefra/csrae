from __future__ import print_function
import math
import torch
from torch import nn
from torch import Tensor
from typing import List
from utils import *
import random

import torch.distributions as D

# Simple Implementation
class CSRAE(nn.Module):

    def __init__(self, input_shape = (3, 32, 32), K = 50, lambd = 0.01):
        super().__init__()
        self.input_shape = input_shape
        self.K = K
        self.lambd = lambd
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

        # Init the prior distribution randomly, use an encoding neural network to train them
        self.prior_means = []
        self.prior_std = []
        for i in range(K):
            self.prior_means.append(torch.randn(self.latent_dim,))
            self.prior_std.append(torch.randn(self.latent_dim,))

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

    def sample(self, samples_num : input, device) -> List[Tensor]:
        # Sample from the mixture of gaussians -> extract random K
        gaussian = random.randint(0, self.K-1)
        distribution = D.MultivariateNormal(self.prior_means[gaussian], torch.diag( (self.prior_std[gaussian])**2 ) )# + torch.diag( torch.div(torch.ones(self.latent_dim), 1e2 )) )
        latent = distribution.sample(samples_num).to(device)
        # latent = torch.randn(samples_num, self.latent_dim).to(device)
        return self.decoder(latent)

    
    def forward(self, x : Tensor) -> List[Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize_trick(mu, log_var)
        output = self.decoder(z)

        return output, mu, log_var


    '''
        mu: tensor of dimension (batch_size, latent_dimension) -> each row is the mu/log_var of a sample from training
    '''
    def loss_function(self, pred : Tensor, real_img : Tensor, mu : Tensor, log_var : Tensor):

        recons_loss = nn.BCELoss(size_average=False)(pred, real_img) / pred.size(0)

        batch = log_var.shape[0]

        cs_loss = torch.tensor(0.0)

        first_term = .0
        mean_sum = .0
        var_sum = .0

        for i in range(batch):
            for j in range(self.K):
                mean_sum += self.prior_means[j]
                var_sum += log_var[i,:].exp() + self.prior_std[j].exp()

        for i in range(batch):
            first_term += ((mu[i,:] - mean_sum).T * torch.inverse(torch.diag(var_sum)) * (mu[i,:] - mean_sum) )

        first_term = torch.mean( torch.Tensor( first_term ) )

        first_term = self.lambd * first_term

        second_term = .0
        mean_sum = .0
        var_sum = .0

        for i in range(batch):
            for j in range(self.K):
                mean_sum += self.prior_means[j]
                var_sum += (2 * self.prior_std[j].exp())

        for i in range(self.K):
            second_term += ((self.prior_means[i] - mean_sum).T * torch.inverse(torch.diag(var_sum)) * (self.prior_means[i] - mean_sum) )

        second_term = second_term.mean()

        second_term = -self.lambd * second_term

        '''
            Approximation of the first term: first factor is computed as 

               - \lambda * \sum_k \mathcal{N}( \mu_\phi | \mu_k, diag(\sigma^2_\phi + \sigma^2_k ))

            hence the summation over all the multinomial distribution parametrized by \mu_k, the prior means and the \sigma_\phi and \sigma_k, 
            respectively the std returned by the encoder and the prior std.

            We iterate over the different K components of the prior distribution and we accumulate the sum, storing the result for each 
            image in the batch.
            Finally we compute the sum over them and we get the first term
        '''
        # for i in range(batch):

        #     summation = 0

        #     for j in range(self.K):

        #         norm = D.Normal( self.prior_means[j], log_var[i,:].exp() + self.prior_std[j].exp() )

        #         val = torch.mean( norm.cdf( mu[i,:] ) ) 

        #         summation += val

        #     first_term.append(summation)

        # first_term = torch.mean( torch.Tensor( first_term ) )

        # first_term = self.lambd * torch.log( first_term )

        # second_term = torch.tensor(0.0)

        # '''
        #     Approximation of the second term: second factor is computed as 

        #        - \lambda * \sum_{k,k'} \mathcal{N}( \mu_k | \mu_k', diag(\sigma^2_{k'} ))

        #     Here only the prior is considered, by having a nested summation of the contribution of the different gaussians in the mixture
        # '''
        # for i in range(self.K):

        #     for j in range(i):

        #         norm = D.Normal( self.prior_means[j], 2 * self.prior_std[j].exp() )

        #         val = torch.mean( norm.cdf( self.prior_means[i] ) )

        #         second_term += val

        # second_term = second_term.mean()

        # second_term = - self.lambd * torch.log( second_term )

        # Third term
        third_term = torch.tensor( - self.lambd * math.log( self.K ) )

        # Fourth term
        fourth_term = self.lambd * self.start_width * torch.log( 2 * torch.norm( log_var ) * torch.sqrt( torch.tensor(math.pi) ) )

        cs_loss = first_term + second_term + third_term + fourth_term

        loss = recons_loss + cs_loss

        print(cs_loss)

        return loss, recons_loss, cs_loss