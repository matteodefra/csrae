# CSRAE
Implementation of a standard VAE using CS loss. The implemented package is based on the following [paper](https://arxiv.org/pdf/2101.02149.pdf).

## General idea

The package contains an implementation of the standard Variational Autoencoder and the modifier Cauchy-Schwartz Regularized Autoencoder.

The general idea is to substitute the classical KL divergence of the standard VAE with a more detailed Cauchy-Schwartz divergence. 

The latter allow us to compute in a closed form the divergence between Mixture of Gaussians, while the KL divergence can be only approximated between MoGs.

In this way it is possible to furnish as prior distribution a mixture of multivariate gaussians (assuming <img src="https://latex.codecogs.com/svg.image?\mu, \sigma \in \mathbb{R}_{> 1}" title="\mu, \sigma \in \mathbb{R}_{> 1}" />)

<img src="https://latex.codecogs.com/svg.image?p(z)&space;=&space;\frac{1}{K}&space;\sum_{k=1}^{K}&space;\mathcal{N}&space;(&space;z&space;|&space;\mu_{k,\phi},&space;\sigma_{k,\phi}^2&space;)" title="p(z) = \frac{1}{K} \sum_{k=1}^{K} \mathcal{N} ( z | \mu_{k,\phi}, \sigma_{k,\phi}^2 )" />


The model is the typical Standard VAE, what changes is the decoder and the sampling method.

We learn the prior distribution above through a decoder network giving the required means and variances of the MoGs distribution.

For the sampling instead, we modify the classical sampling from a normal distribution with a random sampling among the different Gaussians learnt in our mixture prior.


## Implementation

The code is implemented in Pytorch, extending the base nn.Module with our custom implementation. 

src                                     
├─ csrae.py # Contains the implementation of the CS autoencoder           
├─ experiment.py # Utility functions for training, validation, testing and sampling                 
├─ main.py # Start point for the experiment                     
├─ utils.py # Utility functions  
└─ vae.py # Standard VAE implementation for comparison  


## Execution 

Open the file main.py to inspect the parser. It is possible to specify different input arguments, like epochs, number of K, etc..