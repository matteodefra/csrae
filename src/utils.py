import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from scipy.stats import multivariate_normal


#=======================================================================================================================

#=======================================================================================================================
# WEIGHTS INITS
#=======================================================================================================================
def xavier_init(m):
    s =  np.sqrt( 2. / (m.in_features + m.out_features) )
    m.weight.data.normal_(0, s)

#=======================================================================================================================
def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

#=======================================================================================================================
def normal_init(m, mean=0., std=0.01):
    m.weight.data.normal_(mean, std)


#=======================================================================================================================

#=======================================================================================================================
# FUNCTIONS
#=======================================================================================================================
'''
    Compute the pdf of a multivariate normal at point x, with mean _mean_ and covariance _covariance_matrix_, and k dimensions according to the 
    following formula:

    N(x; \mu, \Sigma = diag(\sigma^2)) = \frac{1}{(2 \pi)^k * det(\Sigma)} \exp{-1/2 (x-\mu) \Sigma (x-\mu)^T} 
''' 
def multivariate_distr(x, mean, covariance_matrix):
    a = 1 / math.sqrt( (2 * math.pi) ** mean.shape[0] ) * torch.det(covariance_matrix)
    quad_form_first = torch.matmul ( torch.t(x-mean), torch.inverse(covariance_matrix) )
    quad_form = torch.matmul( quad_form_first , x - mean ) 
    func = a * torch.exp( -0.5 * quad_form )
    return func


def cdf_multivariate_distr(x, mean, covariance_matrix):
    distr = multivariate_normal(mean, covariance_matrix)
    return distr.cdf(x)



#=======================================================================================================================
def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )



def pairwise(it):
    it = iter(it)
    while True:
        try:
            yield next(it), next(it)
        except StopIteration:
            # no more elements in the iterator
            return