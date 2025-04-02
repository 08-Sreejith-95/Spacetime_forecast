import math

import torch
import torch.nn.functional as F

from einops import rearrange, reduce

#Note: the  '...' indexing is actually including all dimension preceding the tensor. eg:- if the tensor is 5D. say A , shape(2,3,4,5,6) :- A[..., 1:, :-1]
#slice A to a tensor of shape A[:, :, :, 1:, :-1] -> A shape(2,3,4,4,5)
# p here is the weight vector [a0, a1, a2,.....,a_d-1]., and d is the dimension of the latent space X.
#doubt:- so should we initialize the p vector randomly?. In the paper, for the sample U_0 there is no history in the initial state so the vector p is initialized as a null vector.
#check the config:   model_dim: 128, n_kernels: 128, kernel_dim: 64. This might indicate d = 64 and 128 such kernals are used so batch size might be 128
#To_do: take a sample series and check how the function below maps


#This is the definition of companion matrix
def companion_from_p(p):
    """
    Arguments:
        p: (..., d)
    Return:
        A: (..., d, d)
    """
    batch_size, d = p.shape[:-1], p.shape[-1]
    A = torch.zeros(*batch_size, d, d, dtype=p.dtype, device=p.device)#initializing A matrix dxd with zeros
    A[..., 1:, :-1] = torch.eye(d - 1, dtype=p.dtype, device=p.device)#replacing all the elements from (1,1) to (d,d-1) with identity matrix
    A[..., -1] = p #replacing the final column with our vector p. 
    return A 
def get_alpha_matrix(self, p):#todo.- change this function to get_alpha
    A = self.row_shift_matrix.to(p.device)        
    for kl in range(self.n_kernels):
        for i in range(self.k):
            A[kl, 2*self.n + i, i] = 1 - p[kl, i]
            A[kl, 2*self.n + i, i +1] = p[kl, i]
        return A

#To_do: if i want to plugin custom A matrices:- Approach:Introduce a conditional statement to run The krylov by custom matrices if companion is NOne 
#.....Functions for calculating the convolution filter Fy..... algorithm 1 in paper. ultimately returns Fy
def companion_krylov(L, p, b, c=None, c_tilde=None):# bug is here, since i use companion_krylov for multiplication. A is overrided by companion from_P. edit this function. This works only when a is shift + low_rank, for our alpha use krylov
    """
    Compute the Krylov matrix (c^T b, c^T A b, c^T A^2 b, ...), where A = shift + p e_d^T.
    Arguments:
        p: (..., d), real
        b: (..., d), real
        c: (..., d), real. One can instead supply c_tilde (below).
        c_tilde: (..., d), real, where c_tilde = c^T (I - A^L)
    At least c or c_tilde must be supplied.
    """
    d = p.shape[-1]
    batch_size = p.shape[:-1]
    e_d = torch.zeros(*batch_size, d, device=p.device, dtype=p.dtype)
    e_d[..., -1] = 1.0
    assert e_d.shape == p.shape
    assert b.shape == p.shape
    if c_tilde is None:
        assert c is not None, 'at least c or c_tilde must be supplied'
        assert c.shape == p.shape
        #todo; plugin a conditional statement here
        A = companion_from_p(p)#we can change the code here for initializing differant matrices
        c_tilde = c - torch.einsum('...m,...mn->...n', c, torch.linalg.matrix_power(A, L).to(dtype=c.dtype))#check torch.einsum for clarity. einsum is actually einstein summation notation
    else:
        assert c_tilde.shape == p.shape

    def fft_conv(u, v):  # This is actually convolution and not cross-correlation. 
        d = u.shape[-1]
        u_f = torch.fft.rfft(u, n=2 * d)#rfft is real fast fourier transform where the inputs are real. In our case u contains real values
        v_f = torch.fft.rfft(v, n=2 * d)#here since n =2*d > d the input is padded with zeros
        return torch.fft.irfft(u_f * v_f.conj(), n=2 * d)[..., :d]#irfft is inverse real fourier transform

    def quadratic_form(u, v):#value representation of polynomial(since a series can be represented as a polynomial function of certain degree)#check the video of fft and polynomial representation
        # and dicrete fourier transform or fast fourier transform can be used to represent it as value representation
        # u, v are vectors belongs to R^d
        d_rounded = math.ceil(d / L) * L
        # The reduce is to deal with the case where d > L
        return torch.fft.rfft(reduce(F.pad(fft_conv(u, v), (0, d_rounded - d)),
                                     '... (m L) -> ... L', L=L, reduction='sum'), n=L)


    Zconj = torch.exp(1j * 2 * math.pi * torch.arange(L // 2 + 1, dtype=torch.float32, device=p.device) / L)#e^{j*theta}..  polar form
    # woodbury = quadratic_form(c_tilde, b) + quadratic_form(c_tilde, p) * quadratic_form(e_d, b) / (Zconj - quadratic_form(e_d, p))
    quad = quadratic_form(rearrange(torch.stack([c_tilde, e_d], dim=-2), '... two d -> ... two 1 d'),
                          rearrange(torch.stack([b, p], dim=-2), '... two d -> ... 1 two d'))#quad(u,v)
    woodbury = quad[..., 0, 0, :] + quad[..., 0, 1, :] * quad[..., 1, 0, :] / (Zconj - quad[..., 1, 1, :])#step 3 in the algorithm
    woodbury_irfft = torch.fft.irfft(woodbury, n=L)#woodbury identity matrix is used for calculating the inverse of higher dimensional matrices efficiently
    return woodbury_irfft


if __name__ == '__main__':
    torch.manual_seed(0)
    d = 25 #hidden state dimension or vector a = [a0,a1,...,a_d-1] dimension
    L = 9 #length of the sequence
    H = 2 #number of heads, the values in two heads are initialized differently, so provides different SSM features
    p = torch.randn(H, d)
    p /= torch.linalg.norm(p, ord=1, dim=-1, keepdim=True)
    b = torch.randn(H, d) #B initialization
    c = torch.randn(H, d) #C initialization

    A = companion_from_p(p) #
    print(p.shape)
    print(p[...,:-1])
    print((A[0,:,-1],A[1,:,-1])) #to check the vectors are initilized randomly 

    from src.ops.krylov import krylov
    K = krylov(L, A, b, c)#convolution filter F_x
    K_fast = companion_krylov(L, p, b, c=c)#convolution filter by FFT
    print((K - K_fast).abs().max())#Checking the error in fft transformed matrix

    from benchmarks.utils import benchmark_all

    torch.manual_seed(0)
    d = 512
    L = 1024
    H = 256
    p = torch.randn(H, d, device='cuda', requires_grad=True)
    p = p / torch.linalg.norm(p, ord=1, dim=-1, keepdim=True)
    b = torch.randn(H, d, device='cuda', requires_grad=True)
    c = torch.randn(H, d, device='cuda', requires_grad=True)
    A = companion_from_p(p)

    benchmark_all(krylov, L, A, b, c, desc='krylov')
    benchmark_all(companion_krylov, L, p, b, c, desc='companion fast krylov')
    benchmark_all(companion_krylov, L, p, b, c_tilde=c, desc='companion fast krylov c_tilde')