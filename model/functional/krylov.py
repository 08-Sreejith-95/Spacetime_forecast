""" Compute a Krylov function efficiently. (S3 renames the Krylov function to a "state space kernel")

A : (N, N)
b : (N,)
c : (N,)
Return: [c^T A^i b for i in [L]]
"""
#one crucial thing is that the resulting sequential tensors ,like the state x, should be contigous in memory. so torch.contigous() is used regularly 


import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from model.functional.toeplitz import causal_convolution

def krylov_sequential(L, A, b, c=None):
    """ Constant matrix A

    A : (..., N, N)
    b : (..., N)
    c : (..., N)

    Returns
    if c:
    x : (..., L)
    x[i, l] = c[i] @ A^l @ b[i]

    else:
    x : (..., N, L)
    x[i, l] = A^l @ b[i]
    """

    # Check which of dim b and c is smaller to save memory
    if c is not None and c.numel() < b.numel():
        return krylov_sequential(L, A.transpose(-1, -2), c, b)

    b_ = b
    x = []
    for _ in range(L):
        if c is not None:
            x_ = torch.sum(c*b_, dim=-1) # (...) # could be faster with matmul or einsum?
        else:
            x_ = b_
        x.append(x_)
        b_ = (A @ b_.unsqueeze(-1)).squeeze(-1)

    x = torch.stack(x, dim=-1)
    return x

#have a look at the krylov iterative method for matrix multiplication with higher degree
def krylov(L, A, b, c=None, return_power=False):
    """
    Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.

    If return_power=True, return A^{L-1} as well
    """
    # TODO There is an edge case if L=1 where output doesn't get broadcasted, which might be an issue if caller is expecting broadcasting semantics... can deal with it if it arises
    #This is used to compute the convolution filter F_x and F_y for (B, AB, A^2B, A^3B, ..., A^{L-1}B), where A-> R^{dxd}, A shape = (...,d, d), x = R^{dxL}  L = sequence length
    x = b.unsqueeze(-1) # (..., N, 1) #state initialization
    A_ = A #matrix initialization first power

    AL = None #Lth power, initially == none. this is iteratively updated
    if return_power:
        AL = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        _L = L-1

    done = L == 1 #if L ==1 -> A{L-1} = I, so initially for the first sample in the sequence, the state is X=0, since there are no history, so this iterative method is initialised 
    #with X_0 = 0... check the paper for clarity
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _L % 2 == 1: AL = A_ @ AL
            _L //= 2

        # Save memory on last iteration
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., :L-l]
        else: _x = x

        try:
            _x = A_ @ _x
        except Exception as e:
            print(e)
            breakpoint()
        x = torch.cat([x, _x], dim=-1) # there might be a more efficient way of ordering axes
        if not done: A_ = A_ @ A_

    try:
        assert x.shape[-1] == L
    except:
        print('x.shape', x.shape)
        print('L', L)
        breakpoint()

    if c is not None:
        x = torch.einsum('...nl, ...n -> ...l', x, c)
    x = x.contiguous() # WOW!!
    if return_power:
        return x, AL
    else:
        return x

def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A] #initially just A, later after each iteration in the loop below, it becomes the following list [A,A^2, A^3,....., A^{L-1}]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)

def krylov_toeplitz(L, A, b, c=None):
    """ Specializes to lower triangular Toeplitz matrix A represented by its diagonals

    A : (..., N)
    b : (..., N)
    c : (..., N)

    Returns
    x : (..., N, L)
    x[i, l] = A^l @ b[i]
    """
    x = b.unsqueeze(0) # (1, ..., N)
    A_ = A
    while x.shape[0] < L:
        xx = causal_convolution(A_, x)
        x = torch.cat([x, xx], dim=0) # there might be a more efficient way of ordering axes
        A_ = causal_convolution(A_, A_)
    x = x[:L, ...] # (L, ..., N)
    if c is not None:
        x = torch.einsum('l...n, ...n -> ...l', x, c)
    else:
        x = rearrange(x, 'l ... n -> ... n l')
    x = x.contiguous()
    return x

def krylov_toeplitz_(L, A, b, c=None):
    """ Padded version of krylov_toeplitz that saves some fft's

    TODO currently not faster than original version, not sure why
    """
    N = A.shape[-1]

    x = b.unsqueeze(0) # (1, ..., N)
    x = F.pad(x, (0, N))
    A = F.pad(A, (0, N))
    done = L == 1
    while not done:
        l = x.shape[0]
        # Save memory on last iteration
        if L - l <= l:
            done = True
            _x = x[:L-l]
        else: _x = x
        Af = torch.fft.rfft(A, n=2*N, dim=-1)
        xf = torch.fft.rfft(_x, n=2*N, dim=-1)
        xf_ = Af * xf
        x_ = torch.fft.irfft(xf_, n=2*N, dim=-1)
        x_[..., N:] = 0
        x = torch.cat([x, x_], dim=0) # there might be a more efficient way of ordering axes
        if not done:
            A = torch.fft.irfft(Af*Af, n=2*N, dim=-1)
            A[..., N:] = 0
    x = x[:L, ..., :N] # (L, ..., N)
    if c is not None:
        x = torch.einsum('l...n, ...n -> ...l', x, c)
    else:
        x = rearrange(x, 'l ... n -> ... n l')
    x = x.contiguous()
    return x
