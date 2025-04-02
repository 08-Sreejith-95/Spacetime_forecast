import torch
import torch.nn.functional as F
import opt_einsum as oe
from einops import repeat, rearrange
import math
###


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print(os.getcwd())
from model.functional.krylov import krylov
from model.functional.companion_krylov import companion_krylov
from model.ssm.base import SSM

#Add code here to save A matrix after each epoch to plot the ssm parameter updates
class CompanionSSM(SSM):
    """
    Open-loop implementation of Companion SSM:
    -> y_t = C \sum_{i = 0}^{k - 1 - i} A^k B u_i
       where A is companion matrix
    """
    def __init__(self, norm_order, **kwargs):
        self.norm_order = norm_order
        kwargs['kernel_repeat'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = True
        # Set kwargs['n_heads'] as n_kernels for preprocessing kernels
        # Set kwargs['head_dim'] to be original sample input dim
        super().__init__(**kwargs)
        
    def init_kernel_weights(self, kernel_init): # vector a init
        if kernel_init == 'normal':
            kernel = torch.randn(self.n_kernels, self.kernel_dim)
        elif kernel_init == 'xavier':
            # Xavier-ish initialization
            stdv = 1. / math.sqrt(self.kernel_dim)
            kernel = torch.FloatTensor(self.n_kernels, 
                                       self.kernel_dim).uniform_(-stdv, stdv)
        else:
            raise NotImplementedError
        return kernel
        
    def init_weights(self):#to_do:- while creating our custom SSMs changes should be made from here. Or use alphas as the vector instead of p and create A matrix accordingly
        super().init_weights()  # Initializes skip connection
        self._fp = (self.n_kernels, self.kernel_dim)
        
        # Shift matrix initialization
        self.shift_matrix = torch.zeros(self.n_kernels, 
                                        self.kernel_dim, 
                                        self.kernel_dim)
        self.shift_matrix[:, 1:, :-1] = torch.eye(self.kernel_dim - 1)
        self.p_padding = torch.zeros(*self._fp)
        self.p_padding[:, -1] = 1.
        
        # A matrix
        a = self.init_kernel_weights(self.kernel_init)#a vector
        self.register("a", a, trainable=True, lr=None, wd=None)
        
        # B matrix
        b = self.init_kernel_weights(self.kernel_init) 
        self.register("b", b, trainable=True, lr=None, wd=None)
        
        # C matrix
        c = self.init_kernel_weights(self.kernel_init)
        self.register("c", c, trainable=True, lr=None, wd=None)
    
    def norm(self, x, ord=1):# A vector normalization
        # x.shape is either (H x D) or (H x D x D)
        x_norm = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)#eigenvalue sum  = 1
        # If norm(x) in batch close to 0, don't normalize 
        # (heuristicky, but we norm for stability)
        try:
            x = x / x_norm if torch.abs(x_norm).mean().item() > 1e-4 else x  
        except Exception as e:
            print(e)
            breakpoint()
        # x = F.normalize(x, dim=1, p=ord, eps=1)
        return x
    
    def matrix_power(self, l, c, b, p):
        # Construct companion matrix
        A = self.shift_matrix.to(p.device) + (
            oe.contract('h i, h j -> h j i', 
                        self.p_padding.to(p.device), p) #This A Matrix value aftr each epoch should be saved to plot the A value changes #expecting:- a0 i.e, A[...,:,0]= max among other values
        )
        # Use repeated squares to power A
        g = krylov(l, A, b, c)
        return g
    
    def get_kernel(self, u, c=None, l=None):
        l = u.shape[-1] if l is None else l
        c = self.c if c is None else c
        a = (self.norm(self.a, ord=self.norm_order) #L1 norm |a0| + |a1| + ....+|ad| = 1 #to_do for our A the normalization will be different since we are not initializing parameters at the last column. define separate func for norm
             if self.norm_order > 0 else self.a)#edit here
        f = self.matrix_power(l, c, self.b, a).to(u.device)# f is the filter f_y
        return f
    
    def forward(self, u):
        return super().forward(u)
        



#---------------Creating SSM with Smoothing Delay Memory SSM:- Thesis project experiment:- state matrix_2-------------------#

class SDMSSM(SSM):
    """
    Open-loop implementation of Companion SSM:
    -> y_t = C \sum_{i = 0}^{k - 1 - i} A^k B u_i
       where A is alpha matrix
    """
    def __init__(self, norm_order, **kwargs):
        self.norm_order = norm_order
        kwargs['kernel_repeat'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = True
        self.segments = 8 #No of delayed active histories 
        self.cascades = 32 #No_of cascade blocks needed for EMA smoothing for the delayed histories
        self.A = None
        super().__init__(**kwargs)
        self.kernel_dim = self.segments *((self.cascades * 2) + 1) #The  hidden state space dimension increases considerably
        
    def init_kernel_weights(self, kernel_init, alphas = False): #The No of learnable parameters are decreased compared to the companion and alpha
        if not alphas:
            n_k, k_sz = (self.n_kernels, self.kernel_dim)
        else:
            n_k, k_sz = (self.n_kernels, self.cascades)
        
        if kernel_init == 'normal':
            kernel = torch.randn(n_k, k_sz)
        elif kernel_init == 'xavier':
            stdv = 1. / math.sqrt(k_sz)
            kernel = torch.FloatTensor(n_k, k_sz).uniform_(-stdv, stdv)
        else:
            raise NotImplementedError
        return kernel
     #--------------------------------------------------------------------------------#   
    def init_weights(self):
        super().init_weights()  # Initializes skip connection
        #self._fp = (self.n_kernels, self.cascades)#alpha vector dim
        
  
        # Shift matrix initialization
        self.row_shift_matrix = torch.zeros(self.n_kernels, 
                                        self.kernel_dim, 
                                        self.kernel_dim)
        #self.p = torch.rand(self._fp)#to_do:_ may be subject to change later
        #need to plugin alphas accordingly to create A matrix
        #self.row_shift_matrix[:, 1:, :-1] = torch.eye(self.kernel_dim - 1)#Here to skip couple of rows construct this matrix accordingly
        #self.p_padding = torch.zeros(*self._fp)#here in companion matrix the p vector is padded at the last column of A
        #self.p_padding[:, -1] = 1. #last column with 1
        
        # A matrix
        a = self.init_kernel_weights(self.kernel_init, alphas = True)#alphas_ EMA trainable parameters
        self.register("a", a, trainable=True, lr=None, wd=None)
        
        # B matrix
        b = self.init_kernel_weights(self.kernel_init) 
        self.register("b", b, trainable=True, lr=None, wd=None)
        
        # C matrix
        c = self.init_kernel_weights(self.kernel_init)
        self.register("c", c, trainable=True, lr=None, wd=None)
    
    def norm(self, x, ord=1):# A vector normalization
        # x.shape is either (H x D) or (H x D x D)
        x_norm = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)#eigenvalue sum  = 1
        # If norm(x) in batch close to 0, don't normalize 
        # (heuristicky, but we norm for stability)
        try:
            x = x / x_norm if torch.abs(x_norm).mean().item() > 1e-4 else x  
        except Exception as e:
            print(e)
            breakpoint()
        # x = F.normalize(x, dim=1, p=ord, eps=1)
        return x
    def construct_ssm(self,p):
        A = self.row_shift_matrix.to(p.device)
        for kl in range(self.n_kernels):
                for c in range(self.cascades): 
                    j_offset = 2 * c * self.segments
                    i_offset = self.segments + 2 * c * self.segments 
        # Copy (diagonal)
                    for d in range(self.segments):
                        A[kl, i_offset + d, j_offset + d] = 1.0
                        i_offset += self.segments
        # Merge (two diagonals)
                    for d in range(self.segments):
                        A[kl, i_offset + d, j_offset + d] = 0.5 * (1.0 - p[kl, c])
                        A[kl, i_offset + d, j_offset + d + 1 * self.segments] = 0.5 * (1.0 - p[kl, c])
                        A[kl, i_offset + d, j_offset + d + 2 * self.segments] = p[kl, c]
        return A

    
    def matrix_power(self, l, c, b, p):
        # Construct Alpha matrix
        self.A = self.construct_ssm(p)
        #print('Alpha_SSM1:', A[16, 34:62, 34:62])#to check if matrix is right
        g = krylov(l, self.A, b, c)
        return g
    
    def get_kernel(self, u, c=None, l=None):
        l = u.shape[-1] if l is None else l
        c = self.c if c is None else c
        a = (self.norm(self.a, ord=self.norm_order) #L1 norm |a0| + |a1| + ....+|ad| = 1 #to_do for our A the normalization will be different since we are not initializing parameters at the last column. define separate func for norm
             if self.norm_order > 0 else self.a)#edit here
        f = self.matrix_power(l, c, self.b, a).to(u.device)# f is the filter f_y
        print('Filter F_y', f, f.shape)
        return f
    
    def forward(self, u):
        return super().forward(u)
    
    
    
 

        