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
        



#Creating SSM with A_alpha:- Thesis project experiment

class AlphaSSM(SSM):
    """
    Open-loop implementation of Companion SSM:
    -> y_t = C \sum_{i = 0}^{k - 1 - i} A^k B u_i
       where A is alpha matrix
    """
    def __init__(self, norm_order, k:int, n:int, **kwargs):
        self.norm_order = norm_order
        kwargs['kernel_repeat'] = 1
        kwargs['kernel_weights'] = None
        kwargs['kernel_train'] = True
        self.k = k #todo:- add to the config file
        self.n = n #todo:- add to the config file
        #----- 2*n + k = hid_dim, so in config for hid_size 64.. put n = 16 and k = 32-------#
        # Set kwargs['n_heads'] as n_kernels for preprocessing kernels
        # Set kwargs['head_dim'] to be original sample input dim
        super().__init__(**kwargs)
        self.kernel_dim = self.k + 2*self.n
        
    def init_kernel_weights(self, kernel_init): # vector alpha init with normalization
        if kernel_init == 'normal':
            kernel = torch.randn(self.n_kernels, self.kernel_dim)#(No. SSM, SSM dim) eg:- (16, 64). so alpha vector with dim 64 for 16 ssms. This is with normalization
        elif kernel_init == 'xavier':
            # Xavier-ish initialization
            stdv = 1. / math.sqrt(self.kernel_dim)
            kernel = torch.FloatTensor(self.n_kernels, 
                                       self.kernel_dim).uniform_(-stdv, stdv)
        else:
            raise NotImplementedError
        return kernel
     #--------------------------------------------------------------------------------#   
    def init_weights(self):#to_do:- while creating our custom SSMs changes should be made from here. Or use alphas as the vector instead of p and create A matrix accordingly
        super().init_weights()  # Initializes skip connection
        self._fp = (self.n_kernels, self.k)#alpha vector dim
        
        #Alpha vector initialization for each SSMs:- eg:- for 16 SSM's with 64 dim in a single ST layer it will be (16, 64) with different values
        #self.alphas = torch.rand(*self._fp)#alphas initialized with random values:- to_do: need to normalize and register as trainable params
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
        a = self.init_kernel_weights(self.kernel_init)#alphas eg: (16,64):- 
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
        # Construct Alpha matrix
        A = self.row_shift_matrix.to(p.device)
        for kl in range(self.n_kernels):
            for i in range(self.k):
                A[kl, 2*self.n + i, i] = 1 - p[kl, i]
                A[kl, 2*self.n + i, i +1] = p[kl, i]
        #print('Alpha_SSM1:', A[0])#to check if matrix is right
        #print('Alpha_SSM2', A[1])
        #print('Alpha_SSM3', A[2])
       #self.shift_matrix.to(p.device) + (
                          #oe.contract('h i, h j -> h j i', 
                          #self.p_padding.to(p.device), p) #This A Matrix value aftr each epoch should be saved to plot the A value changes #expecting:- a0 i.e, A[...,:,0]= max among other values. Here this needs to be changed
        
        # Use repeated squares to power A
        g = krylov(l, A, b, c)
        return g
    
    def get_kernel(self, u, c=None, l=None):
        l = u.shape[-1] if l is None else l
        c = self.c if c is None else c
        a = (self.norm(self.a, ord=self.norm_order) #L1 norm |a0| + |a1| + ....+|ad| = 1 #to_do for our A the normalization will be different since we are not initializing parameters at the last column. define separate func for norm
             if self.norm_order > 0 else self.a)#edit here
        f = self.matrix_power(l, c, self.b, a).to(u.device)# f is the filter f_y
        #print('Filter F_y', f, f.shape)
        return f
    
    def forward(self, u):
        return super().forward(u)
 
#if __name__ == '__main__':
       #compare this with companion SSM, here
       #alpha_ssm = AlphaSSM(n = 2, k = 4, norm_order=1, n_kernels = 3, kernel_dim = 8, model_dim = 16)
       #no_kernals = alpha_ssm.n_kernels
       #kernal_dim = alpha_ssm.kernel_dim
       #u = torch.rand(3, kernal_dim, 10)
       #alpha_ssm.get_kernel(u)
       
        