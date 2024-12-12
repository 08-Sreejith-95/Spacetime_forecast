import torch
import numpy as np

n = 2
k = 4
hid_size = 2*n + k
A = torch.zeros(3,hid_size,hid_size)
a = torch.rand(3,hid_size)
print(A)
print(a)

for i in range(3):
    for j in range(k):
        A[i, 2*n + j, j] = 1 - a[i,j]
        A[i, 2*n + j, j + 1] = a[i,j]
print(A)
    