
import torch

num_patches = 768
dim = 64
x = torch.randn(1,num_patches+1,dim)

print(x.shape)

