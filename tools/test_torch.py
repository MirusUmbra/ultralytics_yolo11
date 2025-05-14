import torch

a = torch.rand(2, 3, 6)
a = a.split([1, 1, 3], dim=0)
print(a)