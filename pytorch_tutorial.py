import torch

x = torch.empty(3, 2, 6)
print(x)

y = torch.rand(2,3)
print(y)

print(y.dtype)

x = torch.ones(2,2, dtype=torch.float16)

x = torch.tensor([2.5, 0.1])
print(x)

