import torch

x = torch.rand(3, requires_grad=True)

print(x)
y = x + 2


# Whenever we do opertions with this tensor, pytorch will create a so-called computational graph for us.

print(y)

z  = y*y*2
z = z.mean()
print(z)

z.backward() # dz/dx
print(x.grad)

# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():

x.requires_grad_(False)
print(x)

#Whenever we want to calculate hte gradients we must speciy the requieres_grad parameter and  set the parameter to true then we can simply calculate 
#calculate the gradients with calling the backward function and befor we want to do the next operation or the next iteration in our
#optimization steps, we must empty our gradient so we must call the zero function again "weights.grad.zero()" and we also should konw how 
# we can prevent some operations from being tracked in the computational graph. 