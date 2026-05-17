import torch
import numpy as np

import torch
W = torch.randn(2, 3, requires_grad=True)
x = torch.randn(3)
y = W @ x              # pure matmul, no activation
loss = (y**2).sum()
loss.backward()
print(W.grad)          # gradient flows through @ just fine
# print(torch.__version__)

# shape = (3,1)
# x = torch.rand(shape)
# y = torch.rand(1,3)

# print(y)
# print("shape: ", x.shape)
# print("type: ", x.dtype)
# print("device: ", x.device)

# z=x+y
# print(z)

# x = torch.tensor(2.0, requires_grad=True)
# y = torch.tensor(5.0, requires_grad=True)
# z = x ** 2 + 3 * y
# z.backward()
# print(x.grad)
# print(y.grad)
# print(z.grad_fn)

# #wgw
# x = torch.tensor(2.0, requires_grad=True)
# y = x ** 2
# print("y.grad_fn:", y.grad_fn)

# z = y + 1
# print("z.grad_fn:", z.grad_fn)
# print("z.grad_fn.next_functions:", z.grad_fn.next_functions)

# z.backward()
# print("x.grad:", x.grad)
# #print("y.grad:", y.grad)   # None — y is not a leaf!

# #backward on a tensor leaf

# x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# y = x ** 2          # shape (3,) — not a scalar
# y.sum().backward()        # RuntimeError: grad can be implicitly created only for scalar outputs
# print(x.grad)

# #complex autograd example
# print("tensor autograd example")
# x = torch.tensor([2.0, 3.0, 6.0],requires_grad=True)
# A = torch.tensor([[1.0, 2.0, 2.0],[2.0, 3.0, 1.0],[1.0, 1.0, 1.0]])
# b = torch.tensor([4.0, 3.0, 1.0])

# fx=x.T@A.T@x + b.T@x
# fx.backward()
# df = (A + A.T)@x + b
# print("fx: ", fx)
# print("xgrad", x.grad)
# print("dfdx: ", df)
