import torch

def log_spiral(input):
    a = input[0][0]
    k = input[0][1]
    theta = 2/k
    x = a*torch.exp(k*theta)*torch.cos(theta)
    y = a*torch.exp(k*theta)*torch.sin(theta)
    z = 0

    return torch.tensor([x,y,z])