#%%
import torch
from model import Model
import numpy as np
import argparse
from utils.pyart import *
from utils.curve import *
from loss import *
from torch.autograd.functional import jacobian

def input2lossfn(model,target_pos):
    def input2loss(input):
        output = model(input)
        loss = Pos_norm2(output,target_pos).squeeze(0)
        return loss
    return input2loss

def main(args):
    weight = torch.load('./output/1116/checkpoint_100.pth')
    n_joint = weight['n_joint']
    input_dim = weight['input_dim']

    model = Model(n_joint, input_dim)
    model.load_state_dict(weight['state_dict'])

    a = args.a
    k = args.k

    target_input = torch.tensor([a,k]).unsqueeze(0)
    target_pos = log_spiral(target_input).unsqueeze(0)
    fn = input2lossfn(model,target_pos)

    input = torch.rand(1,2)
    input[0][0] = torch.Tensor(1).uniform_(0.5,3.5)
    input[0][1] = torch.Tensor(1).uniform_(0.2,3.2)
    # input = target_input
    loss = 1
    loop = 0
    while loss > 1e-2:
        loop = loop + 1
        jacob = jacobian(fn,input)
        loss = fn(input)
        print("Iter:{}  Loss:{}, input#1:{}, input#2:{}".format(loop,loss, input[0][0], input[0][1]))
    
        input[0][0] = input[0][0] -args.nx * jacob[0][0]
        input[0][1] = input[0][1] -args.ny * jacob[0][1]
        input[0][0].clamp_(0.5,3.5)
        input[0][1].clamp_(0.2,3.2)
        

    OutputTxt = np.array([])
    OutputTxt = np.append(OutputTxt, t2p(model(input)).detach().numpy())
    OutputTxt = np.append(OutputTxt, input.squeeze(0).numpy())
    # np.savetxt(args.save_dir,OutputTxt)
    print(OutputTxt)
    print(target_pos)
    
    pass
if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'parse for POENet')
    args.add_argument('--a', default= 1.0, type=float,
                    help='control input #1')
    args.add_argument('--k', default= 3.0, type=float,
                    help='control input #2')
    args.add_argument('--nx', default= 0.05, type=float,
                    help='step size for input #1')
    args.add_argument('--ny', default= 0.05, type=float,
                    help='step size for input #2')
    args.add_argument('--save_dir', default='./2Visualize')
    args = args.parse_args()
    main(args)

#%%
