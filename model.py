#%%
import torch.nn as nn
import torch
import torch.functional as F
from utils.pyart import *


class POELayer(nn.Module):
    def __init__(self, n_joint):
        super(POELayer, self).__init__()
        self.n_joint = n_joint
        self.twist = nn.Parameter(torch.Tensor(n_joint,6))
        self.init_p = nn.Parameter(torch.Tensor(1,3))
        self.init_rpy =  nn.Parameter(torch.Tensor(1,3))
        self.twist.data.uniform_(-1,1)
        self.init_p.data.uniform_(-1,1)
        self.init_rpy.data.uniform_(-1,1)

        # init_SE3 = pr2t(self.init_p,self.init_rpy)
        # self.init_SE3 = init_SE3
        # self.register_buffer('init_SE3',init_SE3)

    def forward(self, q_value):
        n_joint = self.n_joint
        batch_size = q_value.size()[0]
        device = q_value.device
        out = torch.tile(torch.eye(4),(batch_size,1,1)).to(device)
        Twistls = torch.zeros([batch_size,n_joint,6]).to(device)
        for joint in range(n_joint):
            twist = self.twist[joint,:]
            Twistls[:,joint,:] = inv_x(t2x(out))@twist
            out = out @ srodrigues(twist, q_value[:,joint])
        out =  out @ pr2t(self.init_p,self.init_rpy)
        return out,Twistls

class q_layer(nn.Module):
    def __init__(self,n_joint,inputdim,n_layers=7):
        super(q_layer, self).__init__()
        
        LayerList = []
        for _ in range(n_layers):
            layer = nn.Linear(inputdim,2*inputdim)
            torch.nn.init.xavier_uniform_(layer.weight)
            LayerList.append(layer)
            inputdim = inputdim * 2

        for _ in range(n_layers-3):
            layer = nn.Linear(inputdim,inputdim//2)
            torch.nn.init.xavier_uniform_(layer.weight)
            LayerList.append(layer)
            inputdim = inputdim // 2

        layer = nn.Linear(inputdim,n_joint)
        torch.nn.init.xavier_uniform_(layer.weight)
        LayerList.append(layer)

        self.layers = torch.nn.ModuleList(LayerList)
        

    def forward(self, motor_control):
        out =motor_control
        
        for layer in self.layers:
            out = layer(out)
            out = torch.nn.LeakyReLU()(out)
    
        q_value = out
        return q_value

class Model(nn.Module):
    def __init__(self, n_joint, inputdim):
        super(Model,self).__init__()
        self.q_layer = q_layer(n_joint, inputdim)
        self.poe_layer = POELayer(n_joint)

    def forward(self, motor_control):
        out = self.q_layer(motor_control)
        SE3,_ = self.poe_layer(out)

        return SE3




#%%