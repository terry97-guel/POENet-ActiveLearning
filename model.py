#%%
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self,inputdim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(inputdim,16)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2 = nn.Linear(16,32)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        self.layer3 = nn.Linear(32,64)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        self.layer4 = nn.Linear(64,128)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        self.layer5 = nn.Linear(128,256)
        torch.nn.init.xavier_uniform_(self.layer5.weight)
        self.layer6 = nn.Linear(256,512)
        torch.nn.init.xavier_uniform_(self.layer6.weight)

        self.layer11 = nn.Linear(512,512)
        torch.nn.init.xavier_uniform_(self.layer11.weight)
        self.layer12 = nn.Linear(512,512)
        torch.nn.init.xavier_uniform_(self.layer12.weight)
        self.layer13 = nn.Linear(512,512)
        torch.nn.init.xavier_uniform_(self.layer13.weight)
        self.layer14 = nn.Linear(512,256)
        torch.nn.init.xavier_uniform_(self.layer14.weight)
        
        self.layer7 = nn.Linear(256,128)
        torch.nn.init.xavier_uniform_(self.layer7.weight)
        self.layer8 = nn.Linear(128,64)
        torch.nn.init.xavier_uniform_(self.layer8.weight)
        self.layer9 = nn.Linear(64,32)
        torch.nn.init.xavier_uniform_(self.layer9.weight)
        self.layer10 = nn.Linear(32,3)
        torch.nn.init.xavier_uniform_(self.layer10.weight)


    def forward(self, motor_control):
        out = self.layer1(motor_control)
        out = nn.LeakyReLU()(out)

        out = self.layer2(out)
        out = nn.LeakyReLU()(out)

        out = self.layer3(out)
        out = nn.LeakyReLU()(out)

        out = self.layer4(out)
        out = nn.LeakyReLU()(out)

        out = self.layer5(out)
        out = nn.LeakyReLU()(out)

        out = self.layer6(out)
        out = nn.LeakyReLU()(out)

        out = self.layer11(out)
        out = nn.LeakyReLU()(out)
        out = self.layer12(out)
        out = nn.LeakyReLU()(out)
        out = self.layer13(out)
        out = nn.LeakyReLU()(out)
        out = self.layer14(out)
        out = nn.LeakyReLU()(out)
        
        out = self.layer7(out)
        out = nn.LeakyReLU()(out)

        out = self.layer8(out)
        out = nn.LeakyReLU()(out)

        out = self.layer9(out)
        out = nn.LeakyReLU()(out)

        out = self.layer10(out)
        out = nn.LeakyReLU()(out)

        pos_value = out
        
        return pos_value

#%%