#%%
import torch


class Get_Loss_Function():
    def Pos_norm2(self,output, label):
        loss = torch.nn.MSELoss()(output,label)
        return loss
    
    
    # def se3_norm2(output, label):
        #Fill me

