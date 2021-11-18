#%%
import torch


def Pos_norm2(output, label):
    output = output[:,0:3,3]
    loss = torch.nn.MSELoss()(output,label)

    return loss

def Twist_norm(model):
    loss = torch.norm(model.poe_layer.twist,dim=0)
    device = loss.device
    loss = loss - torch.ones(6,1,dtype=torch.float).to(device)
    loss = torch.norm(loss)

    return loss

def q_entropy(q_value):
    loss = torch.distributions.Categorical(q_value).entropy()

    return loss

def Twist2point(Twistls, label):
    w = Twistls[:,:,:3]
    v = Twistls[:,:,3:]

    p_o = torch.div(torch.cross(v,w),(torch.norm(w,dim=2)**2).unsqueeze(-1))
    p_t = torch.einsum('ij,ikj->ik',label,w)
    p_t = p_t/(torch.norm(w,dim=2)**2)
    p_t = torch.einsum('ij,ijk->ijk',p_t,w)+p_o
    loss = p_t-label.unsqueeze(1)
    loss = loss**2
    loss = loss.mean()
    
    return loss

# def Twist2Twsit(model):
#     for i,j in range(model.poe_layer.n_joint):
        
#     pass