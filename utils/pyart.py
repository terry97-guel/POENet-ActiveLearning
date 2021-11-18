import torch
# def expm(vector):
#     if vector.shape[0] == 6:
#         return to_SE3(vector)
#     if vector.shape[0] == 4:
#         return to_SO3(vector)

def t2pr(t):
    p = t[:,0:3,3]
    r = t[:,0:3,0:3]
    return (p,r)

def t2p(t):
    device = t.device
    p = t[:,0:3,3]
    return p

def pr2x(p,r):
    device = p.device
    X = torch.zeros(p.size()[0],6,6,dtype=torch.float).to(device)

    E = torch.transpose(r,1,2)

    X[:,:3,:3] = E
    X[:,3:,:3] = -torch.matmul(E,skew(p))
    X[:,3:,3:] = E

    return X

def t2x(t):
    (p,r) = t2pr(t)
    
    X = pr2x(p,r)

    return X

def pr2t(p,r):
    device = p.device
    T = torch.zeros(p.size()[0],4,4,dtype=torch.float).to(device)

    
    T[:,0:3,0:3] = r[:]
    T[:,0:3,3] =  p[:]
    T[:,3,3] = 1

    return T

def skew(p):
    device = p.device
    skew_p = torch.zeros(p.size()[0],3,3,dtype=torch.float).to(device)

    zero = torch.zeros(p.size()[0],dtype=torch.float).to(device)
    skew_p[:,0,:] = torch.vstack([zero, -p[:,2], p[:,1]]).transpose(0,1)
    skew_p[:,1,:] = torch.vstack([p[:,2], zero, -p[:,0]]).transpose(0,1)
    skew_p[:,2,:] = torch.vstack([-p[:,1], p[:,0],zero]).transpose(0,1)


    return skew_p

def rpy2r(rpy):
    device = rpy.device
    R = torch.zeros(rpy.size()[0],3,3,dtype=torch.float).to(device)

    r = rpy[:,0]
    p = rpy[:,1]
    y = rpy[:,2]

    R[:,0,:] = torch.vstack([
        torch.cos(y)*torch.cos(p),
        -torch.sin(y)*torch.cos(r) + torch.cos(y)*torch.sin(p)*torch.sin(r),
        torch.sin(y)*torch.sin(r)+torch.cos(y)*torch.sin(p)*torch.cos(r)
        ]).transpose(0,1)

    R[:,1,:] = torch.vstack([
        torch.sin(y)*torch.cos(p),
        torch.cos(y)*torch.cos(r) + torch.sin(y)*torch.sin(p)*torch.sin(r),
        -torch.cos(y)*torch.sin(r)+torch.sin(y)*torch.sin(p)*torch.cos(r)
        ]).transpose(0,1)

    R[:,2,:] = torch.vstack([
        -torch.sin(p),
        torch.cos(p)*torch.sin(r),
        torch.cos(p)*torch.cos(r)
        ]).transpose(0,1)
    
    return R

def inv_x(x):
    device = x.device
    invX = torch.zeros(x.size()[0],6,6,dtype=torch.float).to(device)

    E = x[:,:3,:3]
    temp = x[:,3:,:3]
    
    invX[:,:3,:3] = torch.transpose(E,1,2)
    invX[:,:3,3:] = torch.zeros(x.size()[0],3,3).to(device)
    invX[:,3:,:3] = torch.transpose(temp,1,2)
    invX[:,3:,3:] = torch.transpose(E,1,2)

    return invX

def srodrigues(twist, q_value, verbose =False): #number of set of twist is one & number of q_value is n_joint
    eps = 1e-10
    device = twist.device
    batch_size = q_value.size(0)
    T = torch.zeros(batch_size,4,4,dtype=torch.float).to(device)

    #number of joint
    w = twist[:3]
    v = twist[3:]
    theta = w.norm(dim=0)

    if theta.item() < eps:
        theta = v.norm(dim=0)

    q_value = q_value * theta
    w = w/theta
    v = v/theta
    w_skew = skew(w.unsqueeze(0)).squeeze(0)

    # print("q_value:", q_value.device)
    # print("w:", w.device)
    # print("(1-torch.cos(q_value):", (1-torch.cos(q_value)).device)
    # print("w_skew @ v:", (w_skew @ v).device)

    T[:,:3,:3] = rodrigues(w, q_value)
    T[:,:3,3] =  torch.outer(q_value,v) + \
        torch.outer((1-torch.cos(q_value)), w_skew @ v) + \
        torch.outer(q_value-torch.sin(q_value), w_skew@w_skew@v)
    T[:,3,3] = 1
    
    return T

def rodrigues(w,q,verbose = False):
    eps = 1e-10
    device = q.device
    batch_size = q.size()[0]

    if torch.norm(w) < eps:
        R = torch.tile(torch.eye(3),(batch_size,1,1)).to(device)
        return R
    if abs(torch.norm(w)-1) > eps:
        if verbose:
            print("Warning: [rodirgues] >> joint twist not normalized")

    theta = torch.norm(w)
    w = w/theta
    q = q*theta

    w_skew = skew(w.unsqueeze(0)).squeeze(0)
    R = torch.tensordot(torch.ones_like(q).unsqueeze(0), torch.eye(3).unsqueeze(0).to(device), dims=([0],[0])) \
        + torch.tensordot(torch.sin(q).unsqueeze(0), w_skew.unsqueeze(0),dims = ([0],[0]))\
            + torch.tensordot( (1-torch.cos(q)).unsqueeze(0), (w_skew@w_skew).unsqueeze(0), dims =([0],[0]))
    return R
#%%