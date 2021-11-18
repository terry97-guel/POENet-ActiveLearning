#%%
import argparse
import torch
import numpy as np
from dataloader import *
from model import Model
# from trainer import Trainer
from loss import *
import os
import random
from pathlib import Path
import wandb
import time

# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def train_epoch(model, optimizer, input, label,Loss_Fn, args):
    q_value = model.q_layer(input)
    q_loss = q_entropy(torch.abs(q_value))
    q_loss = torch.mean(q_loss,dim=0)
    total_loss = - args.q_entropy * q_loss 

    output, Twistls = model.poe_layer(q_value)
    loss = Loss_Fn(output,label)
    regularizer_loss = args.Twist_norm * Twist_norm(model)
    regularizer_loss = regularizer_loss + args.Twist2point * Twist2point(Twistls,label)
    total_loss = total_loss + loss + regularizer_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss

def test_epoch(model, input, label, Loss_Fn, args):
    q_value = model.q_layer(input)
    q_loss = q_entropy(torch.abs(q_value))
    q_loss = torch.mean(q_loss,dim=0)
    q_loss = args.q_entropy * q_loss

    output,Twistls = model.poe_layer(q_value)
    Twist2pointloss = args.Twist2point * Twist2point(Twistls,label)

    loss = Loss_Fn(output,label)
    regularizer_loss = args.Twist_norm * Twist_norm(model)

    total_loss = loss + regularizer_loss

    return total_loss,q_loss,Twist2pointloss

def main(args):
    #set logger
    if args.wandb:
        wandb.init(project = args.pname)

    #set device
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    #set model
    model = Model(args.n_joint, args.input_dim)
    model = model.to(device)

    #load weight when requested
    if os.path.isfile(args.resume_dir):
        weight = torch.load(args.resume_dir)
        model.load_state_dict(weight['state_dict'])
        print("loading successful!")
    #set optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr= args.lr, weight_decay=args.wd)


    #declare loss function
    if args.loss_function == 'Pos_norm2':
        Loss_Fn = Pos_norm2
    else:
        print("Invalid loss_function")
        exit(0)
    

    #assert path to save model
    pathname = args.save_dir
    Path(pathname).mkdir(parents=True, exist_ok=True)

    #set dataloader
    print("Setting up dataloader")
    train_data_loader = FoldToyDataloader(args.data_path, args.Foldstart, args.Foldend, args.n_workers, args.batch_size)
    test_data_loader = FoldToyDataloader(args.data_path, args.Foldend, -1, args.n_workers, args.batch_size)
    
    print("Initalizing Training loop")
    for epoch in range(args.epochs):
        # Timer start
        time_start = time.time()

        # Train
        model.train()
        data_length = len(train_data_loader)
        for iterate, (input,label) in enumerate(train_data_loader):
            input = input.to(device)
            label = label.to(device)
            train_loss = train_epoch(model, optimizer, input, label, Loss_Fn, args)
            print('Epoch:{}, TrainLoss:{:.2f}, Progress:{:.2f}%'.format(epoch,train_loss,100*iterate/data_length), end='\r')
        print('Epoch:{}, TrainLoss:{:.2f}, Progress:{:.2f}%'.format(epoch,train_loss,100*iterate/data_length))
        
        #Evaluate
        model.eval()
        data_length = len(test_data_loader)
        test_loss = np.array([])
        for iterate, (input,label) in enumerate(test_data_loader):
            input = input.to(device)
            label = label.to(device)
            total_loss,q_loss,Twist2pointloss = test_epoch(model, input, label, Loss_Fn, args)
            total_loss = total_loss.detach().cpu().numpy()
            test_loss = np.append(test_loss, total_loss)
            print('Testing...{:.2f} Epoch:{}, Progress:{:.2f}%'.format(total_loss,epoch,100*iterate/data_length) , end='\r')
        
        test_loss = test_loss.mean()
        print('TestLoss:{:.2f}'.format(test_loss))

        # Timer end    
        time_end = time.time()
        avg_time = time_end-time_start
        eta_time = (args.epochs - epoch) * avg_time
        h = int(eta_time //3600)
        m = int((eta_time %3600)//60)
        s = int((eta_time %60))
        print("Epoch: {}, TestLoss:{:.2f}, eta:{}:{}:{}".format(epoch, test_loss, h,m,s))
        
        # Log to wandb
        if args.wandb:
            wandb.log({'TrainLoss':train_loss, 'TestLoss':test_loss, 'TimePerEpoch':avg_time,
            'q_entropy':q_loss,'Twist2point':Twist2pointloss},step = epoch)

        #save model 
        if (epoch+1) % args.save_period==0:
            filename =  pathname + '/checkpoint_{}.pth'.format(epoch+1)
            print("saving... {}".format(filename))
            state = {
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'n_joint':args.n_joint,
                'input_dim':args.input_dim
            }
            torch.save(state, filename)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'parse for POENet')
    args.add_argument('--n_joint', default= 12, type=int,
                    help='number of joints')
    args.add_argument('--batch_size', default= 1024*8, type=int,
                    help='batch_size')
    args.add_argument('--data_path', default= './data/2dim_log_spiral',type=str,
                    help='path to data')
    args.add_argument('--save_dir', default= './output/temp',type=str,
                    help='path to save model')
    args.add_argument('--resume_dir', default= './output/',type=str,
                    help='path to load model')
    args.add_argument('--device', default= '1',type=str,
                    help='device to use')
    args.add_argument('--n_workers', default= 2, type=int,
                    help='number of data loading workers')
    args.add_argument('--wd', default= 0.001, type=float,
                    help='weight_decay for model layer')
    args.add_argument('--lr', default= 0.001, type=float,
                    help='learning rate for model layer')
    # args.add_argument('--optim', default= 'adam',type=str,
    #                 help='optimizer option')
    args.add_argument('--loss_function', default= 'Pos_norm2', type=str,
                    help='get list of loss function')
    args.add_argument('--Twist_norm', default= 0.01, type=float,
                    help='Coefficient for TwistNorm')
    args.add_argument('--q_entropy', default= 0.01, type=float,
                    help='Coefficient for q_entropy')
    args.add_argument('--Twist2point', default= 0.01, type=float,
                    help='Coefficient for Twist2point')
    args.add_argument('--Twist2Twist', default= 0.1, type=float,
                    help='Coefficient for Twist2point')
    args.add_argument('--wandb', action = 'store_true', help = 'Use wandb to log')
    args.add_argument('--input_dim', default= 2, type=int,
                    help='dimension of input')
    args.add_argument('--epochs', default= 100, type=int,
                    help='number of epoch to perform')
    # args.add_argument('--early_stop', default= 50, type=int,
    #                 help='number of n_Scence to early stop')
    args.add_argument('--save_period', default= 1, type=int,
                    help='number of scenes after which model is saved')
    args.add_argument('--pname', default= 'POE2D-1116',type=str,
                    help='Project name')
    args.add_argument('--Foldstart', default= 0, type=int,
                    help='Number of Fold to start')
    args.add_argument('--Foldend', default= 8, type=int,
                    help='Number of Fole to end')
    args = args.parse_args()
    main(args)
#%%
