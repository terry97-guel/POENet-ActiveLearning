import torch
from model import *
from dataloader import *
from utils.pyart import *
import argparse
import numpy as np
from pathlib import Path

def main(args):
    print("Processing...")

    # make save_dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    # n_joint = checkpoint['n_joint']
    # input_dim = checkpoint['input_dim']
    
    n_joint = 12
    input_dim = 2

    # load model
    model = Model(n_joint, input_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # load data
    test_data_loader = ToyDataloader(args.data_path, 1, 1, shuffle=False)

    # get jointAngle.txt
    jointAngle = np.array([]).reshape(-1,n_joint)
    for input,_ in test_data_loader:
        
        jointAngle_temp = model.q_layer(input)
        jointAngle_temp = jointAngle_temp.detach().cpu().numpy()
        jointAngle = np.vstack((jointAngle,jointAngle_temp))
    np.savetxt(args.save_dir+"/jointAngle.txt", jointAngle)

    # get jointTwist.txt
    jointTwist = np.array([]).reshape(-1,6)
    twists = model.poe_layer.twist
    for twist in twists:
        twist = twist.detach().cpu().numpy()
        jointTwist = np.vstack((jointTwist,twist))
    np.savetxt(args.save_dir+'/jointTwist.txt',jointTwist)

    # get M_se3.txt
    p = model.poe_layer.init_p
    r = model.poe_layer.init_rpy
    M_se3 = pr2t(p,r)
    M_se3 = M_se3.squeeze(0).detach().cpu().numpy()
    np.savetxt(args.save_dir+'/M_se3.txt',M_se3)

    # get targetPose.txt
    targetPose = test_data_loader.dataset.label
    targetPose = targetPose.detach().cpu().numpy()
    np.savetxt(args.save_dir+'/targetPose.txt', targetPose)

    # get outputPose.txt
    outputPose = np.array([]).reshape(-1,3)
    for input,_ in test_data_loader:
        outputPose_temp = model(input)
        outputPose_temp = outputPose_temp.squeeze(0).detach().cpu().numpy()
        outputPose_temp = outputPose_temp[:3,3]
        outputPose = np.vstack((outputPose,outputPose_temp))
    np.savetxt(args.save_dir+"/outputPose.txt", outputPose)

    print("Done...")
if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'parse for POENet')
    args.add_argument('--data_path', \
        default= './data/2dim_log_spiral/fold9/2dim_log_spiral_921.txt',type=str, \
            help='path to model checkpoint')    
    args.add_argument('--checkpoint', default= './output/1116/checkpoint_30.pth',type=str,
                    help='path to model checkpoint')
    args.add_argument('--save_dir', default='./2Visualize')
    args = args.parse_args()
    main(args)