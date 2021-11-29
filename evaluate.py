import torch
from model import *
from dataloader import *
import argparse
import numpy as np
from pathlib import Path

def main(args):
    print("Processing...")

    # make save_dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    input_dim = checkpoint['input_dim']
    
    # load model
    model = Model(input_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # load data
    test_data_loader = ToyDataloader(args.data_path, 1, 1, shuffle=False)

    # get targetPose.txt
    targetPose = test_data_loader.dataset.label
    targetPose = targetPose.detach().cpu().numpy()
    np.savetxt(args.save_dir+'/targetPose.txt', targetPose)

    # get outputPose.txt
    outputPose = np.array([]).reshape(-1,3)
    for input,_ in test_data_loader:
        outputPose_temp = model(input)
        outputPose_temp = outputPose_temp.squeeze(0).detach().cpu().numpy()
        outputPose = np.vstack((outputPose,outputPose_temp))
    np.savetxt(args.save_dir+"/NaiveOutputPose.txt", outputPose)

    print("Done...")
if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'parse for POENet')
    args.add_argument('--data_path', \
        default= './data/2dim_log_spiral/test/2dim_log_spiral_1000.txt',type=str, \
            help='path to model checkpoint')    
    args.add_argument('--checkpoint', default= './output/1020/checkpoint_50.pth',type=str,
                    help='path to model checkpoint')
    args.add_argument('--save_dir', default='./2Visualize')
    args = args.parse_args()
    main(args)