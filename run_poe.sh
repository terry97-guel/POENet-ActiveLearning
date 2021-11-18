#!/bin/bash

python train.py --batch_size=1024 --data_path='./data/2dim_log_spiral' --save_dir='./output/1024/fold1' --device='1' --epochs=50 --wandb --Foldstart=0 --Foldend=1

python train.py --batch_size=1024 --data_path='./data/2dim_log_spiral' --save_dir='./output/1024/fold2' --device='1' --epochs=50 --wandb --Foldstart=0 --Foldend=2

python train.py --batch_size=1024 --data_path='./data/2dim_log_spiral' --save_dir='./output/1024/fold3' --device='1' --epochs=50 --wandb --Foldstart=0 --Foldend=3

python train.py --batch_size=1024 --data_path='./data/2dim_log_spiral' --save_dir='./output/1024/fold4' --device='1' --epochs=50 --wandb --Foldstart=0 --Foldend=4

python train.py --batch_size=1024 --data_path='./data/2dim_log_spiral' --save_dir='./output/1024/fold5' --device='1' --epochs=50 --wandb --Foldstart=0 --Foldend=5

python train.py --batch_size=1024 --data_path='./data/2dim_log_spiral' --save_dir='./output/1024/fold6' --device='1' --epochs=50 --wandb --Foldstart=0 --Foldend=6

python train.py --batch_size=1024 --data_path='./data/2dim_log_spiral' --save_dir='./output/1024/fold7' --device='1' --epochs=50 --wandb --Foldstart=0 --Foldend=7

python train.py --batch_size=1024 --data_path='./data/2dim_log_spiral' --save_dir='./output/1024/fold8' --device='1' --epochs=50 --wandb --Foldstart=0 --Foldend=8

python train.py --batch_size=1024 --data_path='./data/2dim_log_spiral' --save_dir='./output/1024/fold9' --device='1' --epochs=50 --wandb --Foldstart=0 --Foldend=9




