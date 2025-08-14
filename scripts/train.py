######## model training loop #######


import preprocess as pre


# basic package
import numpy as np
import pickle
import tqdm 

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)
import os
import pathlib

# ML package
import torch # basic tensor operation
from torch.utils.data import DataLoader # build dataset and dataloader
import torch.optim as optim # optimizer


import copy
import argparse

from get_data import Mydataset
from losses import Loss
import preprocess as pre
from model import Unet


def parse_args():
    p = argparse.ArgumentParser(description='Parameters for training model')
    p.add_argument("--data_path",type=str,required=True,help='Saving path of MRI scans')
    p.add_argument("--loss_type",type=str,choices=['cce','dice','dice+cce','ftl+cce'],required=True,help='Loss mode')
    p.add_argument('--epoch',type=int,required=True,help='Epoch number of the training')
    

    p.add_argument("--lr",default=1e-3,type=float,help='Learning rate for training')
    p.add_argument('--use_scheduler',action='store_true')

    p.add_argument('--batch_size',default=32,type=int)
    p.add_argument('--seed',default=42,type=int)

    p.add_argument('--use_checkpoint',action='store_true',help='Whether use check point or not')
    p.add_argument('--checkpoint_file',type=str,help='Check point file to use')

    return p.parse_args()

def model_training(model,optimizer,epoch,dl_train,dl_val,lr_scheduler,device,loss_fn,metric_fn):
    
    # append dice and loss for each epoch
    epoch_loss = []
    epoch_dice = []
    epoch_val_loss = []
    epoch_val_dice=[]
    epoch_lr=[]
    
    best_dice = 0
    best_model_path = None
    model.to(device)


    for i in range(epoch):

        epoch_lr.append(optimizer.param_groups[0]['lr'])
        running_loss = 0
        running_dice = 0
        print(f"Epoch {i+1} | {epoch}, lr= {optimizer.param_groups[0]['lr']} \n")
        model.train()
        
        # training of the epoch
        for batch in tqdm.tqdm(dl_train):
            
            img = batch[0][0].to(device) # N 2 H W
            mask = batch[1][0].to(device) # N H W

            if torch.isnan(img).any():
                print('Nan in img')
                raise ValueError('Nan in img')
                

            batch_results = model(img) #N 4 H W, raw logit values from the model

            if torch.isnan(batch_results).any():
                print('Nan in logits')
                raise ValueError('Nan in logits')

            batch_Dice = metric_fn(batch_results,mask)
            #print(batch_Dice)
            running_dice+=batch_Dice.item()

            batch_loss = loss_fn(batch_results,mask)
            #print(batch_loss)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
            # del batch_results, batch_loss, batch_Dice,img,mask
            # torch.cuda.empty_cache()

        #epoch train loss
        epoch_loss.append(running_loss/len(dl_train))
        epoch_dice.append(running_dice/len(dl_train))
        print(f'epoch {i+1}, train dice:{epoch_dice[-1]:>.3f}, train loss:{epoch_loss[-1]:>.3f}')
        
        #epoch validation loss
        val_loss,val_dice = pre.model_validation(model,dl_val,device,metric=metric_fn,loss_func=loss_fn)
        #print(test_loss)
        epoch_val_loss.append(val_loss)
        epoch_val_dice.append(val_dice)

        if (i+1)%4==0:
            np.save('train_loss',epoch_loss)
            np.save('train_dice',epoch_dice)
            np.save('test_loss',epoch_val_loss)
            np.save('test_dice',epoch_val_dice)
            np.save('lr_history',epoch_lr)

        # determine save model or not based on the loss value
        if epoch_val_dice[-1]>best_dice:

            if best_model_path:
                os.remove(best_model_path)

            save_path = f'segmentation_epoch{i+1}_lr0.001_per_patient_batch.pth'
            best_dice = epoch_val_dice[-1]

            print(f'save model at epoch {i+1}')
            model.eval()
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},save_path)
            
            best_model = copy.deepcopy(model)
            best_model_path = save_path
        
        if lr_scheduler!=None:
            lr_scheduler.step(val_dice)

        
    training_output = {'best_model':best_model,'train_dice':epoch_dice,'val_dice':epoch_val_dice,
                       'train_loss':epoch_loss,'val_loss':epoch_val_loss,
                       'lr_history':epoch_lr}
    
    return training_output

    
def main():
    args = parse_args()

    train_names,val_names,test_names = pre.get_train_val_test_names(args.data_path)
    ds_train = Mydataset(pathlib.Path(args.data_path),train_names)
    dl_train = DataLoader(ds_train)

    ds_val = Mydataset(args.data_path,val_names)
    dl_val = DataLoader(ds_val)

    # model initialization
    model = Unet()
    if args.use_checkpoint:
        if args.checkpoint_file==None:
            raise ValueError('No checkpoint file, please provided')
        checkpoint = torch.load(args.checkpoint_file,weights_only=True)
        model.load_state_dict(checkpoint['model'])

    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    if args.use_scheduler:
        print('Use ReduceLROnPlateau scheduler')
        lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=7,min_lr=0.000001)
    else:
        lr_scheduler=None
    
    # loss function
    loss_fn = Loss(args.loss_type,tversky_weight=0.7)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    training_output = model_training(model,optimizer,args.epoch,dl_train,dl_val,lr_scheduler,device,loss_fn,pre.dice_coef)

    with open('training_results.pkl','wb') as f:
        pickle.dump(training_output,f)
 
if __name__ == '__main__':
    main()
    print('Training finished')