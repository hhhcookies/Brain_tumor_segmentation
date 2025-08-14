######################### get data #####################
import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset,DataLoader # build dataset and dataloader

import preprocess as pre

import nibabel as nib
import os
import cv2 

class Mydataset(Dataset):
    def __init__(self,volume_savingpath,volume_namelist,slice_start=22,slice_num = 100,resized_to=160):

        self.path = volume_savingpath
        self.name_list = volume_namelist

        self.slice_num = slice_num
        self.slice_start = slice_start

        self.resized_to = resized_to
        self.transform = transform.Compose([transform.Resize((resized_to,resized_to))])

    def __len__(self):
        return len(self.name_list)
    
    # get image and mask at corresponding index
    def __getitem__(self,index):

        full_path = os.path.join(self.path,self.name_list[index])

        seg_file = pre.get_file(full_path,'_seg.nii')
        seg_array = pre.mask_clean(nib.load(seg_file).get_fdata()) #240,240,155

        t1ce_file = pre.get_file(full_path,'_t1ce.nii')
        t1ce_array = nib.load(t1ce_file).get_fdata()  #240,240,155

        flair_file = pre.get_file(full_path,'_flair.nii')
        flair_array = nib.load(flair_file).get_fdata() #240,240,155

        X = torch.zeros((self.slice_num,2,self.resized_to,self.resized_to))
        Y = torch.zeros((self.slice_num,self.resized_to,self.resized_to))

        for j in range(self.slice_num): # slice num=100
            t1ce = torch.from_numpy(cv2.resize(t1ce_array[:,:,j+self.slice_start],(self.resized_to,self.resized_to)))
            flair = torch.from_numpy(cv2.resize(flair_array[:,:,j+self.slice_start],(self.resized_to,self.resized_to)))
            mask = torch.from_numpy(seg_array[:,:,j+self.slice_start])

            if self.transform:
                    
                mask.unsqueeze_(0)
                mask = self.transform(mask)
                mask.squeeze_()

            X[j,0,:,:] = t1ce
            X[j,1,:,:] = flair

            Y[j,:,:]=mask

        return X/(torch.max(X) + 1e-7), Y.long()