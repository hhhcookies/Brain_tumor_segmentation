######################### get data #####################
import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset,DataLoader # build dataset and dataloader

import preprocess as pre

import nibabel as nib
import os
import numpy as np

class Mydataset(Dataset):
    def __init__(self,volume_savingpath,volume_namelist,mean_list,std_list,
                 slice_start=22,slice_num = 100,transforms=None,augmentation=None):

        self.path = volume_savingpath
        self.name_list = volume_namelist
        self.slice_start = slice_start
        self.slice_num = slice_num
        self.transform = transforms
        self.augmentation = augmentation
        self.mean_list = mean_list
        self.std_list = std_list

        self._cache={}    # cache volume 

    def __len__(self):
        return len(self.name_list)*self.slice_num
    

    def _load_patient(self,pid):

        if pid in self._cache:
            return self._cache[pid]
        else:
            full_path = os.path.join(self.path,self.name_list[pid])

            seg_file = pre.get_file(full_path,'_seg.nii')
            seg_array =  nib.load(seg_file,mmap=True)
            #seg_array = mask_clean(nib.load(seg_file).get_fdata()).astype(np.uint8) #240,240,155

            t1ce_file = pre.get_file(full_path,'_t1ce.nii')
            t1ce_array = nib.load(t1ce_file,mmap=True)
            #t1ce_array = nib.load(t1ce_file).get_fdata(dtype=np.float32)  #240,240,155

            flair_file = pre.get_file(full_path,'_flair.nii')
            flair_array = nib.load(flair_file,mmap=True)
            #flair_array = nib.load(flair_file).get_fdata(dtype=np.float32) #240,240,155

            self._cache[pid] = (seg_array,t1ce_array,flair_array)

            return (seg_array,t1ce_array,flair_array)

    
    # get image and mask at corresponding index
    def __getitem__(self,index):
        # index is the slice index
        patient = index//self.slice_num # patient the slice belong to
        slice_idx = (index%self.slice_num) + self.slice_start

        seg_array,t1ce_array,flair_array = self._load_patient(patient)

        mask = np.asarray(seg_array.dataobj[:,:,slice_idx])
        t1ce_slice = np.asarray(t1ce_array.dataobj[:,:,slice_idx],dtype=np.float32)
        flair_slice = np.asarray(flair_array.dataobj[:,:,slice_idx],dtype=np.float32)

        mask = pre.mask_clean(mask).astype(np.uint8)

        # normalization
        eps = np.float32(1e-8)
        t1ce = (t1ce_slice-self.mean_list[patient,0]) / (self.std_list[patient,0]+eps) #240,240
        flair = (flair_slice-self.mean_list[patient,1]) / (self.std_list[patient,1]+eps) #240,240

        modality = np.stack((t1ce,flair),axis=-1).astype(np.float32) #240 240 2

        if self.augmentation:
            augmented = self.augmentation(image=modality,mask=mask)
            modality = augmented['image']
            mask = augmented['mask']

        modality = torch.from_numpy(modality).permute((2,0,1))

        return modality, torch.from_numpy(mask).long()