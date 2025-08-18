############### Unet segmentation for brain tumor ############
#%%
# basic package
import numpy as np
import pickle
import tqdm 

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)
import os

# ML package
import torch # basic tensor operation
from torch.utils.data import Dataset,DataLoader # build dataset and dataloader
import torch.nn as nn # build model
import torch.nn.functional as F # all functionals
import torch.optim as optim # optimizer
import torchvision.transforms as transform
import torch.nn.init as init
from sklearn.model_selection import train_test_split


# MRI imaging
import nibabel as nib
import nilearn as nl
import nilearn.plotting as nltplot
import copy
import cv2
import albumentations as A

#%%
def show_example_volume(folderpath,slice_num = None,effect_drawing = False):
    for volume in os.listdir(folderpath):
        if volume.endswith('_flair.nii'):
            flair = nib.load(os.path.join(folderpath,volume))
            flair_array = flair.get_fdata()
            continue

        if volume.endswith('_t1.nii'):
            t1 = nib.load(os.path.join(folderpath,volume))
            t1_array = t1.get_fdata()
            continue
        if volume.endswith('_t1ce.nii'):
            t1ce = nib.load(os.path.join(folderpath,volume))
            t1ce_array = t1ce.get_fdata()
            continue
        if volume.endswith('_t2.nii'):
            t2 = nib.load(os.path.join(folderpath,volume))
            t2_array = t2.get_fdata()
            continue
        if volume.endswith('_seg.nii'):
            seg = nib.load(os.path.join(folderpath,volume))
            seg_array = seg.get_fdata()
            continue

    print(seg_array.shape)
    print(seg_array.dtype)
    
    # visualization top-view --> slice across axial direction
    if slice_num:
        fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5,figsize=(20,10))
        im1=ax1.imshow(flair_array[:,:,slice_num],cmap='gray')
        fig.colorbar(im1,ax=ax1)
        ax1.set_title('flair')

        im2=ax2.imshow(t1_array[:,:,slice_num],cmap='gray')
        fig.colorbar(im2,ax=ax2)
        ax2.set_title('t1')

        im3=ax3.imshow(t1ce_array[:,:,slice_num],cmap='gray')
        fig.colorbar(im3,ax=ax3)
        ax3.set_title('t1ce')

        im4=ax4.imshow(t2_array[:,:,slice_num],cmap='gray')
        fig.colorbar(im4,ax=ax4)
        ax4.set_title('t2')

        im5=ax5.imshow(seg_array[:,:,slice_num])
        fig.colorbar(im5,ax=ax5)
        ax5.set_title('Mask')

        fig.tight_layout()
        fig.show()
        
    # if effect_drawing:
    #     fig2,axes = plt.subplots(4,figsize = (40,30))
    #     nltplot.plot_anat(flair,title = 'anatomical',axes = axes[0],colorbar=False)
    #     nltplot.plot_epi(flair,title = 'epi',axes = axes[1],colorbar=False,cmap = 'plasma',black_bg=True)
    #     nltplot.plot_img(flair,title = 'img',axes = axes[2],colorbar=False)
    #     nltplot.plot_roi(seg,title = 'mask',axes = axes[3],bg_img = flair,colorbar=False)

    

    return flair_array,t1_array,t1ce_array,t2_array,seg_array

def dice_coef_necrotic(y_pred,y_true,epsilon=1e-6):

    y_prob = F.softmax(y_pred,dim=1).permute(0,2,3,1)   # B H W C
    y_prob_f = torch.flatten(y_prob[:,:,:,1])

    y_true_onehot = F.one_hot(y_true,num_classes = y_pred.shape[1])
    y_true_f = torch.flatten(y_true_onehot[:,:,:,1])

    intersection = torch.sum(y_prob_f*y_true_f)+epsilon
    union = torch.sum(y_prob_f)+torch.sum(y_true_f)+epsilon
    dice = 2*intersection/union
    return dice 


def dice_coef_edema(y_pred,y_true,epsilon=1e-6):

    y_prob = F.softmax(y_pred,dim=1).permute(0,2,3,1)
    y_prob_f = torch.flatten(y_prob[:,:,:,2])

    y_true_onehot = F.one_hot(y_true,num_classes = y_pred.shape[1])
    y_true_f = torch.flatten(y_true_onehot[:,:,:,2])

    intersection = torch.sum(y_prob_f*y_true_f)+epsilon
    union = torch.sum(y_prob_f)+torch.sum(y_true_f)+epsilon
    dice = 2*intersection/union
    return dice 

def dice_coef_enchancing(y_pred,y_true,epsilon=1e-6):

    y_prob = F.softmax(y_pred,dim=1).permute(0,2,3,1)
    y_prob_f = torch.flatten(y_prob[:,:,:,3])

    y_true_onehot = F.one_hot(y_true,num_classes = y_pred.shape[1])
    y_true_f = torch.flatten(y_true_onehot[:,:,:,3])

    intersection = torch.sum(y_prob_f*y_true_f)+epsilon
    union = torch.sum(y_prob_f)+torch.sum(y_true_f)+epsilon
    dice = 2*intersection/union
    return dice 

def mask_clean(seg_array):

    """
    change the class value 4 to class value 3
    
    Parameters
    ---
        seg_arrary: np.array
            Raw segmentation array read from _seg.nii file

    Return
    ---
        cleaned: np.array 
            cleaned segmentation, where 4 is replaces by 3

    """
    cleaned = seg_array.copy()
    cleaned[cleaned==4]=3
    return cleaned


def split(volume_saving_path,ratio):
    volume_namelist = os.listdir(volume_saving_path)
    train_names,test_names = train_test_split(volume_namelist,test_size=ratio,random_state=42,shuffle=True)

    return train_names, test_names

def get_file(path,suffix):
    """
    Get the file path that ends with suffix, used to select _flair.nii or _t1ce.nii file

    Parameters
    ---
    path: os.path
        full path to each MRI scan folder
    suffix: str
        target file suffix

    Return
    ---
    Full path to the file with suffix, open directly with nib.load()
    
    """
    for file in os.listdir(path): 
        if file.endswith(suffix):
            return os.path.join(path,file) 

def compute_global_mean_std(volume_saving_path,volume_folder_list,slice_start,slice_end):
    
    flair_sum = 0
    t1ce_sum = 0

    flair_square_sum=0
    t1ce_square_sum=0
    voxel = 0

    for volume_folder in volume_folder_list:
        fullpath = os.path.join(volume_saving_path,volume_folder)
        flair_file = get_file(fullpath,'_flair.nii')
        t1ce_file = get_file(fullpath,'_t1ce.nii')

        flair_array = nib.load(flair_file).get_fdata()
        t1ce_array = nib.load(t1ce_file).get_fdata()

        flair_slices = flair_array[:,:,slice_start:slice_end+1]
        t1ce_slices = t1ce_array[:,:,slice_start:slice_end+1]

        flair_sum += np.sum(flair_slices)
        t1ce_sum += np.sum(t1ce_slices)

        flair_square_sum += np.sum(np.square(flair_slices))
        t1ce_square_sum += np.sum(np.square(t1ce_slices))

        voxel += flair_slices.size

    
    flair_mean = flair_sum/voxel
    t1ce_mean = t1ce_sum/voxel

    flair_var = flair_square_sum/voxel-np.square(flair_mean)
    t1ce_var = t1ce_square_sum/voxel-np.square(t1ce_mean)

    return flair_mean, t1ce_mean, np.sqrt(flair_var),np.sqrt(t1ce_var) 
       
t_transform = transform.Compose([

    transform.Resize((160,160))
])


aug = A.Compose([A.RandomGamma(),
                         A.ElasticTransform()])

def compute_global_mean_std_afterResize(volume_saving_path,volume_folder_list,slice_start,slice_end,resize_transform):
    
    flair_sum = 0
    t1ce_sum = 0

    flair_square_sum=0
    t1ce_square_sum=0
    voxel = 0

    for volume_folder in volume_folder_list:
        fullpath = os.path.join(volume_saving_path,volume_folder)

        # read flair and t1ce files
        flair_file = get_file(fullpath,'_flair.nii')
        t1ce_file = get_file(fullpath,'_t1ce.nii')

        # get flair and t1ce array
        flair_array = nib.load(flair_file).get_fdata()
        t1ce_array = nib.load(t1ce_file).get_fdata()

        # get flair and t1ce slices, convert to tensor, transpose; ready for resize
        flair_slices = torch.from_numpy(flair_array[:,:,slice_start:slice_end+1].transpose([2,0,1]))
        t1ce_slices = torch.from_numpy(t1ce_array[:,:,slice_start:slice_end+1].transpose([2,0,1]))

        #resized, torch
        flair_slices = resize_transform(flair_slices) 
        t1ce_slices = resize_transform(t1ce_slices) 

        flair_sum += torch.sum(flair_slices)
        t1ce_sum += torch.sum(t1ce_slices)

        flair_square_sum += torch.sum(torch.square(flair_slices))
        t1ce_square_sum += torch.sum(torch.square(t1ce_slices))

        voxel += flair_slices.numel()

    
    flair_mean = flair_sum/voxel
    t1ce_mean = t1ce_sum/voxel

    flair_var = flair_square_sum/voxel-np.square(flair_mean)
    t1ce_var = t1ce_square_sum/voxel-np.square(t1ce_mean)

    return flair_mean, t1ce_mean, np.sqrt(flair_var),np.sqrt(t1ce_var) 

class Mydataset(Dataset):
    def __init__(self,volume_savingpath,volume_namelist,slice_start=22,slice_num = 100,transforms=None,augmentation=None):

        self.path = volume_savingpath
        self.name_list = volume_namelist
        self.slice_start = slice_start
        self.slice_num = slice_num

        self.transform = transforms
        self.augmentation = augmentation


        
    def __len__(self):
        return len(self.name_list)
    
    # get image and mask at corresponding index
    def __getitem__(self,index):

        full_path = os.path.join(self.path,self.name_list[index])

        seg_file = get_file(full_path,'_seg.nii')
        seg_array = mask_clean(nib.load(seg_file).get_fdata()) #240,240,155


        t1ce_file = get_file(full_path,'_t1ce.nii')
        t1ce_array = nib.load(t1ce_file).get_fdata(dtype=np.float32)  #240,240,155


        flair_file = get_file(full_path,'_flair.nii')
        flair_array = nib.load(flair_file).get_fdata(dtype=np.float32) #240,240,155
    


        X = torch.zeros((self.slice_num,2,160,160))
        Y = torch.zeros((self.slice_num,160,160))

        for j in range(self.slice_num): # slice num=100

            t1ce = cv2.resize(t1ce_array[:,:,j+self.slice_start],(160,160)) #160, 160
            flair = cv2.resize(flair_array[:,:,j+self.slice_start],(160,160)) #160, 160
            modality = np.stack((t1ce,flair),axis=-1) # 160,160,2

            mask = torch.from_numpy(seg_array[:,:,j+self.slice_start]) 

            if self.transform:
                mask.unsqueeze_(0)
                mask = self.transform(mask)
                mask.squeeze_()
            #mask: 160, 160
            mask = mask.numpy() #160*160

            if self.augmentation:
                augmented = self.augmentation(image=modality,mask=mask)
                modality_aug = augmented['image'].transpose((2,0,1))
                mask_aug = augmented['mask']
            else:
                modality_aug = modality.transpose((2,0,1))
                mask_aug = mask

            X[j,:,:,:] = torch.from_numpy(modality_aug) #100,2,160,160
            Y[j,:,:] = torch.from_numpy(mask_aug)     #100,160,160

        X_mean = X.mean(dim=[0,2,3],keepdim=True)
        X_std = X.std(dim=[0,2,3],keepdim=True)

        X_standard = (X-X_mean)/(X_std+1e-7)


        #X (100,2,H,W): slices of the current patient, 
        #Y: segmentation of the current patient

        return X_standard, Y.long()   

    
class Encoder(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        # self.input_c = in_c
        # self.output_c = out_c
        self.conv1 = nn.Conv2d(in_c,out_c,3,padding=1)
        # self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c,out_c,3,padding=1)
        # self.bn2 = nn.BatchNorm2d(out_c)

        self.pool = nn.MaxPool2d(2,2)
        self.activation = nn.ReLU()
        self.dp = nn.Dropout2d(p=0.2)

    def forward(self,x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.activation(x)
        x = self.dp(x)

        return x,self.pool(x)
        
class Decoder(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_c,out_c,2,2)
        self.conv1 = nn.Conv2d(2*out_c,out_c,3,padding=1)
        # self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c,out_c,3,padding=1)
        # self.bn2 = nn.BatchNorm2d(out_c)

        self.activation = nn.ReLU()
        self.dp = nn.Dropout2d(p=0.2)

    def forward(self,x,skip):
        x = self.upconv(x)
        x = torch.cat((skip,x),dim=1)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.activation(x)
        x = self.dp(x)

        return x
    
class Unet(nn.Module):
    def __init__(self,encoder1,encoder2,encoder3,encoder4,encoder5,
                 decoder1,decoder2,decoder3,decoder4):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3
        self.encoder4 = encoder4
        self.encoder5 = encoder5
        
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.decoder3 = decoder3
        self.decoder4 = decoder4
    
        self.conv = nn.Conv2d(32,4,1)

    def forward(self,x):
        skip1,pool1 = self.encoder1(x)
        skip2,pool2 = self.encoder2(pool1)
        skip3,pool3 = self.encoder3(pool2)
        skip4,pool4 = self.encoder4(pool3)
        skip5,_ = self.encoder5(pool4)
        
        skip5 = nn.Dropout2d(p=0.4)(skip5)

        upconv1 = self.decoder1(skip5,skip4)
        upconv2 = self.decoder2(upconv1,skip3)
        upconv3 = self.decoder3(upconv2,skip2)
        upconv4 = self.decoder4(upconv3,skip1)
        
        logit = self.conv(upconv4)

        return logit
    
def dice_coef(y_pred,y_true,smooth=1e-7):
    """
    calculate the batch dice coefficient

    Parameters
    ---
        y_pred: tensor
            raw model output in logits, with dimension N*C*H*W
        y_true: tensor
            ground truth for segmentation (not one-hot format), with dimension N*H*W, already in long()
        smooth: float
            numer to prevent singularity
    Return
    ---
        dice coefficient of the current batch
    """

    class_num = y_pred.shape[1] #class_num=4 

    y_prob = F.softmax(y_pred,dim=1).permute(0,2,3,1) #N H W C batch probablities
    y_true_onehot = F.one_hot(y_true,num_classes = class_num).float()#N H W C

    inter = (y_prob * y_true_onehot).sum(dim=(0,1,2)) # C

    predsum =  y_prob.sum(dim=(0,1,2))        #[C]
    gtsum = y_true_onehot.sum(dim=(0,1,2))    #[C]

    dice = (2*inter+smooth)/(predsum+gtsum+smooth)   #[C] dice per class
    dice = dice[1:]   
    
    present = gtsum[1:]>0
    dice = dice[present] if present.any() else dice


    return dice.mean()

    # for i in range(class_num):
    #     pred_flatten = y_prob[:,:,:,i].flatten()
    #     gt_flatten = y_true_onehot[:,:,:,i].flatten()
        
    #     intersection = torch.sum(pred_flatten*gt_flatten)
    #     union = torch.sum(pred_flatten)+torch.sum(gt_flatten)

    #     batch_dice += (2*intersection)/(union+smooth)

    # return batch_dice/class_num

class Loss(nn.Module):
    def __init__(self,mode,smooth=1e-7,dice_weight=0.5,alpha = 0.3,beta = 0.7, gamma = 1.2,tversky_weight=0.7):
        super().__init__()
        self.mode = mode
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma=gamma
        self.tversky_weight = tversky_weight

    def forward(self,y_pred,gt):

        class_num = y_pred.shape[1]
        y_prob = F.softmax(y_pred,dim=1) #NCHW
        y_true_onehot = F.one_hot(gt,num_classes = class_num).float() #NHWC

        # dice loss
        intersection = torch.sum(y_prob*y_true_onehot.permute(0,3,1,2),dim=(2,3))
        union = torch.sum(y_prob,dim=(2,3))+torch.sum(y_true_onehot.permute(0,3,1,2),dim=(2,3))+self.smooth
        dice = (2*intersection+self.smooth)/union
        dice_loss = 1-dice.mean()

        # cce loss
        y_prob_cce = torch.clamp(y_prob.permute(0,2,3,1),min=1e-7,max=1)
        CCE_loss = (-1*torch.sum(torch.log(y_prob_cce)*y_true_onehot,dim=-1)).mean()

        # fkl loss 
        y_prob_fkl = y_prob.clamp_min(1e-8) #NCHW
        gt_onehot = y_true_onehot.permute(0,3,1,2).float() #NCHW, filter out positive pixel

        tp = torch.sum(y_prob_fkl*gt_onehot,dim=(0,2,3)) # summing probablities at tp pixel
        fp = torch.sum(y_prob_fkl*(1-gt_onehot),dim=(0,2,3)) # summing probablities at fp pixel
        fn = torch.sum((1-y_prob_fkl)*gt_onehot,dim=(0,2,3))# summing porbablities at fn 
        tversky = (tp+self.smooth)/(tp+self.alpha*fp+self.beta*fn+self.smooth)
        focal_tversky_loss = (1 - tversky)**self.gamma  #teversky loss per channel
        ftl = focal_tversky_loss[1:].mean()

        if self.mode =='dice':
            return dice_loss
        elif self.mode =='dice+cce':
            return self.dice_weight*dice_loss+(1-self.dice_weight)*CCE_loss
        elif self.mode =='cce':
            return CCE_loss
        elif self.mode == 'ftl+cce':
            return self.tversky_weight*ftl+(1-self.tversky_weight)*CCE_loss
        else:
            raise ValueError(f"Invalid mode {self.mode}, choose from ['dice','dice+cce','cce',''ftl+cce]")

def model_validation(model,dl_val,device,metric,loss_func):

    model.eval()
    running_dice=0
    running_loss=0

    for batch in dl_val:
            
        img = batch[0][0].to(device) #100 2 160 160
        mask = batch[1][0].to(device) #100 160 160

        with torch.no_grad():

            batch_results = model(img)

            batch_Dice = metric(batch_results,mask)
            running_dice+=batch_Dice.item()

            batch_loss = loss_func(batch_results,mask)
            running_loss += batch_loss.item()

    final_loss = running_loss/len(dl_val)
    final_dice = running_dice/len(dl_val)
    print(f'validation dice:{final_dice:>.3f}, validation loss:{final_loss:>.3f}\n')

    return final_loss, final_dice


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
        val_loss,val_dice = model_validation(model,dl_val,device,metric=metric_fn,loss_func=loss_fn)
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
        
        if lr_scheduler:
            lr_scheduler.step(val_dice)
        
    

        
    training_output = {'best_model':best_model,'train_dice':epoch_dice,'val_dice':epoch_val_dice,
                       'train_loss':epoch_loss,'val_loss':epoch_val_loss,
                       'lr_history':epoch_lr}
    
    return training_output

def show_sample(ds,index):
    image,mask = ds.__getitem__(index)
    fig,axes = plt.subplots(1,3)
    im1=axes[0].imshow(image[0,:,:])
    axes[0].set_title('t1ce')
    fig.colorbar(im1,ax=axes[0])

    im2=axes[1].imshow(image[1,:,:])
    axes[1].set_title('flair')
    fig.colorbar(im2,ax=axes[1])

    im3=axes[2].imshow(mask)
    axes[2].set_title('mask')
    fig.colorbar(im3,ax=axes[2])

    fig.tight_layout()
    fig.show()

    return image, mask

def check_mean_std(volume_saving_path,volume_folder_list,slice_start,slice_end,
                   flair_mean,flair_std,t1ce_mean,t1ce_std):
    flair_sum = 0
    t1ce_sum = 0

    flair_square_sum=0
    t1ce_square_sum=0
    voxel = 0   

    for volume_folder in volume_folder_list:
        fullpath = os.path.join(volume_saving_path,volume_folder)
        flair_file = get_file(fullpath,'_flair.nii')
        t1ce_file = get_file(fullpath,'_t1ce.nii')

        flair_array = nib.load(flair_file).get_fdata()
        t1ce_array = nib.load(t1ce_file).get_fdata()

        flair_slices = (flair_array[:,:,slice_start:slice_end+1]-flair_mean)/flair_std
        t1ce_slices = (t1ce_array[:,:,slice_start:slice_end+1]-t1ce_mean)/t1ce_std

        flair_sum += np.sum(flair_slices)
        t1ce_sum += np.sum(t1ce_slices)

        flair_square_sum += np.sum(np.square(flair_slices))
        t1ce_square_sum += np.sum(np.square(t1ce_slices))

        voxel += flair_slices.size

    
    cal_flair_mean = flair_sum/voxel
    cal_t1ce_mean = t1ce_sum/voxel

    cal_flair_var = flair_square_sum/voxel-np.square(cal_flair_mean)
    cal_t1ce_var = t1ce_square_sum/voxel-np.square(cal_t1ce_mean)

    return cal_flair_mean, cal_t1ce_mean,cal_flair_var,cal_t1ce_var

def model_check():
    x = torch.randn(1, 2, 160, 160)
    with torch.no_grad():
        y = model(x)
    
    if y.shape == torch.Size([1,4,160,160]):
        print(f"y shape is {y.shape}, safe to go")  # should be [1, 4, 160, 160]
    

def apply_he_normal(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
#%%
# check the division of Train, validation and test dataset
pwd = os.getcwd()
actual_path = "c:\\python_script\\CNN\\pytorch_turtorial\\kaggle\\BraTS2020\\BraTS2020_TrainingData"
volume_saving_path = os.path.join(actual_path,'MICCAI_BraTS2020_TrainingData')
train_test_names,val_names = split(volume_saving_path,ratio=0.2)
train_names,test_names = train_test_split(train_test_names,test_size=0.15,random_state=42)
np.save('test_names',test_names)
plt.bar(['Train','validation','test'],
        [len(train_names),len(val_names),len(test_names)],
        color=['green','red','blue'],
        align='center')
plt.ylabel('Patient Volume')
plt.title('Data distribution')
#%%
# Build dataset and dataloader for training and validation folder
ds_train = Mydataset(volume_saving_path,train_names,transforms=t_transform,augmentation=aug)
dl_train = DataLoader(ds_train,batch_size=1,shuffle=True)

ds_val = Mydataset(volume_saving_path,val_names,transforms=t_transform)
dl_val = DataLoader(ds_val,batch_size=1,shuffle=False)

print(f'lenght of ds_train: {len(ds_train)}')
print(f'lenght of dl_train: {len(dl_train)}')
print(ds_train[0][0].shape)
print(ds_train[0][1].shape)

print(f'lenght of ds_val: {len(ds_val)}')
print(f'lenght of dl_val: {len(dl_val)}')
print(ds_val[0][0].shape)
print(ds_val[0][1].shape)
#%%
encoder1 = Encoder(2,32)
encoder2 = Encoder(32,64)
encoder3 = Encoder(64,128)
encoder4 = Encoder(128,256)
encoder5 = Encoder(256,512)

decoder1 = Decoder(512,256)
decoder2 = Decoder(256,128)
decoder3 = Decoder(128,64)
decoder4 = Decoder(64,32)

model = Unet(encoder1,encoder2,encoder3,encoder4,encoder5,
             decoder1,decoder2,decoder3,decoder4)

checkpoint = torch.load('checkpoint2_segmentation_epoch30_lr0.001_per_patient_batch.pth',weights_only=True)
model.load_state_dict(checkpoint['model'])
model_check()

optimizer = optim.Adam(model.parameters(),lr=0.001)
loss_fn = Loss(mode='ftl+cce',dice_weight=0.7,tversky_weight=0.7)
epoch = 300
lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=7,min_lr=0.000001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
training_output = model_training(model,optimizer,epoch,dl_train,dl_val,lr_scheduler,device,loss_fn,dice_coef)
with open("training_log.pkl",'wb') as f:
    pickle.dump(training_output,f) 
#%%
train_dice = np.load('train_dice.npy')
val_dice = np.load('test_dice.npy')

plt.plot(train_dice,label='training')
plt.plot(val_dice,label='validation')
plt.legend()
#%% Model validation with different dice

checkpoint = torch.load('segmentation_epoch106_lr0.001_per_patient_batch.pth',weights_only=True)
model = Unet(encoder1,encoder2,encoder3,encoder4,encoder5,decoder1,decoder2,decoder3,decoder4)
model.load_state_dict(checkpoint['model'])
model.to(device)

model_validation(model,dl_val,device,dice_coef_edema,loss_fn)
















#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# def check_abnormal_value(volume_saving_path):
#     for entry in os.listdir(volume_saving_path):
#         full_path = os.path.join(volume_saving_path,entry)
#         flair = nib.load(get_file(full_path,'_flair.nii')).get_fdata()
#         t1ce = nib.load(get_file(full_path,'_t1ce.nii')).get_fdata()

#         for slice in range(22,121):
#             if np.max(flair[:,:,slice])==0:
#                 print(f"for {entry}, flair slice {slice} is abnormal")
#             if np.max(t1ce[:,:,slice])==0:
#                 print(f"for {entry}, t1ce slice {slice} is abnormal")

# check_abnormal_value(volume_saving_path)
# t_transform = transform.Compose([

#     transform.Resize((192,192))
# ])

# def show_ds_sample(names,index,resize):
#     ds = Mydataset(volume_saving_path,names,transforms=resize)
#     image,mask = ds[index]
#     fig,axes = plt.subplots(1,3)
#     im1 = axes[0].imshow(image[0,:,:])
#     axes[0].set_title('t1ce')
#     im2 = axes[1].imshow(image[1,:,:])
#     axes[1].set_title('flair')
#     im3 = axes[2].imshow(mask)
#     axes[2].set_title('mask')

#     fig.colorbar(im1,ax=axes[0])
#     fig.colorbar(im2,ax=axes[1])

#     fig.colorbar(im3,ax=axes[2])

#     fig.tight_layout()

#     return image[0,:,:],image[1,:,:],mask

# t1ce_ori,flair_ori,mask_ori = show_ds_sample(test_names,150,t_transform)

#%%

t_transform = transform.Compose([

    transform.Resize((160,160))
])

def model_test(test_names,model,patient_ID,slice_ID,resize):

    ds_test = Mydataset(volume_saving_path,test_names,transforms=resize)
    model.to('cpu')
    model.eval()

    batch = ds_test[patient_ID]
    image = batch[0][slice_ID]
    mask = batch[1][slice_ID]
    image.unsqueeze_(0)
    results = model(image)
    probs = F.softmax(results,dim=1)
    preds = torch.argmax(probs,dim=1)

    acc = (torch.sum(preds[0,:,:]==mask))/torch.numel(mask)

    fig,axes = plt.subplots(2,2,figsize=(8,8))
    axes[0,0].imshow(image[0,0,:,:])
    axes[0,0].set_title('t1ce',size=15)
    axes[0,0].axis('off')

    axes[0,1].imshow(image[0,1,:,:])
    axes[0,1].set_title('flair',size=15)
    axes[0,1].axis('off')

    im2=axes[1,0].imshow(mask)
    axes[1,0].set_title('Segmentation',size=15)
    axes[1,0].axis('off')
    cbar2 = fig.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.04)

    im3 = axes[1,1].imshow(preds[0,:,:])
    axes[1,1].set_title('Prediction',size=15)
    axes[1,1].axis('off')
    cbar3 = fig.colorbar(im3, ax=axes[1,1], fraction=0.046, pad=0.04)

    # fig.colorbar(im2,ax=axes[2])
    # fig.colorbar(im3,ax=axes[3])
    fig.show()
    fig.tight_layout()
    print(f"accuracy is {acc}")

    return image[0,0,:,:],image[0,1,:,:],mask,preds[0,:,:]


checkpoint = torch.load('segmentation_epoch106_lr0.001_per_patient_batch.pth',weights_only=True)
model = Unet(encoder1,encoder2,encoder3,encoder4,encoder5,decoder1,decoder2,decoder3,decoder4)
model.load_state_dict(checkpoint['model'])
t1ce,flair,mask,preds=model_test(test_names,model,3,50,t_transform)


























# %%
