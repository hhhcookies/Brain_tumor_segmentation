########################## functions for processing ###############################

import os 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from get_data import Mydataset

plt.ion

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
        
def get_train_val_test_names(scan_saving_path):
    train_test_names,val_names = split(scan_saving_path,ratio=0.2)
    train_names,test_names = train_test_split(train_test_names,test_size=0.15,random_state=42)
    plt.bar(['Train','validation','test'],
        [len(train_names),len(val_names),len(test_names)],
        color=['green','red','blue'],
        align='center')
    plt.ylabel('Patient Volume')
    plt.title('Data distribution')
    plt.show()

    return train_names,val_names,test_names

def model_validation(model,dl_val,device,metric,loss_func):

    model.eval()
    running_dice=0
    running_loss=0

    for batch in dl_val:
            
        img = batch[0].to(device) # N 2 H W
        mask = batch[1].to(device) # N H W

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

    y_prob = torch.argmax(F.softmax(y_pred,dim=1),dim=1) # N,H,W
    y_prob = F.one_hot(y_prob,num_classes=class_num).float() #N H W C
    y_true_onehot = F.one_hot(y_true,num_classes = class_num).float()#N H W C

    inter = (y_prob * y_true_onehot).sum(dim=(0,1,2)) # [C]

    predsum =  y_prob.sum(dim=(0,1,2))        #[C]
    gtsum = y_true_onehot.sum(dim=(0,1,2))    #[C]

    dice = (2*inter+smooth)/(predsum+gtsum+smooth)   #[C] dice per class
    dice = dice[1:]   #[C-1], bg excluded
    
    present = gtsum[1:]>0 #[C-1],bg excluded
    dice = dice[present] if present.any() else dice

    return dice.mean()

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

def model_inference(folder_path,folder_name,model,slice_ID,patient_ID=0):
    namelist = [folder_name]
    ds_test = Mydataset(folder_path,namelist)
    model.to('cpu')
    model.eval()

    batch = ds_test[patient_ID]
    image = batch[0][slice_ID]
    mask = batch[1][slice_ID]
    image.unsqueeze_(0)
    mask_un = mask.unsqueeze(0)
    results = model(image)
    probs = F.softmax(results,dim=1)
    preds = torch.argmax(probs,dim=1)

    dice = dice_coef(results,mask_un)
    dice_necrotic = dice_coef_necrotic(results,mask_un)
    dice_edema = dice_coef_edema(results,mask_un)
    dice_enchancing = dice_coef_enchancing(results,mask_un)

    fig,axes = plt.subplots(2,2,figsize=(8,8))
    axes[0,0].imshow(image[0,0,:,:])
    axes[0,0].set_title(f't1ce slice{slice_ID}',size=15)
    axes[0,0].axis('off')

    axes[0,1].imshow(image[0,1,:,:])
    axes[0,1].set_title(f'flair slice{slice_ID}',size=15)
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
    fig.tight_layout()

    print(f"average dice is {dice}")
    print(f"necrotic dice is {dice_necrotic}")
    print(f"edema dice is {dice_edema}")
    print(f"enchancing dice is {dice_enchancing}")

    plt.show()

    return image[0,0,:,:],image[0,1,:,:],mask,preds[0,:,:]