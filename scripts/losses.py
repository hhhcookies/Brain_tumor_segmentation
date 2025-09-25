################## Loss functions #####################
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        intersection = torch.sum(y_prob*y_true_onehot.permute(0,3,1,2),dim=(2,3))    #[N,C]
        union = torch.sum(y_prob,dim=(2,3))+torch.sum(y_true_onehot.permute(0,3,1,2),dim=(2,3))+self.smooth #[N,C]
        dice = (2*intersection+self.smooth)/union #[N,C]
        dice_loss = 1-dice.mean()

        # cce loss
        # y_prob_cce = torch.clamp(y_prob.permute(0,2,3,1),min=1e-7,max=1)
        CCE_loss = F.cross_entropy(y_pred,gt)

        # fkl loss 
        y_prob_fkl = y_prob.clamp_min(1e-8) #NCHW
        gt_onehot = y_true_onehot.permute(0,3,1,2).float() #NCHW, filter out positive pixel B C H W

        tp = torch.sum(y_prob_fkl*gt_onehot,dim=(0,2,3)) # summing probablities at tp pixel [C]
        fp = torch.sum(y_prob_fkl*(1-gt_onehot),dim=(0,2,3)) # summing probablities at fp pixel[C]
        fn = torch.sum((1-y_prob_fkl)*gt_onehot,dim=(0,2,3))# summing porbablities at fn [C]
        tversky = (tp+self.smooth)/(tp+self.alpha*fp+self.beta*fn+self.smooth) #[C]
        ftl = (1 - tversky)**self.gamma  #teversky loss per channel #[C]
        ftl_tumor = ftl[1:]     #[C-1] bg excluded ftl loss

        presence = gt_onehot.sum(dim=(0,2,3))>0   #[C]
        tumor_presence = presence[1:] #[C-1], background excluded

        ftl = (ftl_tumor[tumor_presence]).mean() if tumor_presence.any() else ftl_tumor.mean() 

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