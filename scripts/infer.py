###### scripts for inference ######
import numpy as np
import argparse
from model import Unet
import torch
from preprocess import model_inference as infer
import pathlib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--folder_path',type=str,required=True,help='saving path of MRI scan folder')
    p.add_argument('--folder_name',type=str,required=True,help='name of MRI scan folder')
    p.add_argument('--model_file',type=str,required=True,help='Model for the inference')
    p.add_argument('--slice_num',type=int,default=50,help='slice to show, choose a number between 0 and 99')

    return p.parse_args()

def main():
    args = parse_args()

    model = Unet()
    weights = torch.load(pathlib.Path(args.model_file),weights_only=True)
    model.load_state_dict(weights['model'])
    t1ce,flair,mask,preds=infer(args.folder_path,args.folder_name,model,args.slice_num)

    infer_results = np.stack((t1ce,flair,mask,preds))
    return infer_results

if __name__ == "__main__":
    main()
    print('inference finished')
