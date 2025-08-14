########################## functions for processing ###############################

import os 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%%
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