import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class DeepSAT(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, return_all=False, version='sat6'):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.return_all = return_all
        self.version = version
            
        if self.return_all:
            self.X_train = pd.read_csv(os.path.join(self.root, f"X_train_{self.version}.csv"), header=None)
            self.y_train = pd.read_csv(os.path.join(self.root, f"y_train_{self.version}.csv"), header=None)
            self.X_test = pd.read_csv(os.path.join(self.root, f"X_test_{self.version}.csv"), header=None)
            self.y_test = pd.read_csv(os.path.join(self.root, f"y_test_{self.version}.csv"), header=None)
            
            self.X_data = pd.concat([self.X_train, self.X_test])
            self.y_data = pd.concat([self.y_train, self.y_test])
        else:
            if self.train:
                self.X_data = pd.read_csv(os.path.join(self.root, f"X_train_{self.version}.csv"), header=None)
                self.y_data = pd.read_csv(os.path.join(self.root, f"y_train_{self.version}.csv"), header=None)
            else:
                self.X_data = pd.read_csv(os.path.join(self.root, f"X_test_{self.version}.csv"), header=None)
                self.y_data = pd.read_csv(os.path.join(self.root, f"y_test_{self.version}.csv"), header=None)

    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, idx):
        image = self.X_data.iloc[idx].values.reshape([28,28,4])[:,:,:3] ##reshape input data to rgb image
        label = self.img_labels.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    def _label_conv(label_arr):
        labels=[]
        for i in range(len(label_arr)):

            if (label_arr[i]==[1,0,0,0,0,0]).all():
                labels.append("Building")  

            elif (label_arr[i]==[0,1,0,0,0,0]).all():  
                labels.append("Barren_land")  

            elif (label_arr[i]==[0,0,1,0,0,0]).all():
                labels.append("Tree") 

            elif (label_arr[i]==[0,0,0,1,0,0]).all():
                labels.append("Grassland")

            elif (label_arr[i]==[0,0,0,0,1,0]).all():
                labels.append("Road") 

            else:
                labels.append("Water")
        return labels