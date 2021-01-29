from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class CIFARDataset(Dataset):
    
    def __init__(self, cifar_obj, meta_dict, label_to_idx, transform):
        self.cifar_obj = cifar_obj
        self.meta_dict = meta_dict
        self.label_to_idx = label_to_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.cifar_obj['labels'])
    
    def __getitem__(self, idx):

        img = self.cifar_obj['data'][idx]
        #img = np.transpose(img, [0, 1, 2])
        
        img = self.transform(img)

        cifar_label = self.cifar_obj['labels'][idx]
        label_name = self.meta_dict['label_names'][cifar_label]

        label_idx = self.label_to_idx[label_name]

        sample = {"image": img, "label": label_idx}
        
        return sample
