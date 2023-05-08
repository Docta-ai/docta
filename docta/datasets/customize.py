import torch
import numpy as np
class CustomizedDataset(torch.utils.data.Dataset):
    def __init__(self, feature, label = None, index = None, preprocess = None):
        self.feature = feature
        self.index = range(len(feature)) if index is None else index
        self.label = [None] * len(feature) if label is None else label
        self.preprocess = preprocess 
    def update(self, dataset_new):
        self.feature = np.concatenate((self.feature, dataset_new.feature), axis = 0)
        self.index = np.concatenate((self.index, dataset_new.index), axis = 0)
        self.label = np.concatenate((self.label, dataset_new.label), axis = 0)



    def __getitem__(self, index):
        idx = self.index[index]
        feature = self.feature[idx]
        
        if self.preprocess is not None:
            feature = self.preprocess(feature)
            if feature.shape[0] == 1: # output of tokenizer
                feature = feature.reshape(feature.shape[1:])

        return feature, self.label[idx], idx
    
    def __len__(self) -> int:
        return len(self.feature)
