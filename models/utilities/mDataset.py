import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .trainer import Trainer

class mDataset(Dataset):
    def __init__(self, data_path=None, data_idx=None):
        assert(data_path is not None)
        data = torch.load(data_path, map_location='cpu')
        if data_idx is None:
            self.data = data
        else:
            self.data = (data[0][data_idx], data[1][data_idx])
        self.features = None
        self.length = len(self.data[1])
        if len(self.data[0][0].shape)==2:
            self.data[0].unsqueeze_(1)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        instances  = self.data[0][index].float()
        labels = self.data[1][index]
        return instances, labels
    
    def updateFeatures(self, model, feature_layer,vit=True):
        data_loader = DataLoader(self, batch_size=64, shuffle=False)
        model.cuda()
        model.eval()
        
        self.feature_list = []
        
        with torch.no_grad():
            for instances, _ in data_loader:
                instances = instances.cuda()
                batch_features = model.instance2feature(instances, feature_layer)
                self.feature_list.append(batch_features.cpu())
            
        self.features = torch.vstack(self.feature_list)
        del self.feature_list
    
    def updateLabels(self, model):
        ''' data: instances, pred_labels '''
        ''' features, true_labels는 별도 저장'''
        _, pred_labels, _ = Trainer.evalAccuracyOfModel(model, self)
        self.true_labels = self.data[1].clone()
        self.data = (self.data[0], torch.tensor(pred_labels))
    
    def getCNNFeatures(self, indices):
        if self.features is None:
            print('Do updateFeatures...')
            return None
        else:
            features = self.features[indices].float()
            if len(features.shape)==3:
                features.unsqueeze_(0)
            return features
        
    def getInstance(self, indices):
        instances = self.data[0][indices].float()
        if len(instances.shape)==3:
            instances.unsqueeze_(0)
        return instances
        
    
    def getTensorLabels(self):
        return self.data[1].long()
    
    