import pickle
import time
import numpy as np
import pandas as pd

import torch 
import torch.nn as nn 
import torch.nn.functional as F # various activation functions for model

import torchvision # You can load various Pretrained Model from this package 
import torchvision.datasets as vision_dsets
import torchvision.transforms as T # Transformation functions to manipulate images
import torchvision.utils
from torchvision import models
import torchvision.transforms as transforms

from torch.autograd import Variable 
from torch.utils import data
from torch.hub import load_state_dict_from_url


class BaseModel(nn.Module):
    def __init__(self, path):
        super().__init__()
        
#         self.model_rootpath = root_mdls_path # the root path to save trained models
        self.model_rootpath = path
        #self._model_rootpath = 's3://vbdai-share/Peter/dnn_interpretation/code/dnn_invariant/mdls/'
        import os
        if not os.path.exists(self.model_rootpath):
            os.makedirs(self.model_rootpath)


    def forward(self, input):
        # loop over all modules in _layers_list
        out = input
        for i in range(self.layers_list.__len__()):

            if self.layers_list[i].__class__.__name__ != 'Linear':
                out = self.layers_list[i](out)
            else:
                out = self.layers_list[i](out.reshape(out.size(0), -1))


        out = out.reshape(out.size(0), -1)

        return out
    
    def getLayerRange(self, input_data, start, end):
        self.eval()

        out = input_data

#         if layer_ptr == -1:
#             return out

        for i in range(start, end+1):

            if self.layers_list[i].__class__.__name__ != 'Linear':
                out = self.layers_list[i](out)
            else:
                out = self.layers_list[i](out.reshape(out.size(0), -1))

        return out
        
    
    def instance2feature(self, instances, feature_layer):
        if len(instances.shape)==3:
            instances.unsqueeze_(0)
            
        start = 0
        end = feature_layer
            
        return self.getLayerRange(instances, start, end)
    
    def feature2logit(self, features, feature_layer):
        if len(features.shape)==3:
            features.unsqueeze_(0)
            
        start = feature_layer+1
        end = len(self.layers_list)-1
            
        return self.getLayerRange(features, start, end)
    
    def getConfiguration(self,features, feature_layer):
        if len(features.shape)==3:
            features.unsqueeze_(0)
            
        start = feature_layer+1
        end = len(self.layers_list)-1
        out = features
        
        config_list=[]

        for i in range(start, end+1):
            if self.layers_list[i].__class__.__name__ != 'Linear':
                out = self.layers_list[i](out)
            else:
                out = self.layers_list[i](out.reshape(out.size(0), -1))
            config_list.append(torch.where(out>0,1,0))  
        
        return config_list
        
    
    def printLayerInfo(self):
        for i, layer in enumerate(self.layers_list):
            print(str(i)+': '+layer.__class__.__name__)
        
    
    def printLayerOutputSize(self, C=3, H=224, W=224):

        out = torch.zeros((1, C, H, W)).cuda()
        for i in range(0, self.layers_list.__len__()):

            print('Layer ' + str(i-1) + ':', out.shape)
            if i < self._layers_list.__len__() - 1:
                out = self.layers_list[i](out)
            else:
                out = self.layers_list[i](out.reshape(out.size(0), -1))
                
    def saveModel(self, model_savepath = None):
        if model_savepath != None:
            print('Saving model to {}'.format(model_savepath))
            torch.save(self.state_dict(), model_savepath)
        else:
            print('Saving model to {}'.format(self.model_savepath))
            torch.save(self.state_dict(), self.model_savepath)

    def loadModel(self, model_savepath = None):
        if model_savepath != None:
            print('Loading model from {}'.format(model_savepath))
            # NOTE: "cpu" forces to store the loaded temporary weightes on RAM,
            # otherwise they will be persistently stored in the gpu0 memory.
            # self.load_state_dict(torch.load(model_savepath, map_location=lambda storage, loc: storage))
            self.load_state_dict(torch.load(model_savepath, map_location='cpu'))
        else:
            print('Loading model from {}'.format(self.model_savepath))
            self.load_state_dict(torch.load(self.model_savepath, map_location='cpu'))
            
            
class CNN(BaseModel):
    def __init__(self, num_classes, model_path='./', has_bias=True,
                 conv_list=[64, 128, 256], linear_list=[128, 128], p=0, in_channel=1):
        super().__init__(model_path)

        self.has_bias = has_bias    # the bool flag for bias
        self.model_savepath = self.model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self.num_classes = num_classes
        self.p = p
        self.in_channel = in_channel
        self.setLayerList(conv_list, linear_list)
    
    def setLayerList(self, conv_list, linear_list):
        
        layer_list = []
        dim1 = self.in_channel
        for n_channel in conv_list:
            dim2 = n_channel
            layer_list.append(nn.Conv2d(dim1, dim2, kernel_size=3, stride=1, padding=1, bias=self.has_bias)) #layer 0, 3, 6, 9
            layer_list.append(nn.ReLU()) #layer 1, 4, 7,10
            layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2)) # layer2, 5, 8, 11, 14
            dim1 = dim2
    
            
        layer_list.append(nn.AdaptiveAvgPool2d(output_size=(5,5))) #layer 15
        
        dim3 = 5*5*dim2
        for n_linear in linear_list:
            dim4 = n_linear
            layer_list.append(nn.Linear(dim3, dim4)) #layer 16 ...
            layer_list.append(nn.ReLU())#layer 17 ...
            layer_list.append(nn.Dropout(p=self.p)) #layer 18 ...
            dim3 = dim4
        layer_list.append(nn.Linear(dim4,self.num_classes)) #layer 19 ...
        
        self.layers_list = nn.ModuleList(layer_list)
        
class MLP(BaseModel):
    def __init__(self, num_classes, model_path='./', has_bias=True,
                 linear_list=[32, 32], p=0, in_dim=784):
        super().__init__(model_path)

        self.has_bias = has_bias    # the bool flag for bias
        self.model_savepath = self.model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self.num_classes = num_classes
        self.p = p
        self.in_dim = in_dim
        self.setLayerList(linear_list)
    
    def setLayerList(self, linear_list):
        
        layer_list = []
        dim1 = self.in_dim
        
        for n_linear in linear_list:
            dim2 = n_linear
            layer_list.append(nn.Linear(dim1, dim2))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(p=self.p))
            dim1 = dim2
        layer_list.append(nn.Linear(dim2,self.num_classes))
        
        self.layers_list = nn.ModuleList(layer_list)