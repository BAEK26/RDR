import torch
import torch.nn as nn
import numpy as np
import torchvision


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
    
    
    def instance2featconfig(self, instances, layer):
        if len(instances.shape)==3:
            instances.unsqueeze_(0)
            
        if self.layers_list[layer+2].__class__.__name__ != 'ReLU':
            print('The layer was not correctly selected.')
            return
            
        feat = self.getLayerRange(instances, 0, layer)
        config = self.getLayerRange(feat, layer+1, layer+1)
        config = torch.where(config>0,1,0)
        return feat, config
    
    
    def feature2logit(self, features, feature_layer):
        if len(features.shape)==3:
            features.unsqueeze_(0)
            
        start = feature_layer+1
        end = len(self.layers_list)-1
            
        return self.getLayerRange(features, start, end)
    
    def getLayerNo(self, feature_layer=-1, tar_layer_type=['Linear']):
        start = feature_layer+1
        end = len(self.layers_list)-1
        layer_dict = dict()
        for i in range(start, end+1):
            layer = self.layers_list[i]
            layer_name = layer.__class__.__name__
            if layer_name in tar_layer_type:
                layer_dict[i] = layer_name
        return layer_dict
    
    def getConfiguration(self, features, feature_layer, tar_layer_type=['Linear']):
        if len(features.shape)==3:
            features.unsqueeze_(0)

        start = feature_layer+1
        end = len(self.layers_list)-1
        out = features

        config_dict =  dict()

        for i in range(start, end+1):
            '''forward pass'''
            layer = self.layers_list[i]
            layer_name = layer.__class__.__name__
            if layer_name == 'Linear':
                out = layer(out.reshape(out.size(0), -1))
            else:
                out = layer(out)
            '''save configuration'''
            if layer_name in tar_layer_type:
                config_dict[i] = torch.where(out>0,1,0)

        return config_dict
        
    
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
            

class VGG19(BaseModel):
    def __init__(self, num_classes, model_path='./', has_bias=True, num_blocks=5,
                 feature_size=5, classifier_dim=[4096, 4096]):
        super().__init__(model_path)

        self.has_bias = has_bias    # the bool flag for bias
        self.model_savepath = self.model_rootpath + self.__class__.__name__ + '.mdl'  # default path to save the model
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.feature_size = feature_size
        self.classifier_dim = classifier_dim
        self.setLayerList()

    def setLayerList(self):
        # set the layers of the model in a list
        block_layer = [4,9,18,27,36]
        self.num_channels = [64,128,256,512,512][self.num_blocks-1]
        layers_list = [
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=self.has_bias),   # layer 0
            nn.ReLU(),  # layer 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 2
            nn.ReLU(),  # layer 3
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 4

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 5
            nn.ReLU(),  # layer 6
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 7
            nn.ReLU(),  # layer 8
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 9

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 10
            nn.ReLU(),  # layer 11
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 12
            nn.ReLU(),  # layer 13
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 14
            nn.ReLU(),  # layer 15
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 16
            nn.ReLU(),  # layer 17
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 18

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 19
            nn.ReLU(),  # layer 20
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 21
            nn.ReLU(),  # layer 22
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 23
            nn.ReLU(),  # layer 24
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 25
            nn.ReLU(),  # layer 26
            nn.MaxPool2d(kernel_size=2, stride=2),  # layer 27

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 28
            nn.ReLU(),  # layer 29
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 30
            nn.ReLU(),  # layer 31
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 32
            nn.ReLU(),  # layer 33
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=self.has_bias),  # layer 34
            nn.ReLU(),  # layer 35
            nn.MaxPool2d(kernel_size=2, stride=2)  # layer 36
        ][:block_layer[self.num_blocks-1]]
        
        layers_list.append(nn.AdaptiveAvgPool2d(output_size=(self.feature_size,self.feature_size)))
        all_feat_size = self.num_channels*(self.feature_size**2)
        
        classifier_dim = self.classifier_dim
        classifier_dim.insert(0,all_feat_size)
        for i in range(len(classifier_dim)-1):
            layers_list.append(nn.Linear(classifier_dim[i], classifier_dim[i+1]))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(p=0.5))
        layers_list.append(nn.Linear(classifier_dim[-1], self.num_classes))
        self.layers_list = nn.ModuleList(layers_list)
#         self.layers_list = nn.ModuleList([

#             nn.AdaptiveAvgPool2d(output_size=(7,7)),

#             nn.Linear(25088, classifier_dim), # layer 0
#             nn.ReLU(), # layer 1
#             nn.Dropout(p=0.5), # layer 2
#             nn.Linear(classifier_dim, classifier_dim), # layer 3
#             nn.ReLU(), # layer 4
#             nn.Dropout(p=0.5), # layer 5
#             nn.Linear(classifier_dim, self.num_classes) # layer 6

#         ])
    
    def load_pretrained(self):
        torch_model_pretrained = torchvision.models.vgg19(pretrained=True)
        with torch.no_grad():
            if self.num_blocks>=1:
                self.layers_list[0].weight = torch_model_pretrained.features[0].weight
                self.layers_list[0].bias = torch_model_pretrained.features[0].bias
                self.layers_list[2].weight = torch_model_pretrained.features[2].weight
                self.layers_list[2].bias = torch_model_pretrained.features[2].bias
            if self.num_blocks>=2:
                self.layers_list[5].weight = torch_model_pretrained.features[5].weight
                self.layers_list[5].bias = torch_model_pretrained.features[5].bias
                self.layers_list[7].weight = torch_model_pretrained.features[7].weight
                self.layers_list[7].bias = torch_model_pretrained.features[7].bias
            if self.num_blocks>=3:
                self.layers_list[10].weight = torch_model_pretrained.features[10].weight
                self.layers_list[10].bias = torch_model_pretrained.features[10].bias
                self.layers_list[12].weight = torch_model_pretrained.features[12].weight
                self.layers_list[12].bias = torch_model_pretrained.features[12].bias
                self.layers_list[14].weight = torch_model_pretrained.features[14].weight
                self.layers_list[14].bias = torch_model_pretrained.features[14].bias
                self.layers_list[16].weight = torch_model_pretrained.features[16].weight
                self.layers_list[16].bias = torch_model_pretrained.features[16].bias
            if self.num_blocks>=4:
                self.layers_list[19].weight = torch_model_pretrained.features[19].weight
                self.layers_list[19].bias = torch_model_pretrained.features[19].bias
                self.layers_list[21].weight = torch_model_pretrained.features[21].weight
                self.layers_list[21].bias = torch_model_pretrained.features[21].bias
                self.layers_list[23].weight = torch_model_pretrained.features[23].weight
                self.layers_list[23].bias = torch_model_pretrained.features[23].bias
                self.layers_list[25].weight = torch_model_pretrained.features[25].weight
                self.layers_list[25].bias = torch_model_pretrained.features[25].bias
            if self.num_blocks>=5:
                self.layers_list[28].weight = torch_model_pretrained.features[28].weight
                self.layers_list[28].bias = torch_model_pretrained.features[28].bias
                self.layers_list[30].weight = torch_model_pretrained.features[30].weight
                self.layers_list[30].bias = torch_model_pretrained.features[30].bias
                self.layers_list[32].weight = torch_model_pretrained.features[32].weight
                self.layers_list[32].bias = torch_model_pretrained.features[32].bias
                self.layers_list[34].weight = torch_model_pretrained.features[34].weight
                self.layers_list[34].bias = torch_model_pretrained.features[34].bias
#                 if self.classifier_dim == [4096, 4096]:
#                     self.layers_list[38].weight = torch_model_pretrained.classifier[0].weight
#                     self.layers_list[38].bias = torch_model_pretrained.classifier[0].bias
#                     self.layers_list[41].weight = torch_model_pretrained.classifier[3].weight
#                     self.layers_list[41].bias = torch_model_pretrained.classifier[3].bias
            
                