import torch
import torch.nn as nn
import numpy as np
import torchvision
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
    
    def getLayerRangeR(self, input_data, start, end_L, end_sub_L):
        self.eval()

        out = input_data

#         if layer_ptr == -1:
#             return out

        for i in range(start, end_L+1):
            if i < end_L:
                if self.layers_list[i].__class__.__name__ != 'Linear':
                  out = self.layers_list[i](out)
                else:
                  out = self.layers_list[i](out.reshape(out.size(0), -1))
            else:
                for j in range(end_sub_L+1):
                    if self.layers_list[i][j].__class__.__name__ != 'Linear':
                        out = self.layers_list[i][j](out)
                    else:
                        out = self.layers_list[i][j](out.reshape(out.size(0), -1))

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

    def instance2featconfig(self, instances, layer):
        if len(instances.shape)==3:
            instances.unsqueeze_(0)
        
            
        if (self.layers_list[layer+2].__class__.__name__ != 'ReLU') & (self.layers_list[layer+1].__class__.__name__ != 'Sequential'):
            print('The layer was not correctly selected')
            return
            
        feat = self.getLayerRange(instances, 0, layer)
        config = self.getLayerRangeR(feat, layer+1, layer+1, 0)
        config = torch.where(config>0,1,0)
        return feat, config

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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(BaseModel):
    def __init__(self, block, layers, linear_layers = [], cutBlock = 4, num_classes_=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, model_name='resnet50', model_path='./', pretrained=False):
        
        super().__init__(model_path)
        
        self.resnet_init(block, layers, num_classes_=1000, cutBlock = -1, zero_init_residual=zero_init_residual,
                 groups=groups, width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation,
                 norm_layer=norm_layer)
        
        self.model_name = model_name
        self.model_savepath = self.model_rootpath + self.model_name + '.mdl'  # default path to save the model
        self.num_classes = num_classes_

        if pretrained:
            self.load_pretrained()
        self.setLayerList()

        feature_dim = 64*(2**(cutBlock-1))

        if linear_layers == []:
            self.classifiers = [nn.Linear(feature_dim * block.expansion, num_classes_)]
            self.setLayerList(cutBlock=cutBlock)

        else:

            classifiers = [nn.Linear(feature_dim * block.expansion,linear_layers[0])]

            for i in range(len(linear_layers)-1):
                classifiers.append(nn.ReLU())
                classifiers.append(nn.Dropout(p=0.5))               
                classifiers.append(nn.Linear(linear_layers[i],linear_layers[i+1]))
            
            if len(linear_layers) == 1:
                classifiers.append(nn.ReLU())
                classifiers.append(nn.Dropout(p=0.5)) 
                classifiers.append(nn.Linear(linear_layers[0],num_classes_))
            else:
                classifiers.append(nn.Linear(linear_layers[i+1],num_classes_))

            self.classifiers = classifiers
            self.setLayerList(cutBlock=cutBlock)
    
    def resnet_init(self, block, layers, num_classes_=1000, cutBlock = -1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes_)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def load_pretrained(self):
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
            'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
            'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
            'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        }
#         model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        state_dict = load_state_dict_from_url(model_urls[self.model_name], progress=True)
        self.load_state_dict(state_dict)
        
    
    def setLayerList(self, cutBlock = -1):
        layer_list = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4, 
                      self.avgpool, self.fc]
        if cutBlock != -1:
            self.layers_list = nn.ModuleList(layer_list[ :cutBlock+4] + [self.avgpool] + self.classifiers)
        else:
            self.layers_list = nn.ModuleList(layer_list)
        
    