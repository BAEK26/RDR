from torch import nn
from torch import Tensor
from typing import Callable, Any, Optional, List
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.hub import load_state_dict_from_url
import math

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
    
    
    def instance2featconfig(self, instances, layer):
        if len(instances.shape)==3:
            instances.unsqueeze_(0)
            
        # if self.layers_list[layer+2].__class__.__name__ != 'ReLU':
        #     print('The layer was not correctly selected.')
        #     return
            
        feat = self.getLayerRange(instances, 0, layer)
        config = self.getLayerRange(feat, layer+1, layer+1)
        config = torch.where(config>0,1,0)
        return feat, config
    
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


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        # stride는 반드시 1 또는 2이어야 하므로 조건을 걸어 둡니다.
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # expansion factor를 이용하여 channel을 확장합니다.
        hidden_dim = int(round(inp * expand_ratio))
        # stride가 1인 경우에만 residual block을 사용합니다.
        # skip connection을 사용하는 경우 input과 output의 크기가 같아야 합니다.
        self.use_res_connect = (self.stride == 1) and (inp == oup)

        # Inverted Residual 연산
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # point-wise convolution
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # depth-wise convolution
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # point-wise linear convolution
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])            
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # use_res_connect인 경우만 connection을 연결합니다.
        # use_res_connect : stride가 1이고 input과 output의 채널 수가 같은 경우 True
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(BaseModel):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        model_name = 'mobilenet_v2',
        model_path='./',
        pretrained=False

    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        #super(MobileNetV2, self).__init__(model_path)
        super().__init__(model_path)
        self.model_name = model_name
        self.model_savepath = self.model_rootpath + self.model_name + '.mdl'  # default path to save the model
 
        

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280
        
        # t : expansion factor
        # c : output channel의 수
        # n : 반복 횟수
        # s : stride
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        

        # Inverted Residual Block을 생성합니다.
        # features에 feature들의 정보를 차례대로 저장합니다.
        for t, c, n, s in inverted_residual_setting:
            # width multiplier는 layer의 채널 수를 일정 비율로 줄이는 역할을 합니다.
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if pretrained:
            self.load_pretrained()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self.layers_list = nn.ModuleList(list(self.features)+[self.avgpool]+list(self.classifier))
        
        # if pretrained:
        #     self.load_pretrained()
 
    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        # x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.avgpool(x).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def load_pretrained(self):

        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth', progress=True)
        self.load_state_dict(state_dict, strict =False)

        # model_urls = {
        # 'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        # }

        # state_dict = load_state_dict_from_url(model_urls[self.model_name], progress=True)
        # self.load_state_dict(state_dict,strict=False)