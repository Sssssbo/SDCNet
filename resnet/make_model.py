from .resnet import ResNet, BasicBlock, Bottleneck
import torch
from torch import nn
from .config import resnet50_path

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        net = ResNet(last_stride=2,
                    block=Bottleneck, frozen_stages=False,
                    layers=[3, 4, 6, 3])
        net.load_param(resnet50_path)

        self.layer0 = net.layer0
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4
    
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


class ResNet50_BIN(nn.Module):
    def __init__(self):
        super(ResNet50_BIN, self).__init__()
        net = ResNet(last_stride=2,
                    block=IN_Bottleneck, frozen_stages=False,
                    layers=[3, 4, 6, 3])
        net.load_param(resnet50_path)

        self.layer0 = net.layer0
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4
    
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


class ResNet50_LowIN(nn.Module):
    def __init__(self):
        super(ResNet50_LowIN, self).__init__()
        net = ResNet_LowIN(last_stride=2,
                    block=Bottleneck, frozen_stages=False,
                    layers=[3, 4, 6, 3])
        net.load_param(resnet50_path)

        self.layer0 = net.layer0
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4
    
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
