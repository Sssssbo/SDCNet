import math

import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GDN_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(GDN_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(
            planes, affine=False, track_running_stats=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(
            planes, affine=False, track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(
            planes * 4, affine=False, track_running_stats=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.in1 = nn.InstanceNorm2d(planes)
        self.in2 = nn.InstanceNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out1 = torch.zeros_like(out)
        if self.training == True:
            #print("training with gdn block")
            out1[:8] = self.bn1_0(out[:8])
            out1[8:16] = self.bn1_0(out[8:16])
            out1[16:] = self.bn1_0(out[16:])
        else:
            #print("test for gdn block")
            out1 = self.in1(out)
        out = self.bn1(out1)
        out = self.relu(out)

        out = self.conv2(out)
        out1 = torch.zeros_like(out)
        if self.training == True:
            out1[:8] = self.bn2_0(out[:8])
            out1[8:16] = self.bn2_0(out[8:16])
            out1[16:] = self.bn2_0(out[16:])
        else:
            out1 = self.in1(out)
        out = self.bn2(out1)
        out = self.relu(out)

        out = self.conv3(out)
        out1 = torch.zeros_like(out)
        if self.training == True:
            out1[:8] = self.bn3_0(out[:8])
            out1[8:16] = self.bn3_0(out[8:16])
            out1[16:] = self.bn3_0(out[16:])
        else:
            out1 = self.in2(out)
        out = self.bn3(out1)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IN_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(IN_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.in1_0 = nn.InstanceNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.in2_0 = nn.InstanceNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.in3_0 = nn.InstanceNorm2d(planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1_0(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2_0(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.in3_0(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IN2_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(IN2_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.in1_0 = nn.InstanceNorm2d(planes)
        self.conv1_1 =  nn.Sequential(
            nn.Conv2d(planes * 2, planes, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(planes), nn.ReLU(inplace=True)
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.in2_0 = nn.InstanceNorm2d(planes)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(planes * 2, planes, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(planes), nn.ReLU(inplace=True)
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.in3_0 = nn.InstanceNorm2d(planes * 4)
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 4, kernel_size=1, bias=False), nn.BatchNorm2d(planes * 4)
        )
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x1 = self.conv1(x)
        out1 = self.in1_0(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        x1 = self.conv1_1(torch.cat((out1,x1),1))

        x2 = self.conv2(x1)
        out2 = self.in2_0(x2)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        x2 = self.conv2_1(torch.cat((out2,x2),1))

        x3 = self.conv3(x2)
        out3 = self.in3_0(x3)
        out3 = self.bn3(out3)
        out3 = self.relu(out3)
        x3 = self.conv3_1(torch.cat((out3,x3),1))

        if self.downsample is not None:
            residual = self.downsample(residual)

        x3 += residual
        x3 = self.relu(x3)

        return x3

class SNR_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SNR_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.in1_0 = nn.InstanceNorm2d(planes)
        self.conv1_1 = nn.Conv2d(planes, planes, kernel_size=3,
                                 padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.in2_0 = nn.InstanceNorm2d(planes)
        self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=3,
                                 padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.in3_0 = nn.InstanceNorm2d(planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x1 = self.conv1(x)
        out1 = self.in1_0(x1)
        res1 = x1 - out1
        res1 = self.conv1_1(res1)
        res1 = self.bn1_1(res1)
        res1 = self.relu(res1)
        x1 = self.bn1(x1)
        x1 = out1 + res1
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        out2 = self.in2_0(x2)
        res2 = x2 - out2
        res2 = self.conv2_1(res2)
        res2 = self.bn2_1(res2)
        res2 = self.relu(res2)
        x2 = self.bn2(x2)
        x2 = out2 + res2
        x2 = self.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x3 += residual
        x3 = self.relu(x3)

        return x3

class SNR2_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SNR2_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.in1_0 = nn.InstanceNorm2d(planes)
        self.conv1_1 = nn.Conv2d(planes, planes, kernel_size=3,
                                 padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.in2_0 = nn.InstanceNorm2d(planes)
        self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=3,
                                 padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.in3_0 = nn.InstanceNorm2d(planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)

    def forward(self, x):
        residual = x

        x1 = self.conv1(x)
        out1 = self.in1_0(x1)
        res1 = x1 - out1
        res1 = self.conv1_1(res1)
        res1 = self.bn1_1(res1)
        res1 = self.relu(res1)
        x1 = out1 + res1
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        out2 = self.in2_0(x2)
        if self.stride == 2: res1 = self.maxpool(res1)
        res2 = x2 - out2 + res1
        res2 = self.conv2_1(res2)
        res2 = self.bn2_1(res2)
        res2 = self.relu(res2)
        x2 = out2 + res2
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x3 += residual
        x3 = self.relu(x3)

        return x3

class SNR3_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SNR3_Bottleneck, self).__init__()
        self.in1 = nn.InstanceNorm2d(planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1_1 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)

    def forward(self, x, x_2=None, x_1=None, r2=None, r1=None):
        if type(x) is tuple:
            # print(len(x))
            x_2 = x[1]
            x_1 = x[2]
            r2 = x[3]
            r1 = x[4]
            x = x[0]

        residual = x
        x1 = self.conv1(x)
        out1 = self.in1(x1)
        res1 = x1 - out1
        res1 = self.conv1_1(res1)
        res1 = self.bn1_1(res1)
        res1 = self.relu(res1)
        # print(out1.shape)
        # print(res1.shape)
        # print(x1.shape)
        x1 = out1 + res1
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        out2 = self.in1(x2)
        res2 = x2 - out2
        res2 = self.conv2_1(res2)
        res2 = self.bn2_1(res2)
        res2 = self.relu(res2)
        x2 = out2 + res2
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x3 += residual
        x3 = self.relu(x3)
        if x_2 is not None:  x2 = x2 + x_2
        if x_1 is not None:  x1 = x1 + x_1
        if r2 is not None:  res2 = res2 + r2
        if r1 is not None:  res1 = res1 + r1
        '''
        print(x3.shape)
        print(x2.shape)
        print(x1.shape)
        print(res2.shape)
        print(res1.shape)
        '''
        if self.stride == 2: 
            x1 = self.maxpool(x1) 
            res1 = self.maxpool(res1)
        return x3, x2, x1, res2, res1

class SNR4_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SNR4_Bottleneck, self).__init__()
        self.in1 = nn.InstanceNorm2d(planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1_1 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2_1 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)

    def forward(self, x, x_2=None, x_1=None, r2=None, r1=None):
        if type(x) is tuple:
            # print(len(x))
            x_2 = x[1]
            x_1 = x[2]
            r2 = x[3]
            r1 = x[4]
            x = x[0]

        residual = x
        x1 = self.conv1(x)
        out1 = self.in1(x1)
        res1 = x1 - out1
        res1 = self.conv1_1(res1)
        res1 = self.bn1_1(res1)
        res1 = self.relu(res1)
        # print(out1.shape)
        # print(res1.shape)
        # print(x1.shape)
        x1 = out1 + res1
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        out2 = self.in1(x2)
        res2 = x2 - out2
        res2 = self.conv2_1(res2)
        res2 = self.bn2_1(res2)
        res2 = self.relu(res2)
        x2 = out2 + res2
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x3 += residual
        x3 = self.relu(x3)
        if x_2 is not None:  x2 = x2 + x_2
        if x_1 is not None:  x1 = x1 + x_1
        if r2 is not None:  res2 = res2 + r2
        if r1 is not None:  res1 = res1 + r1
        '''
        print(x3.shape)
        print(x2.shape)
        print(x1.shape)
        print(res2.shape)
        print(res1.shape)
        '''
        if self.stride == 2: 
            x1 = self.maxpool(x1) 
            res1 = self.maxpool(res1)
        return x3, x2, x1, res2, res1


# --------------------------------- resnet-----------------------------------

class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck,  frozen_stages=-1, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        print(block)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.frozen_stages = frozen_stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            print('layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x, camid=None):
        x = self.conv1(x)
        x = self.bn1(x)

        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# ---------------------------------Comb resnet-----------------------------------

class Comb_ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck,  frozen_stages=-1, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        print(block)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in1 = nn.InstanceNorm2d(64)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1)
        )
        self.in2 = nn.InstanceNorm2d(256)
        self.bn2_1 = nn.BatchNorm2d(256)
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1)
        )
        self.in3 = nn.InstanceNorm2d(512)
        self.bn3_1 = nn.BatchNorm2d(512)
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1)
        )
        self.in4 = nn.InstanceNorm2d(1024)
        self.bn4_1 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )

        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.frozen_stages = frozen_stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            print('layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x, camid=None):
        x = self.conv1(x)
        x = self.bn1(x)

        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)
        xin = self.in1(x)
        xin = self.bn1_1(xin)
        xin = self.relu(xin)
        x = self.conv2(torch.cat((xin,x),1))
        x = self.layer1(x)
        xin = self.in2(x)
        xin = self.bn2_1(xin)
        xin = self.relu(xin)
        x = self.conv3(torch.cat((xin,x),1))
        x = self.layer2(x)
        xin = self.in3(x)
        xin = self.bn3_1(xin)
        xin = self.relu(xin)
        x = self.conv4(torch.cat((xin,x),1))
        x = self.layer3(x)
        xin = self.in4(x)
        xin = self.bn4_1(xin)
        xin = self.relu(xin)
        x = self.conv5(torch.cat((xin,x),1))
        x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# ---------------------------------Pure resnet-----------------------------------
class Pure_ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck,  frozen_stages=-1, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        print(block)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.frozen_stages = frozen_stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            print('layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x, camid=None):
        x = self.conv1(x)
        x = self.bn1(x)
        #print(camid)

        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)
        if False:
            x,_,_,_,_ = self.layer1(x)
            x,_,_,_,_ = self.layer2(x)
            x,_,_,_,_ = self.layer3(x)
            x,_,_,_,_ = self.layer4(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# ---------------------------------jointin resnet-----------------------------------

class Jointin_ResNet(nn.Module):
    def __init__(self, last_stride=2, block=SNR3_Bottleneck,  frozen_stages=-1, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        print(block)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.in1 = nn.InstanceNorm2d(64)

        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.frozen_stages = frozen_stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            print('layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x, camid=None):
        x = self.conv1(x)
        x0 = self.in1(x)
        '''
        res0 = x - x0
        res0 = self.conv1_1(res0)
        res0 = self.bn1_1(res0)
        x0 = x0 + res0
        '''
        x0 = self.bn1(x0)

        # x = self.relu(x)    # add missed relu
        x0 = self.maxpool(x0)
        x1_3, x1_2, x1_1, res1_2, res1_1 = self.layer1(x0)
        x2_3, x2_2, x2_1, res2_2, res2_1 = self.layer2(x1_3)
        x3_3, x3_2, x3_1, res3_2, res3_1 = self.layer3(x2_3)
        x4_3, x4_2, x4_1, res4_2, res4_1 = self.layer4(x3_3)
        if self.training:
            return x4_3, x4_2, x4_1, res4_2, res4_1, x3_3, x3_2, x3_1, res3_2, res3_1, x2_3, x2_2, x2_1, res2_2, res2_1, x1_3, x1_2, x1_1, res1_2, res1_1
        else: 
            return x4_3

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# ---------------------------------jointout resnet-----------------------------------

class Jointout_ResNet(nn.Module):
    def __init__(self, last_stride=2, block=SNR3_Bottleneck,  frozen_stages=-1, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        print(block)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_res = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=1)
        )
        self.in1 = nn.InstanceNorm2d(64)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.in2 = nn.InstanceNorm2d(256)
        self.bn2_1 = nn.BatchNorm2d(256)
        self.bn2_0 = nn.BatchNorm2d(256)
        self.in3 = nn.InstanceNorm2d(512)
        self.bn3_1 = nn.BatchNorm2d(512)
        self.bn3_0 = nn.BatchNorm2d(512)
        self.in4 = nn.InstanceNorm2d(1024)
        self.bn4_1 = nn.BatchNorm2d(1024)
        self.bn4_0 = nn.BatchNorm2d(1024)
        self.in5 = nn.InstanceNorm2d(2048)
        self.bn5_1 = nn.BatchNorm2d(2048)
        self.bn5_0 = nn.BatchNorm2d(2048)

        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.frozen_stages = frozen_stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv2_res = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace = True),
            nn.Conv2d(128, 256, kernel_size=1)
        )
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv3_res = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace = True),
            nn.Conv2d(256, 512, kernel_size=1)
        )
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv4 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv4_res = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace = True),
            nn.Conv2d(512, 1024, kernel_size=1)
        )
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.conv5 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv5_res = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace = True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace = True),
            nn.Conv2d(1024, 2048, kernel_size=1)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            print('layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x, camid=None):
        x = self.conv1(x)
        x0 = self.in1(x)
        res0 = x - x0
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        res0 = self.conv1_res(res0)
        x0 = x0 + res0
        x0 = self.bn1_1(x0)

        # x = self.relu(x)    # add missed relu
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        px1 = self.conv2(x1)
        x1 = self.in2(px1)
        res1 = px1 - x1
        x1 = self.bn2_0(x1) 
        x1 = self.relu(x1)
        res1 = self.conv2_res(res1)
        x1 = x1 + res1
        x1 = self.bn2_1(x1) 
        x1 = self.relu(x1)

        x2 = self.layer2(x1)
        px2 = self.conv3(x2)
        x2 = self.in3(px2)
        res2 = px2 - x2
        x2 = self.bn3_0(x2) 
        x2 = self.relu(x2)
        res2 = self.conv3_res(res2)
        x2 = x2 + res2
        x2 = self.bn3_1(x2) 
        x2 = self.relu(x2)

        x3 = self.layer3(x2)
        px3 = self.conv4(x3)
        x3 = self.in4(px3)
        res3 = px3 - x3
        x3 = self.bn4_0(x3) 
        x3 = self.relu(x3)
        res3 = self.conv4_res(res3)
        x3 = x3 + res3
        x3 = self.bn4_1(x3) 
        x3 = self.relu(x3)

        x4 = self.layer4(x3)
        px4 = self.conv5(x4)
        x4 = self.in5(px4)
        res4 = px4 - x4
        x4 = self.bn5_0(x4) 
        x4 = self.relu(x4)
        res4 = self.conv5_res(res4)
        x4 = x4 + res4
        x4 = self.bn5_1(x4) 
        x4 = self.relu(x4)


        if self.training:
            return x0, x1, x2, x3, x4, res0, res1, res2, res3, res4
        else: 
            return x4

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()