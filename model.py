import torch
import torch.nn.functional as F
from torch import nn

from resnext import ResNeXt101


class R3Net(nn.Module):
    def __init__(self):
        super(R3Net, self).__init__()
        res50 = ResNeXt101()
        self.layer0 = res50.layer0
        self.layer1 = res50.layer1
        self.layer2 = res50.layer2
        self.layer3 = res50.layer3
        self.layer4 = res50.layer4

        self.reduce_low = nn.Sequential(
            nn.Conv2d(64 + 256 + 512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce_high = nn.Sequential(
            nn.Conv2d(1024 + 2048, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            _ASPP(256)
        )

        self.predict0 = nn.Conv2d(256, 1, kernel_size=1)
        self.predict1 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict2 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict3 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict4 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict5 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict6 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x, label = None):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        l0_size = layer0.size()[2:]
        reduce_low = self.reduce_low(torch.cat((
            layer0,
            F.interpolate(layer1, size=l0_size, mode='bilinear', align_corners=True),
            F.interpolate(layer2, size=l0_size, mode='bilinear', align_corners=True)), 1))
        reduce_high = self.reduce_high(torch.cat((
            layer3,
            F.interpolate(layer4, size=layer3.size()[2:], mode='bilinear', align_corners=True)), 1))
        reduce_high = F.interpolate(reduce_high, size=l0_size, mode='bilinear', align_corners=True)

        predict0 = self.predict0(reduce_high)
        predict1 = self.predict1(torch.cat((predict0, reduce_low), 1)) + predict0
        predict2 = self.predict2(torch.cat((predict1, reduce_high), 1)) + predict1
        predict3 = self.predict3(torch.cat((predict2, reduce_low), 1)) + predict2
        predict4 = self.predict4(torch.cat((predict3, reduce_high), 1)) + predict3
        predict5 = self.predict5(torch.cat((predict4, reduce_low), 1)) + predict4
        predict6 = self.predict6(torch.cat((predict5, reduce_high), 1)) + predict5

        predict0 = F.interpolate(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict5 = F.interpolate(predict5, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict6 = F.interpolate(predict6, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict0, predict1, predict2, predict3, predict4, predict5, predict6
        return F.sigmoid(predict6)



#--------------------------------------------------------------------------------------------
class SDCNet(nn.Module):
    def __init__(self, num_classes):
        super(SDCNet, self).__init__()
        res50 = ResNeXt101()
        self.layer0 = res50.layer0
        self.layer1 = res50.layer1
        self.layer2 = res50.layer2
        self.layer3 = res50.layer3
        self.layer4 = res50.layer4

        self.reducex = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            _ASPP(256)
        )
        self.reduce5 = nn.Sequential(
            nn.Conv2d(64 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce8 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce9 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce10 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        # --------------extra module---------------
        self.reduce3_0 = nn.Sequential(
            nn.Conv2d(1024 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce3_1 = nn.Sequential(
            nn.Conv2d(1024 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce3_2 = nn.Sequential(
            nn.Conv2d(1024 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce3_3 = nn.Sequential(
            nn.Conv2d(1024 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce3_4 = nn.Sequential(
            nn.Conv2d(1024 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce2_0 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce2_1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce2_2 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce2_3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce2_4 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )

        self.reduce1_0 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce1_1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce1_2 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce1_3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce1_4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )

        self.reduce0_0 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce0_1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce0_2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce0_3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce0_4 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        # self.predict0 = nn.Conv2d(256, 1, kernel_size=1)
        self.predict1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict9 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict10 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.pre4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 2, kernel_size=1)
        )
        self.pre3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 2, kernel_size=1)
        )
        self.pre2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 2, kernel_size=1)
        )
        self.pre1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 2, kernel_size=1)
        )
        self.reducex_1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reducex_2 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reducex_3 = nn.Sequential(
            nn.Conv2d(1024 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc0 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, c):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        l0_size = layer0.size()[2:]
        l1_size = layer1.size()[2:]
        l2_size = layer2.size()[2:]
        l3_size = layer3.size()[2:]

        F1 = self.reducex(layer4)
        p4 = self.pre4(F1)
        p4 = F.interpolate(p4, size=x.size()[2:], mode='bilinear', align_corners=True)
        F0_4 = F.interpolate(F1, size=l3_size, mode='bilinear', align_corners=True)

        F0_3 = self.reducex_3(torch.cat((F0_4, layer3), 1))
        p3 = self.pre3(F0_3)
        p3 = F.interpolate(p3, size=x.size()[2:], mode='bilinear', align_corners=True)
        F0_3 = F.interpolate(F0_3, size=l2_size, mode='bilinear', align_corners=True)

        F0_2 = self.reducex_2(torch.cat((F0_3, layer2), 1))
        p2 = self.pre2(F0_2)
        p2 = F.interpolate(p2, size=x.size()[2:], mode='bilinear', align_corners=True)
        F0_2 = F.interpolate(F0_2, size=l1_size, mode='bilinear', align_corners=True)

        F0_1 = self.reducex_1(torch.cat((F0_2, layer1), 1))
        p1 = self.pre1(F0_1)
        p1 = F.interpolate(p1, size=x.size()[2:], mode='bilinear', align_corners=True)
        p5 = p4 + p3 + p2 + p1

        #saliency detect
        predict1 = self.predict1(F1)
        predict1 = F.interpolate(predict1, size=l3_size, mode='bilinear', align_corners=True)
        F1 = F.interpolate(F1, size=l3_size, mode='bilinear', align_corners=True)

        F2 = F1[:, :, :, :].clone().detach()
        for i in range(len(c)):
            if c[i] == 0:
                F2[i, :, :, :] = self.reduce3_0(
                    torch.cat((F1[i, :, :, :].unsqueeze(0), layer3[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 1:
                F2[i, :, :, :] = self.reduce3_1(
                    torch.cat((F1[i, :, :, :].unsqueeze(0), layer3[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 2:
                F2[i, :, :, :] = self.reduce3_2(
                    torch.cat((F1[i, :, :, :].unsqueeze(0), layer3[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 3:
                F2[i, :, :, :] = self.reduce3_3(
                    torch.cat((F1[i, :, :, :].unsqueeze(0), layer3[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 4:
                F2[i, :, :, :] = self.reduce3_4(
                    torch.cat((F1[i, :, :, :].unsqueeze(0), layer3[i, :, :, :].unsqueeze(0)), 1))
        predict2 = self.predict2(F2) + predict1
        predict2 = F.interpolate(predict2, size=l2_size, mode='bilinear', align_corners=True)
        F2 = F.interpolate(F2, size=l2_size, mode='bilinear', align_corners=True)

        F3 = F2[:, :, :, :].clone().detach()
        for i in range(len(c)):
            if c[i] == 0:
                F3[i, :, :, :] = self.reduce2_0(
                    torch.cat((F2[i, :, :, :].unsqueeze(0), layer2[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 1:
                F3[i, :, :, :] = self.reduce2_1(
                    torch.cat((F2[i, :, :, :].unsqueeze(0), layer2[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 2:
                F3[i, :, :, :] = self.reduce2_2(
                    torch.cat((F2[i, :, :, :].unsqueeze(0), layer2[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 3:
                F3[i, :, :, :] = self.reduce2_3(
                    torch.cat((F2[i, :, :, :].unsqueeze(0), layer2[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 4:
                F3[i, :, :, :] = self.reduce2_4(
                    torch.cat((F2[i, :, :, :].unsqueeze(0), layer2[i, :, :, :].unsqueeze(0)), 1))
        predict3 = self.predict3(F3) + predict2
        predict3 = F.interpolate(predict3, size=l1_size, mode='bilinear', align_corners=True)
        F3 = F.interpolate(F3, size=l1_size, mode='bilinear', align_corners=True)

        F4 = F3[:, :, :, :].clone().detach()
        for i in range(len(c)):
            if c[i] == 0:
                F4[i, :, :, :] = self.reduce1_0(
                    torch.cat((F3[i, :, :, :].unsqueeze(0), layer1[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 1:
                F4[i, :, :, :] = self.reduce1_1(
                    torch.cat((F3[i, :, :, :].unsqueeze(0), layer1[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 2:
                F4[i, :, :, :] = self.reduce1_2(
                    torch.cat((F3[i, :, :, :].unsqueeze(0), layer1[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 3:
                F4[i, :, :, :] = self.reduce1_3(
                    torch.cat((F3[i, :, :, :].unsqueeze(0), layer1[i, :, :, :].unsqueeze(0)), 1))
            elif c[i] == 4:
                F4[i, :, :, :] = self.reduce1_4(
                    torch.cat((F3[i, :, :, :].unsqueeze(0), layer1[i, :, :, :].unsqueeze(0)), 1))
        predict4 = self.predict4(F4) + predict3

        F5 = self.reduce5(torch.cat((F4, layer0), 1))
        predict5 = self.predict5(F5) + predict4

        F0 = F4[:, :, :, :].clone().detach()
        for i in range(len(c)):
            if c[i] == 0:
                F0[i, :, :, :] = self.reduce0_0(layer0[i, :, :, :].unsqueeze(0))
            elif c[i] == 1:
                F0[i, :, :, :] = self.reduce0_1(layer0[i, :, :, :].unsqueeze(0))
            elif c[i] == 2:
                F0[i, :, :, :] = self.reduce0_2(layer0[i, :, :, :].unsqueeze(0))
            elif c[i] == 3:
                F0[i, :, :, :] = self.reduce0_3(layer0[i, :, :, :].unsqueeze(0))
            elif c[i] == 4:
                F0[i, :, :, :] = self.reduce0_4(layer0[i, :, :, :].unsqueeze(0))

        F1 = F.interpolate(F1, size=l1_size, mode='bilinear', align_corners=True)
        F2 = F.interpolate(F2, size=l1_size, mode='bilinear', align_corners=True)
        F6 = self.reduce6(torch.cat((F0, F5), 1))
        F7 = self.reduce7(torch.cat((F0, F4), 1))
        F8 = self.reduce8(torch.cat((F0, F3), 1))
        F9 = self.reduce9(torch.cat((F0, F2), 1))
        F10 = self.reduce10(torch.cat((F0, F1), 1))
        predict6 = self.predict6(F6) + predict5
        predict7 = self.predict7(F7) + predict6
        predict8 = self.predict8(F8) + predict7
        predict9 = self.predict9(F9) + predict8
        predict10 = self.predict10(F10) + predict9
        predict11 = predict6 + predict7 + predict8 + predict9 + predict10

        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict5 = F.interpolate(predict5, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict6 = F.interpolate(predict6, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict7 = F.interpolate(predict7, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict8 = F.interpolate(predict8, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict9 = F.interpolate(predict9, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict10 = F.interpolate(predict10, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict11 = F.interpolate(predict11, size=x.size()[2:], mode='bilinear', align_corners=True)


        if self.training:
            return p5, p4, p3, p2, p1, predict1, predict2, predict3, predict4, predict5, predict6, predict7, predict8, predict9, predict10, predict11
        return F.sigmoid(predict11)

#----------------------------------------------------------------------------------------


class _ASPP(nn.Module):
    def __init__(self, in_dim):
        super(_ASPP, self).__init__()
        down_dim = in_dim // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear',
                           align_corners=True)
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
