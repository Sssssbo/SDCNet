import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Comb_ResNet, Pure_ResNet, Jointin_ResNet, Jointout_ResNet, BasicBlock, Bottleneck, GDN_Bottleneck, IN_Bottleneck, IN2_Bottleneck, SNR_Bottleneck, SNR2_Bottleneck, SNR3_Bottleneck
from loss.arcface import ArcFace
from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet50_ibn_a, se_resnet101_ibn_a
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        self.model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        #block = cfg.MODEL.BLOCK
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'Pure_resnet50_GDN':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=GDN_Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])  #
            print('using resnet50 as a backbone')
            print(self.base)
        elif model_name == 'Comb_resnet50_IN':
            self.in_planes = 2048
            self.base = Comb_ResNet(last_stride=last_stride,
                               block=IN_Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])  #
            print('using resnet50 as a backbone')
            print(self.base)
        elif model_name == 'Pure_resnet50_IN2':
            self.in_planes = 2048
            self.base = Pure_ResNet(last_stride=last_stride,
                               block=IN2_Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])  #
        elif model_name == 'Pure_resnet50_IN':
            self.in_planes = 2048
            self.base = Pure_ResNet(last_stride=last_stride,
                               block=IN_Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])  #
            print('using resnet50 as a backbone')
            print(self.base)
        elif model_name == 'Pure_resnet50_SNR':
            self.in_planes = 2048
            self.base = Pure_ResNet(last_stride=last_stride,
                               block=SNR_Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])  #
            print('using resnet50 as a backbone')
            print(self.base)
        elif model_name == 'Pure_resnet50_SNR2':
            self.in_planes = 2048
            self.base = Pure_ResNet(last_stride=last_stride,
                               block=SNR2_Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])  #
            print('using resnet50 as a backbone')
            print(self.base)
        elif model_name == 'Jointin_resnet50_SNR3':
            self.in_planes = 2048
            self.base = Jointin_ResNet(last_stride=last_stride,
                               block=SNR3_Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])  #
            print('using resnet50 as a backbone')
            print(self.base)
        elif model_name == 'Jointout_resnet50_None':
            self.in_planes = 2048
            self.base = Jointout_ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])  #
            print('using resnet50 as a backbone')
            print(self.base)
        elif model_name == 'Jointout_resnet50_IN':
            self.in_planes = 2048
            self.base = Jointout_ResNet(last_stride=last_stride,
                               block=IN_Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])  #
            print('using resnet50 as a backbone')
            print(self.base)
        

        elif model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[2, 2, 2, 2])
            print('using resnet18 as a backbone')
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet34 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using se_resnet50_ibn_a as a backbone')
        elif model_name == 'se_resnet50_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet50_ibn_a(
                last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using se_resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(
                last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(
                last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using se_resnet101_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(
                self.in_planes, self.num_classes, s=30.0, m=0.50)
        else:
            self.classifier = nn.Linear(
                self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        if model_name == 'Jointin_resnet50_SNR3':
            self.classifier = nn.Linear(
                self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier1 = nn.Linear(512, self.num_classes, bias=False)
            self.classifier1.apply(weights_init_classifier)
            self.classifier2 = nn.Linear(512, self.num_classes, bias=False)
            self.classifier2.apply(weights_init_classifier)
            self.classifier3 = nn.Linear(512, self.num_classes, bias=False)
            self.classifier3.apply(weights_init_classifier)
            self.classifier4 = nn.Linear(512, self.num_classes, bias=False)
            self.classifier4.apply(weights_init_classifier)

            self.classifier5 = nn.Linear(1024, self.num_classes, bias=False)
            self.classifier5.apply(weights_init_classifier)
            self.classifier6 = nn.Linear(256, self.num_classes, bias=False)
            self.classifier6.apply(weights_init_classifier)
            self.classifier7 = nn.Linear(256, self.num_classes, bias=False)
            self.classifier7.apply(weights_init_classifier)
            self.classifier8 = nn.Linear(256, self.num_classes, bias=False)
            self.classifier8.apply(weights_init_classifier)
            self.classifier9 = nn.Linear(256, self.num_classes, bias=False)
            self.classifier9.apply(weights_init_classifier)

            self.classifier10 = nn.Linear(512, self.num_classes, bias=False)
            self.classifier10.apply(weights_init_classifier)
            self.classifier11 = nn.Linear(128, self.num_classes, bias=False)
            self.classifier11.apply(weights_init_classifier)
            self.classifier12 = nn.Linear(128, self.num_classes, bias=False)
            self.classifier12.apply(weights_init_classifier)
            self.classifier13 = nn.Linear(128, self.num_classes, bias=False)
            self.classifier13.apply(weights_init_classifier)
            self.classifier14 = nn.Linear(128, self.num_classes, bias=False)
            self.classifier14.apply(weights_init_classifier)

            self.classifier15 = nn.Linear(256, self.num_classes, bias=False)
            self.classifier15.apply(weights_init_classifier)
            self.classifier16 = nn.Linear(64, self.num_classes, bias=False)
            self.classifier16.apply(weights_init_classifier)
            self.classifier17 = nn.Linear(64, self.num_classes, bias=False)
            self.classifier17.apply(weights_init_classifier)
            self.classifier18 = nn.Linear(64, self.num_classes, bias=False)
            self.classifier18.apply(weights_init_classifier)
            self.classifier19 = nn.Linear(64, self.num_classes, bias=False)
            self.classifier19.apply(weights_init_classifier)

        elif 'Jointout' in model_name:
            self.classifier0 = nn.Linear(64, self.num_classes, bias=False)
            self.classifier0.apply(weights_init_classifier)
            self.classifier0_1 = nn.Linear(64, self.num_classes, bias=False)
            self.classifier0_1.apply(weights_init_classifier)
            self.classifier1 = nn.Linear(256, self.num_classes, bias=False)
            self.classifier1.apply(weights_init_classifier)
            self.classifier1_1 = nn.Linear(256, self.num_classes, bias=False)
            self.classifier1_1.apply(weights_init_classifier)
            self.classifier2 = nn.Linear(512, self.num_classes, bias=False)
            self.classifier2.apply(weights_init_classifier)

            self.classifier2_1 = nn.Linear(512, self.num_classes, bias=False)
            self.classifier2_1.apply(weights_init_classifier)
            self.classifier3 = nn.Linear(1024, self.num_classes, bias=False)
            self.classifier3.apply(weights_init_classifier)
            self.classifier3_1 = nn.Linear(1024, self.num_classes, bias=False)
            self.classifier3_1.apply(weights_init_classifier)
            self.classifier4 = nn.Linear(2048, self.num_classes, bias=False)
            self.classifier4.apply(weights_init_classifier)
            self.classifier4_1 = nn.Linear(2048, self.num_classes, bias=False)
            self.classifier4_1.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, camid=None):  # label is unused if self.cos_layer == 'no'
        if self.training and self.model_name == 'Jointin_resnet50_SNR3':
            x, x4_2, x4_1, res4_2, res4_1, x3_3, x3_2, x3_1, res3_2, res3_1, x2_3, x2_2, x2_1, res2_2, res2_1, x1_3, x1_2, x1_1, res1_2, res1_1 = self.base(x, camid)
            global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
            global_feat = global_feat.view(global_feat.shape[0], -1)
            feat = self.bottleneck(global_feat)
            cls_score =  self.classifier(feat)
            fx4_2 = nn.functional.avg_pool2d(x4_2, x4_2.shape[2:4])
            fx4_2 = fx4_2.view(fx4_2.shape[0], -1)
            ax4_2 = self.classifier1(fx4_2)
            fx4_1 = nn.functional.avg_pool2d(x4_1, x4_1.shape[2:4])
            fx4_1 = fx4_1.view(fx4_1.shape[0], -1)
            ax4_1 = self.classifier2(fx4_1)
            fres4_2 = nn.functional.avg_pool2d(res4_2, res4_2.shape[2:4])
            fres4_2 = fres4_2.view(fres4_2.shape[0], -1)
            ares4_2 =  self.classifier3(fres4_2)
            fres4_1 = nn.functional.avg_pool2d(res4_1, res4_1.shape[2:4])
            fres4_1 = fres4_1.view(fres4_1.shape[0], -1)
            ares4_1 =  self.classifier4(fres4_1)

            fx3_3 = nn.functional.avg_pool2d(x3_3, x3_3.shape[2:4])
            fx3_3 = fx3_3.view(fx3_3.shape[0], -1)
            ax3_3 =  self.classifier5(fx3_3)
            fx3_2 = nn.functional.avg_pool2d(x3_2, x3_2.shape[2:4])
            fx3_2 = fx3_2.view(fx3_2.shape[0], -1)
            ax3_2 =  self.classifier6(fx3_2)
            fx3_1 = nn.functional.avg_pool2d(x3_1, x3_1.shape[2:4])
            fx3_1 = fx3_1.view(fx3_1.shape[0], -1)
            ax3_1 =  self.classifier7(fx3_1)
            fres3_2 = nn.functional.avg_pool2d(res3_2, res3_2.shape[2:4])
            fres3_2 = fres3_2.view(fres3_2.shape[0], -1)
            ares3_2 =  self.classifier8(fres3_2)
            fres3_1 = nn.functional.avg_pool2d(res3_1, res3_1.shape[2:4])
            fres3_1 = fres3_1.view(fres3_1.shape[0], -1)
            ares3_1 =  self.classifier9(fres3_1)

            fx2_3 = nn.functional.avg_pool2d(x2_3, x2_3.shape[2:4])
            fx2_3 = fx2_3.view(fx2_3.shape[0], -1)
            ax2_3 =  self.classifier10(fx2_3)
            fx2_2 = nn.functional.avg_pool2d(x2_2, x2_2.shape[2:4])
            fx2_2 = fx2_2.view(fx2_2.shape[0], -1)
            ax2_2 =  self.classifier11(fx2_2)
            fx2_1 = nn.functional.avg_pool2d(x2_1, x2_1.shape[2:4])
            fx2_1 = fx2_1.view(fx2_1.shape[0], -1)
            ax2_1 =  self.classifier12(fx2_1)
            fres2_2 = nn.functional.avg_pool2d(res2_2, res2_2.shape[2:4])
            fres2_2 = fres2_2.view(fres2_2.shape[0], -1)
            ares2_2 =  self.classifier13(fres2_2)
            fres2_1 = nn.functional.avg_pool2d(res2_1, res2_1.shape[2:4])
            fres2_1 = fres2_1.view(fres2_1.shape[0], -1)
            ares2_1 =  self.classifier14(fres2_1)

            fx1_3 = nn.functional.avg_pool2d(x1_3, x1_3.shape[2:4])
            fx1_3 = fx1_3.view(fx1_3.shape[0], -1)
            ax1_3 =  self.classifier15(fx1_3)
            fx1_2 = nn.functional.avg_pool2d(x1_2, x1_2.shape[2:4])
            fx1_2 = fx1_2.view(fx1_2.shape[0], -1)
            ax1_2 =  self.classifier16(fx1_2)
            fx1_1 = nn.functional.avg_pool2d(x1_1, x1_1.shape[2:4])
            fx1_1 = fx1_1.view(fx1_1.shape[0], -1)
            ax1_1 =  self.classifier17(fx1_1)
            fres1_2 = nn.functional.avg_pool2d(res1_2, res1_2.shape[2:4])
            fres1_2 = fres1_2.view(fres1_2.shape[0], -1)
            ares1_2 =  self.classifier18(fres1_2)
            fres1_1 = nn.functional.avg_pool2d(res1_1, res1_1.shape[2:4])
            fres1_1 = fres1_1.view(fres1_1.shape[0], -1)
            ares1_1 =  self.classifier19(fres1_1)
            return cls_score, global_feat, ax4_2, ax4_1, ares4_2, ares4_1, ax3_3, ax3_2, ax3_1, ares3_2, ares3_1, ax2_3, ax2_2, ax2_1, ares2_2, ares2_1, ax1_3, ax1_2, ax1_1, ares1_2, ares1_1
        
        elif 'Jointout' in self.model_name and self.training:
            x0, x1, x2, x3, x4, res0, res1, res2, res3, res4 = self.base(x, camid)
            global_feat = nn.functional.avg_pool2d(x4, x4.shape[2:4])
            global_feat = global_feat.view(global_feat.shape[0], -1)
            feat = self.bottleneck(global_feat)
            cls_score =  self.classifier4(feat)
            res4 = nn.functional.avg_pool2d(res4, res4.shape[2:4])
            res4 = res4.view(res4.shape[0], -1)
            res4 = self.classifier4_1(res4)

            x3 = nn.functional.avg_pool2d(x3, x3.shape[2:4])
            x3 = x3.view(x3.shape[0], -1)
            x3 = self.classifier3_1(x3)
            res3 = nn.functional.avg_pool2d(res3, res3.shape[2:4])
            res3 = res3.view(res3.shape[0], -1)
            res3 = self.classifier3(res3)

            x2 = nn.functional.avg_pool2d(x2, x2.shape[2:4])
            x2 = x2.view(x2.shape[0], -1)
            x2 = self.classifier2(x2)
            res2 = nn.functional.avg_pool2d(res2, res2.shape[2:4])
            res2 = res2.view(res2.shape[0], -1)
            res2 = self.classifier2_1(res2)

            x1 = nn.functional.avg_pool2d(x1, x1.shape[2:4])
            x1 = x1.view(x1.shape[0], -1)
            x1 = self.classifier1(x1)
            res1 = nn.functional.avg_pool2d(res1, res1.shape[2:4])
            res1 = res1.view(res1.shape[0], -1)
            res1 = self.classifier1_1(res1)

            x0 = nn.functional.avg_pool2d(x0, x0.shape[2:4])
            x0 = x0.view(x0.shape[0], -1)
            x0 = self.classifier0(x0)
            res0 = nn.functional.avg_pool2d(res0, res0.shape[2:4])
            res0 = res0.view(res0.shape[0], -1)
            res0 = self.classifier0_1(res0)
            return global_feat, x0, x1, x2, x3, cls_score, res0, res1, res2, res3, res4
        
        x = self.base(x, camid)
        # print(x.shape)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        # print(global_feat.shape)
        # print(x.shape)
        # for convert to onnx, kernel size must be from x.shape[2:4] to a constant [20,20]
        #global_feat = nn.functional.avg_pool2d(x, [16, 16])
        # flatten to (bs, 2048), global_feat.shape[0]
        global_feat = global_feat.view(global_feat.shape[0], -1)
        feat = self.bottleneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)

        # for i in param_dict:
        # print(i)#change by sb
        # self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
